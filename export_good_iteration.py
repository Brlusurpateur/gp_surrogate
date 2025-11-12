#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
from datetime import datetime
import psutil
import gc
import json
import plotly.express as px
import argparse
import metrics_core as mx
from typing import List, Dict, Tuple, Set
from metrics_core import compute_median_daily_pnl, compute_skew_daily_pnl
from config import load_thresholds, Thresholds, ANN_FACTOR_DAILY
from config import hard_soft_thresholds, score_weights  # seuils multi-obj & poids score
try:
    from config import SHARPE_CAP
except Exception:
    SHARPE_CAP = 5.0  # fallback robuste

# ──────────────────────────────────────────────────────────────────────────────
# Switchs
DO_EXTRACTION = 1
chargement_good_and_bad_logs = 0
CREATION_HISTOGRAM = 0
TRACE_HYPER = 0
extract_very_good_iteration = 0


# Limites
MAX_LOGS_PER_ID = 5000
ROW_SUBBATCH_SIZE = 1000

# Chemins DB
FOLDER_PATH = "/Users/brandonmoncoucut/Desktop/Najas_king/log_sqlite/backtest_pipeline"
SRC_DB  = os.path.join(FOLDER_PATH, "suggest_strat_test.db")
GOOD_DB = os.path.join(FOLDER_PATH, "good_iterations.db")         # passe SOFT (pool GP/BO)
BAD_DB  = os.path.join(FOLDER_PATH, "bad_iterations.db")
BEST_DB = os.path.join(FOLDER_PATH, "filtered_full_copy.db")      # passe HARD (vitrine/portfolio)

# Seuil minimum de trades pour considérer un backtest "évaluable" au stade extraction
# Résolution adaptative : 1) ENV EXTRACT_MIN_TRADES, 2) HARD.MIN_TRADES via config, 3) défaut 30.
try:
    from config import hard_soft_thresholds  # lit les seuils adaptatifs si disponibles
except Exception:
    hard_soft_thresholds = None  # type: ignore

def _resolve_extract_min_trades() -> int:
    """
    Résout EXTRACT_MIN_TRADES avec priorité:
      1) ENV EXTRACT_MIN_TRADES (int/float accepté)
      2) HARD.MIN_TRADES issu de hard_soft_thresholds() (adaptatif si activé)
      3) défaut: 30
    Borne à >= 1 pour sûreté.
    """
    # 1) ENV direct
    _env = os.environ.get("EXTRACT_MIN_TRADES", "").strip()
    if _env:
        try:
            v = int(float(_env))
            return max(1, v)
        except Exception:
            pass

    # 2) Adaptatif via HARD.MIN_TRADES
    if hard_soft_thresholds is not None:
        try:
            _HARD, _SOFT = hard_soft_thresholds()
            v = int(_HARD.get("MIN_TRADES", 30))
            return max(1, v)
        except Exception:
            pass

    # 3) Défaut
    return 30

EXTRACT_MIN_TRADES = _resolve_extract_min_trades()
print(f"[EXTRACT] MIN_TRADES (extraction) = {EXTRACT_MIN_TRADES}")

# ── Adaptive: table de décision pour logger N_eff (post-HARD) ─────────────────
def _adaptive_ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS adaptive_decisions (
            ts_utc TEXT NOT NULL,
            source TEXT NOT NULL,
            n_eff  INTEGER NOT NULL,
            meta   TEXT
        );
    """)
    conn.commit()

def _adaptive_log_n_eff(conn: sqlite3.Connection, n_eff: int, source: str, meta: dict | None = None) -> None:
    _adaptive_ensure_table(conn)
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    payload = None
    try:
        payload = json.dumps(meta or {}, ensure_ascii=False)
    except Exception:
        payload = None
    conn.execute(
        "INSERT INTO adaptive_decisions (ts_utc, source, n_eff, meta) VALUES (?,?,?,?)",
        (ts, str(source), int(n_eff), payload)
    )
    conn.commit()

# ──────────────────────────────────────────────────────────────────────────────
# Filtres "hard cut" : paramétrables par ENV (via config.load_thresholds)
# et surchargés par arguments CLI si fournis.

def _as_fraction(x: float) -> float:
    """
    Convertit un seuil exprimé en pourcentage (ex: 58) en fraction (0.58).
    Laisse inchangé s'il est déjà sous forme fractionnaire (ex: 0.58).
    """
    try:
        return x / 100.0 if x is not None and x > 1.5 else x
    except Exception:
        return x

def _parse_cli_thresholds(base: Thresholds) -> Thresholds:
    """
    Parse les overrides CLI et retourne un objet Thresholds fusionné
    (les valeurs None ne remplacent pas le base).
    """
    parser = argparse.ArgumentParser(
        description="Extraction & filtrage strict des bonnes itérations (consistency-first)."
    )
    parser.add_argument("--min_trades", type=int, help="Min trades éligibilité (def: ENV ou 30)")
    parser.add_argument("--min_pct_green", type=float, help="Min %% de jours verts (def: ENV ou 0.58). Accepte 58 ou 0.58")
    parser.add_argument("--min_sharpe_d", type=float, help="Sharpe/jour annualisé min (def: ENV ou 1.20)")
    parser.add_argument("--min_profit_factor", type=float, help="Profit factor min (def: ENV ou 1.40)")
    parser.add_argument("--max_mdd_abs", type=float, help="Max drawdown absolu (def: ENV ou 0.12). Accepte 12 ou 0.12")
    parser.add_argument("--max_top5_share", type=float, help="Part des 5 meilleurs trades max (def: ENV ou 0.25)")
    parser.add_argument("--max_ulcer", type=float, help="Ulcer index max (def: ENV ou 0.05)")
    parser.add_argument("--min_median_daily_pnl", type=float, help="Médiane journalière min (def: ENV ou 0.0)")
    parser.add_argument("--min_skew_daily_pnl", type=float, help="Skewness journalière min (def: ENV ou 0.0)")
    # On ne modifie pas ici FOLDER_PATH/SRC_DB... qui restent configurés plus haut
    args, _unknown = parser.parse_known_args()

    def coalesce(cli_val, env_val):
        return env_val if (cli_val is None) else cli_val

    return Thresholds(
        min_trades=coalesce(args.min_trades, base.min_trades),
        min_pct_green=_as_fraction(coalesce(args.min_pct_green, base.min_pct_green)),
        min_sharpe_d=coalesce(args.min_sharpe_d, base.min_sharpe_d),
        min_profit_factor=coalesce(args.min_profit_factor, base.min_profit_factor),
        max_mdd_abs=_as_fraction(coalesce(args.max_mdd_abs, base.max_mdd_abs)),
        max_top5_share=coalesce(args.max_top5_share, base.max_top5_share),
        max_ulcer=coalesce(args.max_ulcer, base.max_ulcer),
        min_median_daily_pnl=coalesce(args.min_median_daily_pnl, base.min_median_daily_pnl),
        min_skew_daily_pnl=coalesce(args.min_skew_daily_pnl, base.min_skew_daily_pnl),
    )

# Charge les seuils depuis ENV puis applique les overrides CLI
_THRESH_ENV = load_thresholds()
T = _parse_cli_thresholds(_THRESH_ENV)


# ──────────────────────────────────────────────────────────────────────────────
# Utils
def apply_sqlite_optimizations(conn: sqlite3.Connection):
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def rankpct(s: pd.Series) -> pd.Series:
    """
    Rang percentile (0..1) robuste :
      - ignore NaN (renvoie NaN pour ces lignes),
      - method='average' pour stabilité.
    """
    try:
        return s.rank(pct=True, method="average")
    except Exception:
        return pd.Series(np.nan, index=s.index)

def tag_pareto_nondominated(df: pd.DataFrame,
                            cols_up: List[str],
                            cols_down: List[str],
                            flag_col: str = "pareto_nd") -> pd.DataFrame:
    """
    Marque les points 'non-dominés' (front de Pareto) sur les colonnes :
      - cols_up   : à maximiser
      - cols_down : à minimiser
    Complexité O(n^2) (OK pour quelques milliers d'obs).
    """
    if df.empty:
        df[flag_col] = False
        return df
    vals_up = df[cols_up].to_numpy()
    vals_dn = df[cols_down].to_numpy()
    n = len(df)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        # j domine i si (tous >= en cols_up et tous <= en cols_down) et strict sur au moins une dim
        ge_up  = (vals_up >= vals_up[i]).all(axis=1)
        le_dn  = (vals_dn <= vals_dn[i]).all(axis=1)
        strict = (vals_up >  vals_up[i]).any(axis=1) | (vals_dn <  vals_dn[i]).any(axis=1)
        dominates = ge_up & le_dn & strict
        dominates[i] = False
        if dominates.any():
            nd[i] = False
    out = df.copy()
    out[flag_col] = nd
    return out

def load_valid_backtest_ids(conn: sqlite3.Connection) -> tuple[set, dict]:
    """
    Retourne l’ensemble des backtest_id considérés valides côté KPI :
      - COALESCE(is_valid,1)=1
      - COALESCE(flag_sharpe_outlier,0)=0
      - (pas de filtre cap ici : le cap est géré par clipping côté modèle)
    Si la table/colonnes n’existent pas, retourne (set(), {"mode": "passthrough"})
    pour que l’appelant sache qu’il ne peut pas filtrer ici.
    """
    if not table_exists(conn, "kpi_by_backtest"):
        return set(), {"mode": "passthrough", "reason": "no_kpi_table"}

    # Détecte les colonnes
    cur = conn.execute("PRAGMA table_info(kpi_by_backtest);")
    cols = {row[1] for row in cur.fetchall()}
    need = {"sharpe_d_365"}
    if not need.issubset(cols):
        return set(), {"mode": "passthrough", "reason": "missing_sharpe_365"}

    has_is_valid = "is_valid" in cols
    has_outlier  = "flag_sharpe_outlier" in cols

    f1 = "AND COALESCE(is_valid,1)=1" if has_is_valid else ""
    f2 = "AND COALESCE(flag_sharpe_outlier,0)=0" if has_outlier else ""

    # Stats avant/après
    tot = conn.execute(
        "SELECT COUNT(DISTINCT backtest_id) FROM kpi_by_backtest WHERE sharpe_d_365 IS NOT NULL;"
    ).fetchone()[0] or 0
    ok  = conn.execute(
        f"""SELECT COUNT(DISTINCT backtest_id)
            FROM kpi_by_backtest
            WHERE sharpe_d_365 IS NOT NULL {f1} {f2};"""
    ).fetchone()[0] or 0

    # IDs valides
    rows = conn.execute(
        f"""SELECT DISTINCT backtest_id
            FROM kpi_by_backtest
            WHERE sharpe_d_365 IS NOT NULL {f1} {f2};"""
    ).fetchall()
    valid_ids = {r[0] for r in rows}

    return valid_ids, {
        "mode": "filter",
        "total_kpi_ids": int(tot),
        "kept_kpi_ids": int(ok),
        "drop_pct": (100.0 * (tot - ok) / tot) if tot else 0.0,
        "has_is_valid": has_is_valid,
        "has_outlier": has_outlier,
        "cap": None,  # cap retiré du filtre
    }

def load_kpis_hard_filtered(conn: sqlite3.Connection, cap: float) -> pd.DataFrame:
    """
    Charge depuis kpi_by_backtest les colonnes nécessaires et applique :
      - is_valid == 1 (si dispo)
      - flag_sharpe_outlier == 0 (si dispo)
      - ABS(sharpe_d_365) ≤ cap
      - HARD cuts : nb_trades, mdd_abs, ulcer, top5 (si dispo)
    Retourne un DataFrame prêt pour le scoring.
    """
    HARD, _SOFT = hard_soft_thresholds()

    # Détection colonnes dispo
    cur = conn.execute("PRAGMA table_info(kpi_by_backtest);")
    cols = {r[1] for r in cur.fetchall()}

    need_any = {"backtest_id", "sharpe_d_365", "profit_factor", "pct_green_days", "max_drawdown"}
    if not {"backtest_id", "sharpe_d_365"}.issubset(cols):
        return pd.DataFrame(columns=list(need_any) + ["ulcer_index", "top5_share", "nb_trades"])

    has_is_valid = "is_valid" in cols
    has_outlier  = "flag_sharpe_outlier" in cols
    has_ulcer    = "ulcer_index" in cols
    has_top5     = "top5_share" in cols
    has_trades   = "nb_trades" in cols

    sel_cols = ["backtest_id", "sharpe_d_365", "profit_factor", "pct_green_days", "max_drawdown"]
    if has_ulcer:  sel_cols.append("ulcer_index")
    if has_top5:   sel_cols.append("top5_share")
    if has_trades: sel_cols.append("nb_trades")
    if has_is_valid:  sel_cols.append("is_valid")
    if has_outlier:   sel_cols.append("flag_sharpe_outlier")

    q = f"SELECT {', '.join(sel_cols)} FROM kpi_by_backtest WHERE sharpe_d_365 IS NOT NULL"
    if has_is_valid:
        q += " AND COALESCE(is_valid,1)=1"
    if has_outlier:
        q += " AND COALESCE(flag_sharpe_outlier,0)=0"
    q += f" AND ABS(sharpe_d_365) <= {float(cap):.6f}"

    df = pd.read_sql_query(q, conn)
    if df.empty:
        return df

    # Renommages/sûretés
    df = df.drop_duplicates(subset=["backtest_id"]).reset_index(drop=True)
    df["mdd_abs"]  = df["max_drawdown"].abs()
    df["pct_green"] = df["pct_green_days"].astype(float)

    # HARD (seulement) — on ignore les NaN → False
    mask = np.ones(len(df), dtype=bool)
    if has_trades:
        mask &= (df["nb_trades"].fillna(-1) >= int(HARD["MIN_TRADES"]))
    mask &= (df["mdd_abs"].fillna(np.inf) <= float(HARD["MAX_MDD"]))
    if has_ulcer:
        mask &= (df["ulcer_index"].fillna(np.inf) <= float(HARD["MAX_ULCER"]))
    if has_top5:
        mask &= (df["top5_share"].fillna(np.inf) <= float(HARD["MAX_TOP5"]))

    kept = df[mask].copy()
    dropped = len(df) - len(kept)
    print(f"🧹 HARD filter: kept={len(kept)}/{len(df)} ({(len(kept)/max(1,len(df)))*100:.1f}%), dropped={dropped}")
    return kept

def load_kpis_soft_candidates(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    SOFT pool pour le GP/BO : on applique uniquement les garde-fous data + seuils SOFT.
    ❌ Pas de filtre |Sharpe| <= SHARPE_CAP ici (le cap est géré par clipping côté modèle).
    ✅ Seuils lus depuis cfg.SOFT (et donc contrôlables par env/RELAX_SOFT).
    """
    HARD, SOFT = hard_soft_thresholds()  # lit cfg.SOFT dynamiquement

    base = """
        SELECT *
        FROM kpi_by_backtest
        WHERE COALESCE(is_valid,1)=1
          AND COALESCE(flag_sharpe_outlier,0)=0
          AND sharpe_d_365 IS NOT NULL
    """
    # Seuils SOFT (uniquement ces 3 critères de perf)
    where_soft = f"""
          AND sharpe_d_365 >= {float(SOFT['MIN_SHARPE'])}
          AND profit_factor >= {float(SOFT['MIN_PF'])}
          AND pct_green_days >= {float(SOFT['MIN_GREEN'])}
    """

    q = base + where_soft + ";"
    df = pd.read_sql(q, conn)

    # --- Adaptive log: N_soft (taille du pool SOFT calculé en SQL) ---------------
    try:
        _adaptive_log_n_eff(
            conn,
            n_eff=int(len(df)),
            source="export_good_iteration:soft_pool_size_sql",
            meta={"cap": float(SHARPE_CAP),
                "soft_min": {
                    "sharpe": float(SOFT.get("MIN_SHARPE", float("nan"))),
                    "pf":     float(SOFT.get("MIN_PF", float("nan"))),
                    "green":  float(SOFT.get("MIN_GREEN", float("nan"))),
                }}
        )
    except Exception as _e:
        print(f"[ADAPTIVE][WARN] N_soft (SQL) non journalisé: {type(_e).__name__}: {_e}")
    # -----------------------------------------------------------------------------

    try:
        import logging
        logging.getLogger(__name__).info(
            "❖ SOFT pass: valid & !outlier (no cap filter) — cfg.SOFT=%s → %d lignes",
            SOFT, len(df),
        )
    except Exception:
        pass

    return df


def build_daily_pnl_matrix(conn: sqlite3.Connection, ids: List[str]) -> pd.DataFrame:
    """
    Construit une matrice (index=date, colonnes=backtest_id) des PnL journaliers.
    Utilise trades.pnl_net et agrège par jour (UTC). Remplit les dates manquantes par 0
    pour stabiliser les corrélations (post-événement).
    """
    if not ids:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(ids))
    q = f"SELECT backtest_id, timestamp, exit_time, pnl_net FROM trades WHERE backtest_id IN ({placeholders})"
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(q, ids)]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # choisir la meilleure colonne temporelle dispo
    ts_col = "timestamp" if "timestamp" in df.columns else ("exit_time" if "exit_time" in df.columns else None)
    if ts_col is None:
        return pd.DataFrame()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col, "pnl_net"])
    df["date"] = df[ts_col].dt.floor("D")

    # agrégation journalière par backtest
    daily = df.groupby(["backtest_id", "date"])["pnl_net"].sum().reset_index()
    pivot = daily.pivot(index="date", columns="backtest_id", values="pnl_net").sort_index()
    # alignement (remplissage) — neutre : 0
    pivot = pivot.fillna(0.0)
    return pivot

def greedy_diversified_selection(corr: pd.DataFrame,
                                 ranking_ids: List[str],
                                 k: int,
                                 corr_thr: float) -> List[str]:
    """
    Greedy : parcourt ranking_ids (déjà triés décroissant), ajoute si
    la corrélation absolue avec tous les déjà retenus est < corr_thr.
    """
    selected: List[str] = []
    for bid in ranking_ids:
        ok = True
        for s in selected:
            c = corr.loc[bid, s] if (bid in corr.index and s in corr.columns) else 0.0
            if abs(float(c)) >= corr_thr:
                ok = False
                break
        if ok:
            selected.append(bid)
            if len(selected) >= k:
                break
    return selected

def to_epoch_ms(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return (ts.view("int64") // 1_000_000).astype("Int64")  # nullable int

def dedup_logs(df_logs: pd.DataFrame) -> pd.DataFrame:
    if df_logs.empty:
        return df_logs
    return df_logs.sort_values(["backtest_id", "timestamp"]).drop_duplicates(
        subset=["backtest_id", "timestamp"], keep="last"
    )

def _table_columns(conn: sqlite3.Connection, name: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({name});")
    return [r[1] for r in cur.fetchall()]

def upsert_kpi_for_ids(conn_src: sqlite3.Connection,
                       conn_dst: sqlite3.Connection,
                       backtest_ids: list[str]) -> int:
    """
    Copie (DELETE+INSERT) les lignes KPI pour les backtest_id fournis
    depuis conn_src.kpi_by_backtest → conn_dst.kpi_by_backtest.

    - Intersecte les colonnes pour rester robuste.
    - Copie *toutes* les iterations trouvées pour chaque backtest_id.
    Retourne le nombre de lignes insérées.
    """
    if not backtest_ids:
        return 0
    if not table_exists(conn_src, "kpi_by_backtest") or not table_exists(conn_dst, "kpi_by_backtest"):
        return 0

    src_cols = _table_columns(conn_src, "kpi_by_backtest")
    dst_cols = set(_table_columns(conn_dst, "kpi_by_backtest"))
    cols = [c for c in src_cols if c in dst_cols]
    if not cols:
        return 0

    placeholders = ",".join(["?"] * len(backtest_ids))
    sel = f"SELECT {', '.join(cols)} FROM kpi_by_backtest WHERE backtest_id IN ({placeholders})"
    df = pd.read_sql_query(sel, conn_src, params=backtest_ids)
    if df.empty:
        return 0

    # upsert naïf = delete then insert
    with conn_dst:
        conn_dst.executemany(
            "DELETE FROM kpi_by_backtest WHERE backtest_id = ?;",
            [(bid,) for bid in set(df["backtest_id"].tolist())]
        )
        insert_ph = ",".join(["?"] * len(cols))
        conn_dst.executemany(
            f"INSERT INTO kpi_by_backtest ({', '.join(cols)}) VALUES ({insert_ph});",
            list(map(tuple, df[cols].itertuples(index=False, name=None)))
        )

        # Index utiles (idempotent)
        conn_dst.execute("CREATE INDEX IF NOT EXISTS idx_kpi_backtest ON kpi_by_backtest(backtest_id);")
        if "sharpe_d_365" in cols:
            conn_dst.execute("CREATE INDEX IF NOT EXISTS idx_kpi_backtest_sharpe365 ON kpi_by_backtest(backtest_id, sharpe_d_365);")
    return len(df)

def extraction_data():
    """
    Pipeline d’extraction et de filtrage “consistency-first”.

    Étapes:
    1) Lit la table 'trades' depuis SRC_DB (en chunks), agrège par backtest_id.
    2) Calcule des KPI standardisés via metrics_core (Sharpe/Sortino/Ulcer/MDD, etc.).
    3) Applique des *filtres durs* paramétrables (ENV/CLI) sur ces KPI.
    4) Écrit les trades des backtests retenus dans GOOD_DB; échantillonne des “bad” dans BAD_DB.
    5) (Optionnel) Stream les logs associés, avec normalisation de timestamp (epoch-ms).
    6) Sauvegarde un échantillon 'bad' en Parquet pour analyses ultérieures.

    Remarques:
    - Seules les **clés standardisées** sont conservées dans 'stats'.
    - Les filtres utilisent 'sharpe_d_365' (crypto 24/7) et 'max_drawdown' (pas d’alias).
    """

    conn_src  = sqlite3.connect(SRC_DB)
    conn_good = sqlite3.connect(GOOD_DB)
    conn_bad  = sqlite3.connect(BAD_DB)
    conn_best = sqlite3.connect(BEST_DB)

    try:
        # Source prête ?
        if not table_exists(conn_src, "trades"):
            print("⚠️ La base source ne contient pas la table 'trades'. Extraction impossible pour l’instant.")
            return

        # Charger les IDs valides côté KPI (si disponible)
        valid_ids, meta = load_valid_backtest_ids(conn_src)
        if meta.get("mode") == "filter":
            print(
                f"🧹 Filtre KPI (SRC_DB): gardés={meta['kept_kpi_ids']}/{meta['total_kpi_ids']} "
                f"({100.0 - meta['drop_pct']:.1f}% kept, {meta['drop_pct']:.1f}% dropped) | "
                f"is_valid={meta['has_is_valid']} outlier={meta['has_outlier']} (pas de cap en filtre; cap=clipping modèle)"
            )
        else:
            print(f"ℹ️ Pas de filtre KPI amont ({meta.get('reason')}). "
                f"On gardera le cap **uniquement** en clipping côté modèle.")

        # Prépare destination (copie schéma si possible)
        for conn in (conn_good, conn_bad, conn_best):
            apply_sqlite_optimizations(conn)
            for tbl in ("trades", "logs"):
                if not table_exists(conn, tbl):
                    src_schema = conn_src.execute(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tbl,)
                    ).fetchone()
                    if src_schema and src_schema[0]:
                        conn.execute(src_schema[0])
                    else:
                        # Schéma minimal fallback
                        if tbl == "trades":
                            conn.execute("""
                                CREATE TABLE trades(
                                    backtest_id TEXT,
                                    timestamp   INTEGER,
                                    exit_time   INTEGER,
                                    pnl_net     REAL,
                                    balance     REAL
                                );
                            """)
                        else:
                            conn.execute("""
                                CREATE TABLE logs(
                                    backtest_id TEXT,
                                    timestamp   INTEGER,
                                    position    REAL,
                                    reason_refus TEXT,
                                    nb_signaux_valides REAL
                                );
                            """)
            conn.commit()

        # Lecture en chunks
        try:
            chunks = pd.read_sql_query("SELECT * FROM trades", conn_src, chunksize=100_000)
        except Exception as e:
            print(f"❌ Lecture SQL trades échouée : {e}")
            return

        trades_grouped = {}
        stats = {}
        MIN_TRADES = EXTRACT_MIN_TRADES

        print("🔍 Lecture & agrégation des trades...")
        for chunk in chunks:
            # tri léger par backtest pour de meilleures perfs groupby
            chunk = chunk.sort_values("backtest_id")
            for backtest_id, group in chunk.groupby("backtest_id"):
                # Si KPI présents → rejette les IDs invalides/outliers/capés
                if valid_ids and (backtest_id not in valid_ids):
                    continue

                n_trades = int(group.shape[0])  # ✅ compter les lignes, pas pnl.dropna()
                if n_trades < MIN_TRADES:
                    continue

                # Tri temporel pour KPI (on ne transforme pas en place ce qu'on insèrera)
                ts_col = "timestamp" if "timestamp" in group.columns else ("exit_time" if "exit_time" in group.columns else None)
                group_kpi = group
                if ts_col is not None:
                    group_kpi = group_kpi.copy()
                    group_kpi[ts_col] = pd.to_datetime(group_kpi[ts_col], errors="coerce", utc=True)
                    group_kpi = group_kpi.sort_values(ts_col)

                # KPI “consistency-first” (API standard : df_trades/df_logs/price_col/ts_col)
                try:
                    k = mx.compute_intraday_consistency_kpis(
                        df_trades=group_kpi,
                        df_logs=None,                 # pas de logs ici → métriques dépendantes = NaN
                        price_col="price",            # si absent en DB, certaines métriques seront NaN
                        timestamp_col=ts_col or "timestamp",
                        fee_rate=0.0,                 # on ne recalcule pas les frais ici
                        ann_factor=ANN_FACTOR_DAILY   # << annualisation daily→annual (crypto 24/7)
                    )

                except Exception as e:
                    print(f"[WARN] KPI failed for {backtest_id}: {e}")
                    continue

                # Nouveaux KPI “daily” : médiane journalière & skewness journalière
                # (s’appuient sur timestamp + pnl_net ; la fonction gère le resample par jour)
                try:
                    daily_median = float(compute_median_daily_pnl(group_kpi))
                except Exception:
                    daily_median = np.nan

                try:
                    daily_skew = float(compute_skew_daily_pnl(group_kpi))
                except Exception:
                    daily_skew = np.nan

                def _f(x):
                    return float(x) if (x is not None and np.isfinite(x)) else np.nan

                def _ensure_sharpe_365(kdict: dict, ann_factor: int) -> float:
                    """
                    Retourne un Sharpe annualisé (crypto 24/7) :
                      1) 'sharpe_d_365' si présent,
                      2) sinon 'sharpe_daily_ann' si présent,
                      3) sinon (mean_daily_return / vol_daily_return) * sqrt(ann_factor) si dispo,
                      4) sinon NaN.
                    """
                    if kdict is None:
                        return np.nan
                    if kdict.get("sharpe_d_365") is not None:
                        return _f(kdict["sharpe_d_365"])
                    if kdict.get("sharpe_daily_ann") is not None:
                        return _f(kdict["sharpe_daily_ann"])
                    m, s = kdict.get("mean_daily_return"), kdict.get("vol_daily_return")
                    if (m is not None) and (s is not None) and (s not in (0, 0.0)):
                        try:
                            return float(m) / float(s) * (ANN_FACTOR_DAILY ** 0.5)
                        except Exception:
                            return np.nan
                    return np.nan

                # Écriture avec clés standardisées (crypto 24/7)
                sharpe365 = _ensure_sharpe_365(k, ANN_FACTOR_DAILY)

                # Garde-fou local si pas de KPI-table : hard-reject si Sharpe hors cap
                if not valid_ids:
                    if np.isfinite(sharpe365) and abs(sharpe365) > SHARPE_CAP:
                        # on n'écrit pas cette itération dans 'good' pour ne pas polluer
                        continue

                mdd_std   = _f(k.get("max_drawdown"))
                cvar_d95  = _f(k.get("cvar_95_d"))

                # NOTE: on fige ici **uniquement** les clés standardisées pour éviter
                # tout doublon/alias silencieux. Toute la suite du code s'aligne sur
                # ces noms (ex: 'sharpe_d_365', 'max_drawdown', 'cvar_95_d').
                stats[backtest_id] = {
                    "nb_trades": n_trades,
                    "win_rate": _f(k.get("win_rate")),
                    "profit_factor": _f(k.get("profit_factor")),
                    "sharpe_d_365": sharpe365,
                    "max_drawdown": mdd_std,
                    "pct_green_days": _f(k.get("pct_green_days")),
                    "sortino_d_252": _f(k.get("sortino_d_252")),  # tu peux décliner une version 365 plus tard
                    "ulcer_index": _f(k.get("ulcer_index")),
                    "cvar_95_d": cvar_d95,
                    "top5_share": _f(k.get("top5_share")),
                    "median_daily_pnl": _f(daily_median),
                    "skew_daily_pnl": _f(daily_skew),
                }

                # Préparer lignes pour insertion (timestamps → epoch-ms)
                group_ins = group.copy()
                if "timestamp" in group_ins.columns:
                    group_ins["timestamp"] = to_epoch_ms(group_ins["timestamp"])
                if "exit_time" in group_ins.columns:
                    group_ins["exit_time"] = to_epoch_ms(group_ins["exit_time"])

                if backtest_id not in trades_grouped:
                    trades_grouped[backtest_id] = []
                trades_grouped[backtest_id].extend(group_ins.to_records(index=False))

        print(f"✅ {len(trades_grouped)} backtests traités (avec au moins {EXTRACT_MIN_TRADES} trades)")

        # ──────────────────────────────────────────────────────────────────────────────
        # Double passe SOFT/HARD
        # - SOFT → pool d’apprentissage (good_iterations.db) : filtres larges mais sains
        #   (garde-fous + SOFT minima de perf) pour maximiser la diversité utile au GP/BO.
        # - HARD → vitrine/portfolio (filtered_full_copy.db) : filtres durs (garde-fous + T strict).
        #   NB: T provient de load_thresholds() / CLI (min_sharpe_d, min_profit_factor, min_pct_green, etc.)
        HARD, SOFT = hard_soft_thresholds()

        soft_trades, hard_trades = [], []
        bad_trades, valid_stats_soft, valid_stats_hard = [], [], []

        for backtest_id, trade_rows in trades_grouped.items():
            m = stats[backtest_id]

            # Garde-fous de base
            n_tr = int(m.get("nb_trades", 0))
            base_ok = (n_tr >= int(HARD["MIN_TRADES"]))  # min trades
            max_dd_val = m.get("max_drawdown", np.nan)
            c_mdd   = (np.isfinite(max_dd_val) and abs(max_dd_val) <= float(HARD["MAX_MDD"]))
            c_top5  = (m.get("top5_share", np.nan) <= float(HARD["MAX_TOP5"]))
            ulcer_v = m.get("ulcer_index", np.nan)
            c_ulcer = (ulcer_v <= float(HARD["MAX_ULCER"])) if np.isfinite(ulcer_v) else True

            # Minima "daily consistency"
            c_med  = (m.get("median_daily_pnl", np.nan) >  T.min_median_daily_pnl)
            c_skew = (m.get("skew_daily_pnl", np.nan)   >  T.min_skew_daily_pnl)

            # SOFT minima (perf)
            cS_soft = (m.get("sharpe_d_365", np.nan)      >= float(SOFT["MIN_SHARPE"]))
            cPF_soft= (m.get("profit_factor", np.nan)     >= float(SOFT["MIN_PF"]))
            cG_soft = (m.get("pct_green_days", np.nan)    >= float(SOFT["MIN_GREEN"]))

            # HARD (perf) issus de T (héritage CLI/ENV "strict")
            cS_hard = (m.get("sharpe_d_365", np.nan)      >= float(T.min_sharpe_d))
            cPF_hard= (m.get("profit_factor", np.nan)     >= float(T.min_profit_factor))
            cG_hard = (m.get("pct_green_days", np.nan)    >= float(T.min_pct_green))

            # Éligibilités
            core_ok = (base_ok and c_mdd and c_top5 and c_ulcer and c_med and c_skew)

            is_soft = bool(core_ok and cS_soft and cPF_soft and cG_soft)
            is_hard = bool(core_ok and cS_hard and cPF_hard and cG_hard)

            if is_soft:
                soft_trades.extend(trade_rows)
                valid_stats_soft.append({**m, "backtest_id": backtest_id})
            else:
                bad_trades.extend(trade_rows)

            if is_hard:
                hard_trades.extend(trade_rows)
                valid_stats_hard.append({**m, "backtest_id": backtest_id})

        print(f"✅ SOFT (good_iterations) : {len(soft_trades)} trades (ids={len(valid_stats_soft)})")
        print(f"✅ HARD (filtered_full)  : {len(hard_trades)} trades (ids={len(valid_stats_hard)})")

        # --- Adaptive log: N_soft (taille du pool SOFT post-métriques Python) --------
        try:
            _adaptive_log_n_eff(
                conn_src,
                n_eff=len(valid_stats_soft),
                source="export_good_iteration:soft_pool_size_post_metrics",
                meta={"note": "len(valid_stats_soft) après garde-fous + SOFT minima"}
            )
        except Exception as _e:
            print(f"[ADAPTIVE][WARN] log N_soft (post_metrics) échoué: {type(_e).__name__}: {_e}")
        # -----------------------------------------------------------------------------

        # N_eff (post-HARD, avant toute décorrélation) basé sur la passe locale
        try:
            _adaptive_log_n_eff(conn_src, n_eff=len(valid_stats_hard),
                                source="export_good_iteration:extraction_post_hard",
                                meta={"note": "len(valid_stats_hard) avant diversification"})
        except Exception as e:
            print(f"[ADAPTIVE][WARN] log N_eff (extraction_post_hard) échoué: {e}")

        print(f"❌ Mauvais trades (rejet ou hors SOFT) : {len(bad_trades)}")
        # ──────────────────────────────────────────────────────────────────────────────


        # Insert helpers (DELETE then INSERT, avec noms de colonnes explicites)
        def insert_batch(conn, table, rows, *, delete_ids=None):
            if not rows:
                return
            cur = conn.cursor()

            # Colonnes depuis le dtype du recarray
            colnames = list(rows[0].dtype.names)
            placeholders = ",".join(["?"] * len(colnames))
            insert_sql = f"INSERT INTO {table} ({', '.join(colnames)}) VALUES ({placeholders})"

            if delete_ids:
                cur.execute("BEGIN;")
                cur.executemany(f"DELETE FROM {table} WHERE backtest_id = ?", [(bid,) for bid in delete_ids])
                cur.execute("COMMIT;")

            cur.execute("BEGIN;")
            buf = []
            for r in rows:
                buf.append(tuple(r[c] for c in colnames))
                if len(buf) >= ROW_SUBBATCH_SIZE:
                    cur.executemany(insert_sql, buf)
                    buf.clear()
            if buf:
                cur.executemany(insert_sql, buf)
            cur.execute("COMMIT;")

        # Insert SOFT dans GOOD_DB
        soft_ids_set = set(row["backtest_id"] for row in valid_stats_soft)
        insert_batch(conn_good, "trades", soft_trades, delete_ids=soft_ids_set)

        # Insert HARD dans BEST_DB (vitrine)
        hard_ids_set = set(row["backtest_id"] for row in valid_stats_hard)
        insert_batch(conn_best, "trades", hard_trades, delete_ids=hard_ids_set)

        # Upsert KPI → GOOD / BEST pour éviter des KPI “orphelins”
        ins_soft = upsert_kpi_for_ids(conn_src, conn_good, list(soft_ids_set))
        ins_hard = upsert_kpi_for_ids(conn_src, conn_best, list(hard_ids_set))

        print(f"🚚 promoted {len(soft_ids_set)} backtests → {GOOD_DB} (kpi + logs + trades) | kpi_rows={ins_soft}, trades_rows≈{len(soft_trades)}")
        print(f"🚚 promoted {len(hard_ids_set)} backtests → {BEST_DB} (kpi + logs + trades) | kpi_rows={ins_hard}, trades_rows≈{len(hard_trades)}")

        # (indices KPI supplémentaires déjà créés dans upsert_kpi_for_ids)

        # Index pour prochaines passes
        for conn in (conn_good, conn_bad, conn_best):
            if table_exists(conn, "trades"):
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_bid ON trades(backtest_id)")
            if table_exists(conn, "logs"):
                try:
                    cols = {r[1] for r in conn.execute("PRAGMA table_info(logs);")}
                    if "backtest_id" in cols and "timestamp" in cols:
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_bid_ts ON logs(backtest_id, timestamp)")
                    elif "backtest_id" in cols:
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_bid ON logs(backtest_id)")
                except Exception:
                    pass

            conn.commit()

        # Échantillonnage intelligent “bad”
        # Union des IDs retenus par SOFT et HARD
        _valid_soft_ids = {row["backtest_id"] for row in valid_stats_soft}
        _valid_hard_ids = {row["backtest_id"] for row in valid_stats_hard}
        _valid_all_ids  = _valid_soft_ids | _valid_hard_ids

        bad_stats = {k: v for k, v in stats.items() if k not in _valid_all_ids}

        grouped_bad_ids = defaultdict(list)
        for backtest_id, metrics in bad_stats.items():
            # Clé standardisée (fallback NaN si absente)
            sharpe = metrics.get("sharpe_d_365", np.nan)
            if not np.isfinite(sharpe):
                grouped_bad_ids["nan"].append(backtest_id)
            elif sharpe < -1:
                grouped_bad_ids["< -1"].append(backtest_id)
            elif sharpe < 0:
                grouped_bad_ids["-1 to 0"].append(backtest_id)
            elif sharpe < 0.5:
                grouped_bad_ids["0 to 0.5"].append(backtest_id)
            else:
                grouped_bad_ids["0.5 to 1"].append(backtest_id)

        max_per_group = 300
        final_bad_ids = []
        for _, ids in grouped_bad_ids.items():
            sampled = random.sample(ids, min(len(ids), max_per_group))
            final_bad_ids.extend(sampled)

        borderline_ids = [
            bid for bid, m in bad_stats.items()
            if (0.55 < (m.get("win_rate") or 0) < 0.6) or (0.9 < (m.get("profit_factor") or 0) < 1.0)
        ]
        final_bad_ids = list(set(final_bad_ids + borderline_ids))

        bad_sampled = defaultdict(list)
        for row in bad_trades:
            if row.backtest_id in final_bad_ids and len(bad_sampled[row.backtest_id]) < 2:
                bad_sampled[row.backtest_id].append(row)

        final_bad_sample = [row for sub in bad_sampled.values() for row in sub]
        print(f"🎯 Mauvais trades retenus (diversifiés): {len(final_bad_sample)}")
        insert_batch(conn_bad, "trades", final_bad_sample)

        # Sauvegarde parquet (best effort)
        bad_sample_df = pd.DataFrame.from_records(final_bad_sample)
        parquet_path = os.path.join(FOLDER_PATH, f"bad_sample_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}.parquet")
        try:
            bad_sample_df.to_parquet(parquet_path)
        except Exception as e:
            print(f"❌ Échec Parquet ({parquet_path}) : {e}")
        print(f"💾 Sauvegarde Parquet : {parquet_path}")
        del bad_sample_df
        gc.collect()

        # Checkpoint IDs “bad” déjà traités
        checkpoint_file = os.path.join(FOLDER_PATH, "bad_ids_checkpoint.json")
        already_done = set()
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file) as f:
                    already_done = set(json.load(f))
            except Exception:
                already_done = set()

        remaining_bad_ids = [bid for bid in final_bad_ids if bid not in already_done]
        print(f"🔁 {len(remaining_bad_ids)} backtests à traiter après checkpoint")

        # Streaming logs (limite RAM)
        MAX_LOG_BACKTESTS = 300
        remaining_bad_ids = remaining_bad_ids[:MAX_LOG_BACKTESTS]

        def stream_logs_by_ids(conn_src, conn_dst, tag, ids, cols_to_keep):
            print(f"📥 Traitement logs {tag} ({len(ids)} backtests)...")
            if not ids or not table_exists(conn_src, "logs") or not table_exists(conn_dst, "logs"):
                return

            # Préparer insert (sécuriser le set de colonnes par intersection avec les schémas)
            dst_cur = conn_dst.cursor()
            requested = cols_to_keep[:]

            try:
                src_cols = {r[1] for r in conn_src.execute("PRAGMA table_info(logs);")}
                dst_cols = {r[1] for r in conn_dst.execute("PRAGMA table_info(logs);")}
            except Exception:
                src_cols, dst_cols = set(), set()

            available = [c for c in requested if (c in src_cols and c in dst_cols)]
            missing   = [c for c in requested if c not in available]
            if missing:
                # On log juste les 5 premières pour éviter le spam
                print(f"[WARN] logs: colonnes absentes (drop) → {missing[:5]}{'...' if len(missing)>5 else ''}")

            columns = available
            if not columns:
                print("[WARN] logs: aucune colonne commune entre source et destination → skip")
                return

            placeholders = ",".join(["?"] * len(columns))
            insert_query = f"INSERT INTO logs ({', '.join(columns)}) VALUES ({placeholders})"

            for i, bid in enumerate(ids):
                # Fetch logs limités
                query = f"SELECT {', '.join(columns)} FROM logs WHERE backtest_id=? LIMIT {MAX_LOGS_PER_ID}"
                conn_src.row_factory = sqlite3.Row
                rows = [dict(r) for r in conn_src.execute(query, (bid,))]

                if not rows:
                    continue

                # Normaliser timestamps si la colonne est réellement insérée
                if "timestamp" in columns:
                    for r in rows:
                        if r.get("timestamp") is not None:
                            try:
                                r["timestamp"] = int(to_epoch_ms(pd.Series([r["timestamp"]])).iloc[0])
                            except Exception:
                                r["timestamp"] = None

                # Upsert simple : delete then insert
                dst_cur.execute("BEGIN;")
                dst_cur.execute("DELETE FROM logs WHERE backtest_id=?", (bid,))
                # executemany en sous-batches
                for start in range(0, len(rows), ROW_SUBBATCH_SIZE):
                    sub = rows[start:start + ROW_SUBBATCH_SIZE]
                    batch = [tuple(r[c] for c in columns) for r in sub]
                    dst_cur.executemany(insert_query, batch)
                dst_cur.execute("COMMIT;")

                already_done.add(bid)
                if (i + 1) % 10 == 0:
                    with open(checkpoint_file, "w") as f:
                        json.dump(list(already_done), f)
                    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
                    print(f"✅ {i+1}/{len(ids)} traités | RAM: {mem_mb:.2f} MB")
                del rows
                gc.collect()

        # Colonnes logs à conserver
        cols_to_keep = [
            "backtest_id", "timestamp", "iteration",
            "ema_short_period", "ema_long_period", "rsi_period", "rsi_buy_zone", "rsi_sell_zone",
            "rsi_past_lookback", "atr_tp_multiplier", "atr_sl_multiplier", "atr_period",
            "macd_signal_period", "rsi_thresholds_1m", "rsi_thresholds_5m", "rsi_thresholds_15m",
            "rsi_thresholds_1h", "ewma_period", "weight_atr_combined_vol", "threshold_volume",
            "hist_volum_period", "detect_supp_resist_period", "trend_period", "threshold_factor",
            "min_profit_margin", "resistance_buffer_margin", "risk_reward_ratio", "confidence_score_params",
            "signal_weight_bonus", "penalite_resistance_factor", "penalite_multi_tf_step",
            "override_score_threshold", "rsi_extreme_threshold", "signal_pure_threshold",
            "signal_pure_weight", "tp_mode", "confidence_score_real", "score_signal_pure", "condition_marche"
        ]

        soft_ids = [row["backtest_id"] for row in valid_stats_soft]
        hard_ids = [row["backtest_id"] for row in valid_stats_hard]

        stream_logs_by_ids(conn_src, conn_good, "good(SOFT)", soft_ids, cols_to_keep)
        stream_logs_by_ids(conn_src, conn_best, "best(HARD)", hard_ids, cols_to_keep)
        stream_logs_by_ids(conn_src, conn_bad,  "bad",        remaining_bad_ids, cols_to_keep)

        print(f"✅ Logs stream terminés → GOOD({len(soft_ids)}) & BEST({len(hard_ids)})")

    finally:
        conn_src.close()
        conn_good.close()
        conn_bad.close()
        conn_best.close()
        print("🧹 Vacuuming databases (in separate connections)...")
        for _db in (BAD_DB, GOOD_DB, BEST_DB):
            try:
                with sqlite3.connect(_db) as vac_conn:
                    vac_conn.execute("VACUUM;")
            except Exception:
                pass
        print("✅ Vacuum terminé.")


# ──────────────────────────────────────────────────────────────────────────────
# Analyses complémentaires (inchangé)
def load_entire_table_in_chunks(db_path, table_name="logs", chunksize=100_000):
    with sqlite3.connect(db_path) as conn:
        reader = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, chunksize=chunksize)
        return pd.concat(reader, ignore_index=True)

def good_and_bad_strat_extraction():
    print("🔄 Chargement progressif complet de df_good_logs...")
    df_good_logs = load_entire_table_in_chunks(GOOD_DB, table_name="logs")
    print("🔄 Chargement progressif complet de df_bad_logs...")
    df_bad_logs = load_entire_table_in_chunks(BAD_DB, table_name="logs")

    df_good_logs = dedup_logs(df_good_logs)
    df_bad_logs  = dedup_logs(df_bad_logs)

    charts_dir = "/Users/brandonmoncoucut/Desktop/Najas_king/Charts"
    os.makedirs(charts_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d_%Hh%M")

    def save_interactive_histogram(data, col, xlabel, threshold_range=None, title=None):
        fig = px.histogram(
            data_frame=data, x=col, nbins=50, opacity=0.75,
            labels={col: xlabel}, title=title or col
        )
        mean_value = data[col].mean()
        fig.add_vline(x=mean_value, line_dash="dash", line_color="red",
                      annotation_text=f"Moyenne: {mean_value:.2f}",
                      annotation_position="top right")
        if threshold_range:
            fig.add_vrect(x0=threshold_range[0], x1=threshold_range[1],
                          fillcolor="green", opacity=0.2, layer="below", line_width=0)
        filename = os.path.join(charts_dir, f"{col}_interactive_{today_str}.html")
        fig.write_html(filename)
        print(f"💾 Histogramme interactif enregistré : {filename}")

    def compute_good_stats_from_db(db_path):
        """
        Charge 'trades' depuis la DB fournie et calcule un sous-ensemble de KPI via
        metrics_core.compute_intraday_consistency_kpis pour diagnostic/histogrammes.

        Notes
        -----
        - Utilise timestamp 'timestamp' si présent, sinon 'exit_time'.
        - Retourne **uniquement** les clés standardisées (ex: 'sharpe_d_365', 'max_drawdown').

        """

        conn = sqlite3.connect(db_path)
        try:
            query = "SELECT * FROM trades"
            chunks = pd.read_sql_query(query, conn, chunksize=100_000)
            stats = []

            def _f(x):
                try:
                    return float(x) if (x is not None and np.isfinite(x)) else np.nan
                except Exception:
                    return np.nan

            for chunk in chunks:
                ts_col = "timestamp" if "timestamp" in chunk.columns else ("exit_time" if "exit_time" in chunk.columns else None)
                if ts_col is not None:
                    chunk[ts_col] = pd.to_datetime(chunk[ts_col], errors="coerce", utc=True)

                for backtest_id, group in chunk.groupby("backtest_id"):
                    if "pnl_net" not in group.columns or group["pnl_net"].isna().all():
                        continue
                    grp = group.sort_values(ts_col) if ts_col is not None else group

                    # KPI unifiés (tolère colonnes manquantes → NaN ciblés)
                    k = mx.compute_intraday_consistency_kpis(
                        df_trades=grp,
                        df_logs=None,
                        price_col="price",           # si absent en DB, certaines métriques seront NaN
                        timestamp_col=ts_col or "timestamp",
                        fee_rate=0.0,
                        ann_factor=ANN_FACTOR_DAILY  # annualisation crypto 24/7
                    )

                    # Helper local pour robustifier le Sharpe 365
                    def _ensure_sharpe_365_local(kdict: dict) -> float:
                        if kdict.get("sharpe_d_365") is not None:
                            return _f(kdict["sharpe_d_365"])
                        if kdict.get("sharpe_daily_ann") is not None:
                            return _f(kdict["sharpe_daily_ann"])
                        m, s = kdict.get("mean_daily_return"), kdict.get("vol_daily_return")
                        if (m is not None) and (s is not None) and (s not in (0, 0.0)):
                            try:
                                return float(m) / float(s) * (ANN_FACTOR_DAILY ** 0.5)
                            except Exception:
                                return np.nan
                        return np.nan

                    stats.append({
                        "backtest_id": backtest_id,
                        "win_rate": _f(k.get("win_rate")),
                        "profit_factor": _f(k.get("profit_factor")),
                        "sharpe_d_365": _ensure_sharpe_365_local(k),
                        "max_drawdown": _f(k.get("max_drawdown")),
                        "sortino_d_252": _f(k.get("sortino_d_252")),
                    })

            return pd.DataFrame(stats)
        finally:
            conn.close()

    if CREATION_HISTOGRAM == 1:
        df_stats = compute_good_stats_from_db(GOOD_DB)
        df_stats["profit_factor"] = df_stats["profit_factor"].replace([np.inf, -np.inf], np.nan)
        df_stats = df_stats.dropna(subset=["profit_factor"])
        if not df_stats.empty:
            save_interactive_histogram(df_stats, "profit_factor", "Profit Factor", (1.5, 2), "Profit Factor (interactive)")
            save_interactive_histogram(df_stats, "win_rate", "Win Rate", (0.6, 1.0), "Win Rate (interactive)")
            # Clé standardisée (crypto 24/7) — on trace la version **capée**
            df_stats["sharpe_d_365_cap"] = df_stats["sharpe_d_365"].clip(-SHARPE_CAP, SHARPE_CAP)
            save_interactive_histogram(
                df_stats, "sharpe_d_365_cap",
                f"Sharpe (daily, ann=365, cap={SHARPE_CAP})",
                (1, 2),
                f"Sharpe (daily, annualized 365, capped at {SHARPE_CAP})"
            )

    if TRACE_HYPER == 1:
        hyperparams = [
            "ema_short_period", "ema_long_period", "rsi_period", "rsi_buy_zone", "rsi_sell_zone",
            "rsi_past_lookback", "atr_tp_multiplier", "atr_sl_multiplier", "atr_period",
            "macd_signal_period", "rsi_thresholds_1m", "rsi_thresholds_5m", "rsi_thresholds_15m",
            "rsi_thresholds_1h", "ewma_period", "weight_atr_combined_vol", "threshold_volume",
            "hist_volum_period", "detect_supp_resist_period", "trend_period", "threshold_factor",
            "min_profit_margin", "resistance_buffer_margin", "risk_reward_ratio", "confidence_score_params",
            "signal_weight_bonus", "penalite_resistance_factor", "penalite_multi_tf_step",
            "override_score_threshold", "rsi_extreme_threshold", "signal_pure_threshold",
            "signal_pure_weight", "tp_mode"
        ]

        mean_diff = df_good_logs[hyperparams].mean() - df_bad_logs[hyperparams].mean()
        charts_dir = "/Users/brandonmoncoucut/Desktop/Najas_king/Charts"
        mean_diff.sort_values(ascending=False).round(2).to_csv(
            os.path.join(charts_dir, f"hyperparam_mean_diff_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}.csv")
        )

        for param in hyperparams:
            if param not in df_good_logs.columns or param not in df_bad_logs.columns:
                continue
            plt.figure(figsize=(6, 3))
            try:
                sns.kdeplot(df_bad_logs[param], label="Mauvais", fill=True, color="red", alpha=0.4)
                sns.kdeplot(df_good_logs[param], label="Bons", fill=True, color="green", alpha=0.4)
                plt.title(f"Distribution : {param}")
                plt.xlabel(param)
                plt.legend(); plt.grid(True); plt.tight_layout()
                plt.savefig(os.path.join(charts_dir, f"distribution_{param}_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}.png"))
                plt.close()
            except Exception as e:
                print(f"❌ Erreur lors du tracé de {param} : {e}")

    print("\n📊 Analyse terminée.")


def extract_good_strategies_streamed(FOLDER_PATH, output_db_name="filtered_full_copy.db", min_trades=100, chunk_size=100_000):
    source_db_path = os.path.join(FOLDER_PATH, "good_iterations.db")
    output_db_path = os.path.join(FOLDER_PATH, output_db_name)

    conn_src = sqlite3.connect(source_db_path)
    conn_out = sqlite3.connect(output_db_path)

    # Tables out
    for table in ('trades', 'logs'):
        if not table_exists(conn_out, table):
            src_schema = conn_src.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if src_schema and src_schema[0]:
                conn_out.execute(src_schema[0])
            else:
                if table == 'trades':
                    conn_out.execute("""
                        CREATE TABLE trades(
                            backtest_id TEXT,
                            timestamp   INTEGER,
                            exit_time   INTEGER,
                            pnl_net     REAL,
                            balance     REAL,
                            sharpe_ratio REAL
                        );
                    """)
                else:
                    conn_out.execute("""
                        CREATE TABLE logs(
                            backtest_id TEXT,
                            timestamp   INTEGER,
                            iteration   INTEGER
                        );
                    """)
    conn_out.commit()

    if not table_exists(conn_out, "trades"):
        print("⚠️ La table 'trades' n'existe pas encore dans la base finale. Rien à extraire pour le moment.")
        conn_src.close(); conn_out.close(); return
    if not table_exists(conn_src, "trades"):
        print("⚠️ La base source ne contient pas la table 'trades'. Extraction impossible pour l’instant.")
        conn_src.close(); conn_out.close(); return

    existing_ids_df = pd.read_sql_query("SELECT DISTINCT backtest_id FROM trades", conn_out)
    existing_ids = set(existing_ids_df["backtest_id"])
    print(f"📂 {len(existing_ids)} stratégies déjà présentes dans la base finale.")

    print("🔍 Extraction des backtests valides...")
    valid_ids = set()
    chunks = pd.read_sql_query("SELECT * FROM trades", conn_src, chunksize=chunk_size)

    for chunk in chunks:
        for backtest_id, group in chunk.groupby("backtest_id"):
            if backtest_id in existing_ids:
                continue
            if int(group.shape[0]) < min_trades:   # ✅ nombre de lignes
                continue

            pnl = group["pnl_net"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if len(pnl) < 2:
                continue

            wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
            win_rate = float((pnl > 0).mean())
            pf = float(wins.sum() / abs(losses.sum())) if len(losses) else float('inf')
            sharpe = float(pnl.mean() / pnl.std(ddof=1)) if pnl.std(ddof=1) > 0 else 0.0

            if (win_rate > 0.60) and (pf > 1.50) and (sharpe > 0.50):
                valid_ids.add(backtest_id)

        del chunk; gc.collect()

    print(f"✅ {len(valid_ids)} nouvelles stratégies retenues.")

    if valid_ids:
        placeholders = ",".join(["?"] * len(valid_ids))
        trades_query = f"SELECT * FROM trades WHERE backtest_id IN ({placeholders})"
        logs_query =   f"SELECT * FROM logs   WHERE backtest_id IN ({placeholders})"
        trades_filtered = pd.read_sql_query(trades_query, conn_src, params=list(valid_ids))
        logs_filtered = pd.read_sql_query(logs_query,   conn_src, params=list(valid_ids))
        trades_filtered.to_sql("trades", conn_out, if_exists="append", index=False)
        logs_filtered.to_sql("logs",   conn_out, if_exists="append", index=False)
        print(f"✅ {len(trades_filtered)} lignes insérées dans 'trades'")
        print(f"✅ {len(logs_filtered)} lignes insérées dans 'logs'")
    else:
        print("⚠️ Aucune nouvelle stratégie à insérer.")

    conn_src.close(); conn_out.close()
    print("✅ Terminé.")

def final_selection_mo(k_max: int = 100, corr_thr: float = 0.30, pool_top: int = 500) -> None:
    """
    Sélection multi-objectif :
      1) HARD (seulement) sur KPI,
      2) Score par rangs (Sharpe/PF/Green à ↑ ; MDD/Ulcer/Top5 à ↓) avec poids cfg.score_weights(),
      3) (Optionnel) tag Pareto ND,
      4) Diversification : ordre mo_score↓ puis greedy |corr|<corr_thr sur pnl_journalier,
      5) Logging complet : #candidats, quantiles score, #retenus, mean|corr|.
    """
    print("\n🚦 Sélection finale multi-objectif (HARD → score → diversification)")
    W = score_weights(normalize=True)
    with sqlite3.connect(SRC_DB) as conn:
        # 1) KPI HARD (+ valid/outlier/cap)
        df = load_kpis_hard_filtered(conn, cap=SHARPE_CAP)
        if df.empty:
            print("⚠️ Aucun candidat après HARD ; abandon sélection.")
            return

        n_candidates = len(df)
        print(f"📦 Candidats post-HARD: {n_candidates}")

        # N_eff (post-HARD) calculé depuis la vue KPI/SQL, cohérent avec la sélection
        try:
            _adaptive_log_n_eff(conn, n_eff=n_candidates,
                                source="export_good_iteration:final_selection_post_hard",
                                meta={"cap": float(SHARPE_CAP)})
        except Exception as e:
            print(f"[ADAPTIVE][WARN] log N_eff (final_selection_post_hard) échoué: {e}")

        # 2) Score par rangs (percentiles)
        # colonnes attendues (avec NaN-safe)
        for col in ("sharpe_d_365", "profit_factor", "pct_green", "mdd_abs", "ulcer_index", "top5_share"):
            if col not in df.columns:
                df[col] = np.nan

        zS  = rankpct(df["sharpe_d_365"])
        zPF = rankpct(df["profit_factor"])
        zG  = rankpct(df["pct_green"])
        zM  = 1.0 - rankpct(df["mdd_abs"])
        zU  = 1.0 - rankpct(df["ulcer_index"])
        zT5 = 1.0 - rankpct(df["top5_share"])

        df["mo_score"] = (
            W["SHARPE"]*zS + W["PF"]*zPF + W["GREEN"]*zG +
            W["MDD"]*zM + W["ULCER"]*zU + W["TOP5"]*zT5
        )

        # Quantiles de contrôle
        q = df["mo_score"].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).round(4).to_dict()
        print(f"📈 mo_score quantiles: 5%={q.get(0.05)} 25%={q.get(0.25)} 50%={q.get(0.5)} 75%={q.get(0.75)} 95%={q.get(0.95)}")

        # 3) Pareto (optionnel / tag)
        try:
            df = tag_pareto_nondominated(df, cols_up=["sharpe_d_365", "profit_factor", "pct_green"],
                                              cols_down=["mdd_abs", "ulcer_index"])
            print(f"⭐ Pareto ND (tag): {int(df['pareto_nd'].sum())} / {len(df)}")
        except Exception as e:
            print(f"[WARN] Pareto tagging skipped: {e}")

        # 4) Diversification par corrélation sur top 'pool_top' par score
        df_ranked = df.sort_values("mo_score", ascending=False).reset_index(drop=True)
        pool_ids = df_ranked["backtest_id"].head(pool_top).tolist()
        pnl_daily = build_daily_pnl_matrix(conn, pool_ids)
        if pnl_daily.empty or len(pool_ids) <= 1:
            print("⚠️ Impossible de calculer la corrélation (pas de PnL daily). Sélection = top-k mo_score.")
            selected_ids = pool_ids[:k_max]
            mean_abs_corr = float('nan')
        else:
            corr = pnl_daily.corr().fillna(0.0)
            selected_ids = greedy_diversified_selection(corr, pool_ids, k=k_max, corr_thr=corr_thr)
            # métrique de diag
            if len(selected_ids) >= 2:
                sub = corr.loc[selected_ids, selected_ids].copy()
                iu = np.triu_indices(len(selected_ids), k=1)
                mean_abs_corr = float(np.abs(sub.values[iu]).mean())
            else:
                mean_abs_corr = 0.0

        # 5) Logs & résumé
        df_sel = df_ranked[df_ranked["backtest_id"].isin(selected_ids)].copy()
        print(f"🎯 Portefeuille sélectionné (k≤{k_max}, |corr|<{corr_thr:.2f}) : {len(selected_ids)} strats")
        try:
            mean_sharpe = float(df_sel["sharpe_d_365"].mean())
            print(f"   • mean Sharpe 365 (cap) = {mean_sharpe:.3f} | mean|corr|={mean_abs_corr:.3f}")
        except Exception:
            print(f"   • mean|corr|={mean_abs_corr:.3f}")

        # Sauvegarde JSON (ids + score + tag pareto)
        artifacts_dir = os.path.join(FOLDER_PATH, "Artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        out_path = os.path.join(artifacts_dir, f"selected_portfolio_mo_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}.json")
        payload = df_sel[["backtest_id", "mo_score", "sharpe_d_365", "profit_factor", "pct_green", "mdd_abs"]]
        try:
            payload.to_json(out_path, orient="records", indent=2)
            print(f"💾 Portefeuille (MO, diversifié) écrit : {out_path}")
        except Exception as e:
            print(f"[WARN] Écriture JSON échouée : {e}")

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Log des seuils actifs (ENV + CLI)
    HARD, SOFT = hard_soft_thresholds()
    print("🔧 Seuils actifs")
    print("  • Garde-fous globaux : "
        f"ANN_FACTOR_DAILY={ANN_FACTOR_DAILY} | SHARPE_CAP={SHARPE_CAP}")
    print("  • HARD (vitrine/portfolio) : "
        f"MIN_TRADES={HARD['MIN_TRADES']} | MAX_MDD={HARD['MAX_MDD']} | "
        f"MAX_ULCER={HARD['MAX_ULCER']} | MAX_TOP5={HARD['MAX_TOP5']} | "
        f"perf_min via CLI/ENV (min_sharpe_d={T.min_sharpe_d}, "
        f"min_profit_factor={T.min_profit_factor}, min_pct_green={T.min_pct_green})")
    print("  • SOFT (pool GP/BO) : "
        f"MIN_SHARPE={SOFT['MIN_SHARPE']} | MIN_PF={SOFT['MIN_PF']} | MIN_GREEN={SOFT['MIN_GREEN']} | "
        f"MIN_TRADES=HARD({HARD['MIN_TRADES']}) ; garde-fous (MDD/ULCER/TOP5) = HARD")
    print("  • Daily consistency : "
        f"min_median_daily_pnl={T.min_median_daily_pnl} | min_skew_daily_pnl={T.min_skew_daily_pnl}")

    if DO_EXTRACTION == 1:
        extraction_data()
    else:
        print("⏭️  DO_EXTRACTION=0 → skip extraction_data()")

    if extract_very_good_iteration == 1:
        extract_good_strategies_streamed(
            FOLDER_PATH=FOLDER_PATH,
            output_db_name="filtered_full_copy.db"
        )

    # Sélection finale multi-objectif + diversification
    try:
        W = score_weights(normalize=True)
        W_info = ", ".join([f"{k}={v:.2f}" for k, v in W.items()])
        print(f"🧮 SCORE weights (normalisés) → {W_info}")
    except Exception:
        print("🧮 SCORE weights → défauts (normalisation échouée)")

    # k_max & corr_thr peuvent passer en ENV : MO_K_MAX, MO_CORR_THR
    k_max = int(os.environ.get("MO_K_MAX", "100"))
    corr_thr = float(os.environ.get("MO_CORR_THR", "0.30"))
    final_selection_mo(k_max=k_max, corr_thr=corr_thr, pool_top=500)

    if chargement_good_and_bad_logs == 1:
        # Analyses comparatives good vs bad (optionnelles)
        def _noop(*args, **kwargs): pass
        good_and_bad_strat_extraction()

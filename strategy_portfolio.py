"""
====================================================================================
Fichier : strategy_portfolio.py
Objectif : Sélection gloutonne d’un portefeuille de stratégies faiblement corrélées
====================================================================================

Description générale :
Ce module construit un portefeuille de stratégies en maximisant le Sharpe du
portefeuille (égal-pondéré) sous contrainte de décorrélation. Il reconstruit pour
chaque stratégie une série de retours (à partir de 'balance' ou de 'pnl_net'),
calcule la matrice de corrélation, puis sélectionne jusqu’à k stratégies de façon
gloutonne : on n’ajoute une stratégie que si sa corrélation moyenne (absolue)
avec l’ensemble déjà retenu reste < max_corr, et on choisit celle qui améliore le
plus le Sharpe du portefeuille.

Nouveaux critères/outils :
- pairwise_corr_to_others : corrélation absolue moyenne d’une stratégie vs toutes les autres (statistique).
- beta_to_btc_intraday   : bêta vs un benchmark intraday (ex. BTC) si série de benchmark fournie.
- overlap_trade_ratio    : contrainte optionnelle sur le chevauchement temporel de trades vs set sélectionné.

Entrées attendues :
- trades_df : DataFrame avec au minimum les colonnes :
    backtest_id, timestamp (ou exit_time), pnl_net, balance (optionnelle)
- Paramètres d’usage :
    max_corr : seuil de corrélation absolue moyenne autorisée (< 1)
    k        : nombre maximum de stratégies retenues
    periods_per_year : facteur d’annualisation du Sharpe (ex. 252 si returns “quotidiens”)
    annualize : annualiser ou non le Sharpe

Sorties :
- dict avec :
    "selected_ids"        : liste des backtest_id retenus (ordre d’ajout)
    "portfolio_sharpe"    : Sharpe du portefeuille égal-pondéré
    "mean_pairwise_corr"  : corrélation absolue moyenne des stratégies retenues
    "stats"               : DataFrame (Sharpe individuel, corr moyenne contre set courant)
      (peut être utilisé pour journaliser l’historique de sélection)

Remarques :
- Si 'balance' n’est pas disponible pour un backtest_id, on normalise 'pnl_net'
  par son écart-type comme fallback, ce qui est suffisant pour mesurer la
  corrélation relative et faire une sélection robuste (échelle-invariante).
- La corrélation est calculée pairwise en excluant les NaN (comportement pandas).
- Pour le Sharpe, on suppose que l’unité temporelle des retours est homogène
  (sinon adapter periods_per_year à la granularité des returns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os  # ← pour fallback ENV

# --- Adaptive CAP: import tolérant -------------------------------------------
# On essaie d'importer le SHARPE_CAP résolu par config (mode auto/fixed).
# En cas d'échec (ex: import order), on tombera sur l'ENV plus bas.
try:
    from config import SHARPE_CAP as _SHARPE_CAP  # valeur déjà résolue (auto/fixed)
except Exception:
    _SHARPE_CAP = None


# ----------------------------- Utils Sharpe -----------------------------------
def _sharpe_from_returns(returns: pd.Series | pd.DataFrame,
                         periods_per_year: int = 252,
                         annualize: bool = True) -> float:
    """
    Calcule un Sharpe simple à partir d’une série (ou DataFrame) de retours.
    - Si DataFrame : égal-pondère sur les colonnes puis calcule le Sharpe du portefeuille.
    - Si annualize : multiplie par sqrt(periods_per_year).

    Args:
        returns: pd.Series (1d) ou pd.DataFrame (2d) de retours.
        periods_per_year: facteur d’annualisation (252 ~ daily, 365 ~ daily calendaire, etc.)
        annualize: True pour annualiser, False sinon.

    Returns:
        float: Sharpe estimé (peut être NaN si variance nulle).

    Notes:
        - Annualisation identique à metrics_core (sqrt(periods_per_year), défaut=252).
        - Le calcul s’effectue sur des retours (et non des PnL cumulés), ce qui est
          cohérent pour un Sharpe de portefeuille égal-pondéré.
    """
    if isinstance(returns, pd.DataFrame):
        # égal-pondéré avec skipna (moyenne par ligne)
        eq_weighted = returns.mean(axis=1, skipna=True)
    else:
        eq_weighted = returns

    mu = eq_weighted.mean()
    sig = eq_weighted.std(ddof=0)
    if sig == 0 or np.isnan(sig):
        return np.nan
    sharpe = mu / sig
    return float(sharpe * np.sqrt(periods_per_year) if annualize else sharpe)


# ----------------------- Construction séries de retours -----------------------
def _build_returns_matrix(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un DataFrame 'R' de retours par backtest_id (colonnes) et temps (index).

    Règles :
    - Si la colonne 'balance' existe et n'est pas entièrement NaN pour un ID, on prend
      returns = balance.pct_change().
    - Sinon on fallback sur 'pnl_net' normalisé : ret = pnl_net / std(pnl_net) (si std>0).
    - Les timestamps sont pris depuis 'timestamp' si présent, sinon 'exit_time'.
    - Les séries sont alignées par jointure temporelle (outer join).
      Les corrélations pandas sont calculées en “pairwise complete observations”.

    Args:
        trades_df: DataFrame brut des trades.

    Returns:
        pd.DataFrame: R (index = datetime, columns = backtest_id, values = returns)
    """
    df = trades_df.copy()

    # Choix du timestamp (robuste aux colonnes vides)
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        ts_col = "timestamp"
    elif "exit_time" in df.columns and df["exit_time"].notna().any():
        ts_col = "exit_time"
    else:
        raise ValueError("Trades DataFrame must have non-null 'timestamp' or 'exit_time'")

    # conversion datetime
    # timezone-aware, aligné UTC
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col, "backtest_id"]).sort_values([ts_col])

    frames = []
    for bid, g in df.groupby("backtest_id"):
        g = g.sort_values(ts_col)

        if "balance" in g.columns and not g["balance"].isna().all():
            ret = g["balance"].astype(float).pct_change()
            ret = ret.replace([np.inf, -np.inf], np.nan)
        else:
            pnl = g["pnl_net"].astype(float)
            std = pnl.std(ddof=0)
            if std > 0:
                ret = pnl / std
            else:
                # série plate → aucun signal exploitable
                ret = pd.Series(np.zeros(len(pnl)), index=g.index, dtype=float)

        s = pd.Series(ret.values, index=g[ts_col].values, name=str(bid))
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    R = pd.concat(frames, axis=1)  # outer join on time
    # Nettoyage léger
    R = R.sort_index()
    return R

def _compute_intervals_by_id(trades_df: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Construit, pour chaque backtest_id, la liste d'intervalles [entry_time, exit_time].
    Les timestamps sont forcés en UTC et les lignes invalides sont ignorées.
    """
    if not {"backtest_id", "entry_time", "exit_time"}.issubset(trades_df.columns):
        return {}
    df = trades_df[["backtest_id", "entry_time", "exit_time"]].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  errors="coerce", utc=True)
    df = df.dropna(subset=["backtest_id", "entry_time", "exit_time"])
    out: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for bid, g in df.groupby("backtest_id"):
        intervals = [(et, xt) for et, xt in zip(g["entry_time"], g["exit_time"]) if et < xt]
        out[str(bid)] = intervals
    return out


def _overlap_seconds(a: list[tuple[pd.Timestamp, pd.Timestamp]],
                     b: list[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    """Somme des secondes de chevauchement entre deux listes d'intervalles fermés."""
    if not a or not b:
        return 0.0
    i = j = 0
    total = 0.0
    a = sorted(a)
    b = sorted(b)
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        start = max(s1, s2)
        end   = min(e1, e2)
        if start < end:
            total += (end - start).total_seconds()
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return float(total)


def _duration_seconds(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    """Somme des durées (en secondes) d'une liste d'intervalles."""
    return float(sum((e - s).total_seconds() for s, e in intervals)) if intervals else 0.0


def _overlap_trade_ratio_between(a: list[tuple[pd.Timestamp, pd.Timestamp]],
                                 b: list[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    """
    Jaccard-like overlap ratio = overlap / union, avec union = dur(a)+dur(b)-overlap.
    Renvoie 0.0 si union==0 (aucun trade des deux côtés).
    """
    if not a and not b:
        return 0.0
    ov  = _overlap_seconds(a, b)
    da  = _duration_seconds(a)
    db  = _duration_seconds(b)
    uni = max(ov, da + db - ov)  # garde ≥ ov
    return float(ov / uni) if uni > 0 else 0.0


def _beta_vs_benchmark(ret: pd.Series, bench: pd.Series) -> float:
    """
    Bêta simple = Cov(ret, bench) / Var(bench) (pairwise sur index commun).
    Renvoie NaN si insuffisant.
    """
    if ret is None or bench is None:
        return np.nan
    x = pd.concat([ret, bench], axis=1, join="inner").dropna()
    if len(x) < 3:
        return np.nan
    bench_var = x.iloc[:, 1].var(ddof=0)
    if not np.isfinite(bench_var) or bench_var == 0:
        return np.nan
    cov = np.cov(x.iloc[:, 0], x.iloc[:, 1], ddof=0)[0, 1]
    return float(cov / bench_var)


# ------------------------------ Sélection gloutonne ---------------------------
def select_strategy_portfolio(
    trades_df: pd.DataFrame,
    max_corr: float = 0.30,
    k: int = 100,
    periods_per_year: int = 252,
    annualize: bool = True,
    *,
    benchmark_returns: pd.Series | None = None,
    max_overlap_ratio: float | None = None,
    max_global_pairwise_corr: float | None = None,
    max_abs_beta: float | None = None,
) -> dict:
    """
    Sélectionne jusqu’à k stratégies faiblement corrélées en maximisant le Sharpe
    du portefeuille égal-pondéré.

    Contraintes :
      - Corrélation moyenne absolue **vs set sélectionné** < max_corr.
      - (Optionnel) Corrélation absolue moyenne **vs toutes les autres** ≤ max_global_pairwise_corr.
      - (Optionnel) |beta| ≤ max_abs_beta si un benchmark est fourni.
      - (Optionnel) Overlap moyen ≤ max_overlap_ratio.

    Options supplémentaires :
      - benchmark_returns : Série de retours intraday (index datetime UTC) pour calculer
        beta_to_btc_intraday (ou autre benchmark).
      - max_overlap_ratio : si fourni, rejette toute candidate dont le chevauchement moyen
        (overlap_trade_ratio) avec le set sélectionné excède ce seuil (0..1).

    Sorties enrichies :
      - stats inclut: individual_sharpe, mean_abs_corr_to_selected, pairwise_corr_to_others,
        beta_to_btc_intraday (si bench), overlap_trade_ratio_to_selected.

    Étapes :
        1) Construit la matrice de retours R[t, backtest_id].
        2) Calcule Sharpe individuel par colonne.
        3) Tri initial par Sharpe individuel décroissant.
        4) Boucle gloutonne :
           - Si S est vide : ajoute la meilleure.
           - Sinon : parmi les candidates C dont la corr moyenne abs vs S < max_corr,
             ajoute celle qui maximise le Sharpe du portefeuille S∪{c}.
           - Stop quand |S| == k ou plus de candidates admissibles.
        5) Renvoie la liste des IDs, le Sharpe du portefeuille, la corr moyenne, et un
           tableau récapitulatif (stats).

    Args:
        trades_df: DataFrame des trades (backtest_id, timestamp/exit_time, pnl_net, balance).
        max_corr: seuil de corrélation absolue moyenne autorisée (<1).
        k: taille maximale du portefeuille sélectionné.
        periods_per_year: facteur d’annualisation du Sharpe.
        annualize: annualiser le Sharpe (True/False).

    Returns:
        dict: {
            "selected_ids": list[str],
            "portfolio_sharpe": float,
            "mean_pairwise_corr": float,
            "stats": pd.DataFrame
        }
    """

    # --- Log informatif : SHARPE_CAP utilisé (auto/fixed via config, sinon ENV) ---
    try:
        if _SHARPE_CAP is not None:
            cap_val = float(_SHARPE_CAP)
            print(f"🧭 [PORTFOLIO] SHARPE_CAP utilisé (config): {cap_val}")
        else:
            env_val = os.environ.get("SHARPE_CAP", "")
            cap_val = float(env_val) if env_val not in ("", None) else float("nan")
            if np.isfinite(cap_val):
                print(f"🧭 [PORTFOLIO] SHARPE_CAP utilisé (ENV fallback): {cap_val}")
            else:
                print("🧭 [PORTFOLIO] SHARPE_CAP non défini (aucune valeur résolue).")
    except Exception as _e:
        print(f"🧭 [PORTFOLIO][WARN] Impossible d'afficher SHARPE_CAP: {type(_e).__name__}: {_e}")
    # -------------------------------------------------------------------------------

    if k <= 0:
        return {"selected_ids": [], "portfolio_sharpe": np.nan, "mean_pairwise_corr": np.nan,
                "stats": pd.DataFrame()}

    # 1) Retours
    R = _build_returns_matrix(trades_df)
    if R.empty:
        return {"selected_ids": [], "portfolio_sharpe": np.nan, "mean_pairwise_corr": np.nan,
                "stats": pd.DataFrame()}

    # 2) Sharpe individuel
    indiv_sharpes = {col: _sharpe_from_returns(R[col].dropna(),
                                               periods_per_year, annualize)
                     for col in R.columns}
    rank = sorted(indiv_sharpes.items(), key=lambda kv: (-(kv[1] if np.isfinite(kv[1]) else -np.inf), kv[0]))

    # 3) Corrélation pairwise (pandas gère pairwise NaN)
    C = R.corr()  # Pearson, pairwise complete observations
    C = C.fillna(0.0)

    # 3bis) Intervalles d'ouverture/fermeture pour overlap (si colonnes dispo)
    intervals_by_id = _compute_intervals_by_id(trades_df)

    # 3ter) Corrélation absolue moyenne vs toutes les autres (pairwise_corr_to_others) — pré-calcul
    global_mean_abs_corr: dict[str, float] = {}
    for bid in C.columns:
        vals = [abs(C.loc[bid, x]) for x in C.columns if x != bid]
        global_mean_abs_corr[bid] = float(np.mean(vals)) if vals else np.nan

    # 3quater) Bêta vs benchmark (si fourni) — pré-calcul
    beta_by_id: dict[str, float] = {}
    if benchmark_returns is not None:
        for col in R.columns:
            beta_by_id[col] = _beta_vs_benchmark(R[col].dropna(), benchmark_returns)

    selected = []
    remaining = [bid for bid, _ in rank]

    # 4) Glouton
    while len(selected) < min(k, len(remaining)):
        if not selected:
            # Ajoute le meilleur
            selected.append(remaining.pop(0))
            continue

        # Candidates admissibles : corr vs set sélectionné, overlap, corr globale, bêta
        admissibles = []
        for cand in remaining:
            # 1) Corrélation moyenne absolue vs set sélectionné
            rcorr = np.mean([abs(C.loc[cand, s]) for s in selected if cand in C.index and s in C.columns]) \
                    if selected else 0.0
            if np.isfinite(rcorr) and rcorr >= max_corr:
                continue

            # 2) Overlap moyen vs set sélectionné (si demandé)
            if max_overlap_ratio is not None and selected:
                ov_mean = np.mean([
                    _overlap_trade_ratio_between(intervals_by_id.get(cand, []),
                                                 intervals_by_id.get(s, []))
                    for s in selected
                ])
                if np.isfinite(ov_mean) and ov_mean > float(max_overlap_ratio):
                    continue

            # 3) Corrélation absolue moyenne vs toutes les autres (si seuil global fourni)
            if max_global_pairwise_corr is not None:
                gmean = global_mean_abs_corr.get(cand, np.nan)
                if np.isfinite(gmean) and gmean > float(max_global_pairwise_corr):
                    continue

            # 4) Contrainte de bêta (si benchmark et seuil fournis)
            if (benchmark_returns is not None) and (max_abs_beta is not None):
                b = beta_by_id.get(cand, np.nan)
                if np.isfinite(b) and abs(b) > float(max_abs_beta):
                    continue

            admissibles.append(cand)

        if not admissibles:
            break

        # Choisir la candidate qui maximise le Sharpe du portefeuille S U {cand}
        best_cand, best_ptf_sharpe = None, -np.inf
        for cand in admissibles:
            cols = selected + [cand]
            ptf_sh = _sharpe_from_returns(R[cols], periods_per_year, annualize)
            if np.isfinite(ptf_sh) and ptf_sh > best_ptf_sharpe:
                best_ptf_sharpe = ptf_sh
                best_cand = cand

        if best_cand is None:
            break

        selected.append(best_cand)
        remaining.remove(best_cand)

    # 5) Résumé final
    sel_cols = selected
    ptf_sh = _sharpe_from_returns(R[sel_cols], periods_per_year, annualize) if sel_cols else np.nan
    if len(sel_cols) >= 2:
        # moyenne des corrélations absolues sur le sous-graphe retenu (hors diagonale)
        subC = C.loc[sel_cols, sel_cols].copy()
        mean_pairwise_corr = float(np.mean(np.abs(subC.values[np.triu_indices_from(subC, k=1)])))
    else:
        mean_pairwise_corr = 0.0 if len(sel_cols) == 1 else np.nan

    # Stats détaillées (Sharpe individuel + corr moyennes + bêta + overlap)
    stats = pd.DataFrame({
        "backtest_id": list(indiv_sharpes.keys()),
        "individual_sharpe": list(indiv_sharpes.values())
    })

    # Corr moyenne vs set sélectionné
    if sel_cols:
        def mean_abs_corr_to_sel(bid: str) -> float:
            if bid not in C.index:
                return np.nan
            vals = [abs(C.loc[bid, s]) for s in sel_cols if bid in C.index and s in C.columns and bid != s]
            return float(np.mean(vals)) if vals else 0.0
        stats["mean_abs_corr_to_selected"] = [mean_abs_corr_to_sel(b) for b in stats["backtest_id"]]
    else:
        stats["mean_abs_corr_to_selected"] = np.nan

    # Corr absolue moyenne vs toutes les autres (pairwise_corr_to_others) — via pré-calcul
    stats["pairwise_corr_to_others"] = [global_mean_abs_corr.get(str(b), np.nan) for b in stats["backtest_id"]]

    # Overlap moyen vs set sélectionné
    if sel_cols and intervals_by_id:
        def mean_overlap_to_sel(bid: str) -> float:
            a = intervals_by_id.get(bid, [])
            if not a:
                return 0.0
            vals = [_overlap_trade_ratio_between(a, intervals_by_id.get(s, [])) for s in sel_cols if s != bid]
            return float(np.mean(vals)) if vals else 0.0
        stats["overlap_trade_ratio_to_selected"] = [mean_overlap_to_sel(b) for b in stats["backtest_id"]]
    else:
        stats["overlap_trade_ratio_to_selected"] = np.nan

    # Bêta vs benchmark si fourni — via pré-calcul
    if benchmark_returns is not None:
        stats["beta_to_btc_intraday"] = [beta_by_id.get(str(col), np.nan) for col in stats["backtest_id"]]
    else:
        stats["beta_to_btc_intraday"] = np.nan

    stats = stats.sort_values("individual_sharpe", ascending=False).reset_index(drop=True)

    return {
        "selected_ids": sel_cols,
        "portfolio_sharpe": float(ptf_sh) if np.isfinite(ptf_sh) else np.nan,
        "mean_pairwise_corr": mean_pairwise_corr,
        "stats": stats,
    }
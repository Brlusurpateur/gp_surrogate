"""
====================================================================================
Fichier : backtest_interface_code.py
Objectif : Interface autonome de backtesting d'une stratégie de trading paramétrée
====================================================================================

Description générale :
Ce module permet d'exécuter un backtest complet d'une stratégie algorithmique sur 
données historiques, à partir d'un simple dictionnaire d’hyperparamètres. Il centralise 
la logique d’interface entre :

    - le chargement de données multi-unités de temps,
    - la construction des paramètres de stratégie,
    - l’appel au moteur de backtest vectoriel,
    - et la journalisation des résultats dans une base SQLite.

Ce fichier sert de brique d’exécution unitaire pour automatiser des séries de tests
(expérimentations GP, AutoML, BO, etc.).

Contexte méthodologique :
Le pipeline assume un paradigme basé sur des simulations ex-post (backtesting),
où chaque combinaison d’hyperparamètres définit une stratégie unique. La performance
de cette stratégie est mesurée à l’aide d’indicateurs robustes comme :

    - Sharpe Ratio (moyenne des rendements / volatilité empirique)
    - Max Drawdown (perte maximale relative)
    - Profit Factor (total gains / total pertes)
    - Win Rate (proportion de trades gagnants)

Composants internes appelés :
- `scalping_backtest_engine.py` : moteur vectorisé de backtesting.
- `strategy_params_core.py` : constructeur paramétrique `StrategyParams`.
- `parallel_backtest_launcher.py` : utilitaire de chargement des données OHLCV multi-timeframe.
- `backtest_db_manager.py` : conversion des logs en `DataFrame` propre.
- `sqlite3` : stockage persistant des résultats dans `trade_logs.db`.

Fonction centrale :
    - `run_backtest_from_params(param_dict: dict) -> bool`
        Cette fonction encapsule le cycle complet :
            → préparation des données,
            → génération des paramètres,
            → exécution du backtest,
            → calcul et journalisation des KPIs,
            → validation finale.

Utilisation typique :
Appelée de façon itérative dans une boucle d’optimisation (GP, BoTorch, Grid Search),
ou bien intégrée dans un moteur d’exploration AutoML pour évaluer des milliers de
configurations.

Entrées :
    - `param_dict` (dict) : hyperparamètres de la stratégie (seuils, durées, poids, etc.)

Sorties :
    - Booléen : succès ou échec du backtest (selon la présence des métriques critiques).

Sécurité mémoire :
    - Les données marché sont clonées à chaque appel (pas de mutation du cache global)
    - Utilisation de `gc.collect()` systématique en `finally` pour éviter toute fuite
    - Les UUIDs garantissent la traçabilité de chaque exécution dans la base de logs

Cas d’usage typiques :
    - Boucle AutoML ou Meta-Backtesting
    - Fonction de score dans un framework d’optimisation (BoTorch, Optuna, Ax)
    - Interface API pour scoring de configurations RL ou rule-based

Auteur : Moncoucut Brandon
Version : Juin 2025
"""


# === Imports fondamentaux ===
import sqlite3
import uuid
import numpy as np
import pandas as pd  # ← daily PnL stats
import gc
from typing import Tuple, Dict

# Annualisation daily pour crypto 24/7 (fallback robuste)
try:
    from config import ANN_FACTOR_DAILY
except Exception:
    ANN_FACTOR_DAILY = 365

from scalping_signal_engine import get_pair_info
from scalping_backtest_engine import backtest_strategy
from strategy_params_core import init_params, params_to_dict
from parallel_backtest_launcher import load_or_fetch_market_data

# DB helpers (DDL/migration + insert + KPI upsert)
from backtest_db_manager import (
    array_to_clean_dataframe,
    init_sqlite_database,
    migrate_sqlite_schema,
    insert_logs_from_array,
    insert_trades_from_array,
    upsert_kpis,
)

# Calcul des KPI “consistency-first”
import metrics_core as mx
# Annualisation & garde-fous (crypto 24/7) — fallback robuste si config indisponible
try:
    from config import ANN_FACTOR_DAILY, SHARPE_CAP, MIN_ACTIVE_DAYS, MIN_STD_DAILY
except Exception:
    ANN_FACTOR_DAILY = 365
    SHARPE_CAP = 5.0
    MIN_ACTIVE_DAYS = 30
    MIN_STD_DAILY = 1e-6


# 🔄 Fichier autonome pour exécuter un backtest à partir d'un dictionnaire de paramètres

PAIR = "BTCUSDC"
NB_DAYS = 185
INTERVALS = ["1m", "5m", "15m", "1h"]
DATA_PATH = "/Users/brandonmoncoucut/Desktop/Najas_king/log_sqlite/backtest_pipeline/histo_data_185d.db"
FINAL_DB_PATH = "/Users/brandonmoncoucut/Desktop/Najas_king/log_sqlite/backtest_pipeline/trade_logs.db"

INIT_BALANCE = 5000.0
FEE_RATE = 0.001
RISK_PER_TRADE = 0.02  # 2% (fraction)

# Cache des données (une seule fois pour toute la session)
_market_data_cache = None

def run_backtest_from_params(param_dict: dict, iteration: int = 0) -> Tuple[bool, str, Dict[str, float]]:
    """
    Exécute un backtest unitaire à partir d'un jeu d'hyperparamètres, persiste les données
    dans SQLite (init + migration idempotentes), calcule les KPI via metrics_core et fait
    un UPSERT des KPI.

    Args:
        param_dict (dict): Hyperparamètres de la stratégie (réels ou normalisés).
        iteration (int): Identifiant d'itération/batch (par défaut 0) utilisé comme PK avec backtest_id.

    Returns:
        Tuple[bool, str, Dict[str, float]]:
            - success (bool): True si KPI clés valides et upsertés, False sinon.
            - backtest_id (str): UUID de l'exécution pour traçabilité.
            - kpi (dict): Dictionnaire des KPI calculés (peut contenir des NaN si données manquantes).

    Notes
    -----
    - Les inserts utilisent les helpers dédiés (array→DB) pour préserver les dtypes.
    - L'UPSERT KPI s'appuie sur la clé primaire (backtest_id, iteration).
    """

    global _market_data_cache

    try:
        # Chargement ou mise en cache des données marché (multi-timeframe)
        if _market_data_cache is None:
            _market_data_cache = load_or_fetch_market_data(PAIR, INTERVALS, NB_DAYS, DATA_PATH)

        # Récupération des contraintes de trading pour la paire (quantités et prix)
        min_qty, step_size, tick_size = get_pair_info(PAIR)

        # Génération des paramètres sous forme d'objet StrategyParams (avec injection BoTorch)
        params = init_params(min_qty, step_size, tick_size, custom_values=param_dict)

        # ID unique pour tracer ce backtest
        backtest_id = str(uuid.uuid4())

        # Duplication des données pour ne pas muter le cache global
        data_clone = {
            interval: {col: arr.copy() for col, arr in interval_data.items()}
            for interval, interval_data in _market_data_cache.items()
        }

        # ===== 1) Lancement du backtest (moteur vectoriel) =====
        trade_history, log_array, early_stop = backtest_strategy(
            data_clone,
            INIT_BALANCE,
            params,
            FEE_RATE,
            RISK_PER_TRADE
        )

        # ===== 2) Init + migration du schéma DB (idempotent) =====
        init_sqlite_database(FINAL_DB_PATH)
        migrate_sqlite_schema(FINAL_DB_PATH)

        # ===== 3) Inserts (logs & trades) — robustes via helpers dtype-aware =====
        _ = insert_logs_from_array(FINAL_DB_PATH, backtest_id, iteration, log_array)

        # ===== 4) KPI “consistency-first” =====
        df_log = array_to_clean_dataframe(log_array)
        df_trade = array_to_clean_dataframe(trade_history)

        # NB: compute_intraday_consistency_kpis tolère des colonnes manquantes (remplies à NaN)
        #     On tente ann_factor=ANN_FACTOR_DAILY, sinon on retombe sur signature historique.
        try:
            kpi_raw = mx.compute_intraday_consistency_kpis(
                df_trades=df_trade,
                df_logs=df_log,
                price_col="price",
                timestamp_col="timestamp",
                fee_rate=FEE_RATE,
                ann_factor=ANN_FACTOR_DAILY,  # ← peut ne pas exister selon ta version
            )
        except TypeError:
            kpi_raw = mx.compute_intraday_consistency_kpis(
                df_trades=df_trade,
                df_logs=df_log,
                price_col="price",
                timestamp_col="timestamp",
                fee_rate=FEE_RATE,
            )

        # --- Enrichissement KPI utile à la sélection ---
        kpi = dict(kpi_raw)  # shallow copy

        # 1) Harmonisation Sharpe 365 :
        #    - si la lib renvoie déjà 'sharpe_d_365', on garde
        #    - sinon si 'sharpe_daily_ann' → mappe dessus
        #    - sinon si 'sharpe_d_252' → convertit vers 365 via √(365/252)
        if "sharpe_d_365" not in kpi or kpi.get("sharpe_d_365") is None:
            if "sharpe_daily_ann" in kpi and kpi["sharpe_daily_ann"] is not None:
                kpi["sharpe_d_365"] = float(kpi["sharpe_daily_ann"])
            elif "sharpe_d_252" in kpi and kpi["sharpe_d_252"] is not None:
                try:
                    kpi["sharpe_d_365"] = float(kpi["sharpe_d_252"]) * float(np.sqrt(365.0 / 252.0))
                except Exception:
                    pass  # on laisse None si conversion impossible

        # 2) % jours verts : alias si besoin (pct_green_days est ce que consomme l’aval)
        if "pct_green_days" not in kpi and "green_days_ratio" in kpi and kpi["green_days_ratio"] is not None:
            kpi["pct_green_days"] = float(kpi["green_days_ratio"])

        # 3) mdd_abs (copie sign-free de max_drawdown)
        if "mdd_abs" not in kpi and "max_drawdown" in kpi and kpi["max_drawdown"] is not None:
            try:
                kpi["mdd_abs"] = float(abs(kpi["max_drawdown"]))
            except Exception:
                kpi["mdd_abs"] = None

        # 4) n_unique_days (fallback léger sans pandas)
        if "n_unique_days" not in kpi or kpi.get("n_unique_days") is None:
            try:
                if df_trade is not None and not df_trade.empty and "timestamp" in df_trade.columns:
                    ts = df_trade["timestamp"].to_numpy()
                    ts = ts[~np.isnan(ts)]
                    if ts.size > 0:
                        # Heuristique ms vs s
                        denom = 86_400_000.0 if float(np.nanmax(ts)) > 1e12 else 86_400.0
                        days_bucket = (ts // denom).astype(np.int64)
                        kpi["n_unique_days"] = int(np.unique(days_bucket).size)
                    else:
                        kpi["n_unique_days"] = None
                else:
                    kpi["n_unique_days"] = None
            except Exception:
                kpi["n_unique_days"] = None

        # (ulcer_index, top5_share, profit_factor sont propagés tels quels si fournis par metrics_core)
        # ----------------------------------------------------------------

        # UPSERT KPI dans la table dédiée
        upsert_kpis(FINAL_DB_PATH, backtest_id, iteration, kpi)

        def _ensure_sharpe_365(k: dict, ann_factor: int) -> dict:
            """
            Garantit une clé 'sharpe_d_365' dans le dict KPI, avec logique “defense in depth”.
            Ordre des tentatives :
            1) Si 'sharpe_d_365' existe → inchangé.
            2) Si 'sharpe_daily_ann' existe (nom générique) → remap vers 'sharpe_d_365'.
            3) Sinon, si 'mean_daily_return' & 'vol_daily_return' dispo → calcule Sharpe = (m/s)*sqrt(ann_factor).
            4) Sinon, fallback: laisse 'sharpe_d_365' manquant (None) plutôt que d’inventer.
            NB: Pas de conversion naïve depuis 'sharpe_d_252' → évite des erreurs d’échelle.
            """
            out = dict(k) if isinstance(k, dict) else {}
            if out.get("sharpe_d_365") is not None:
                return out
            if out.get("sharpe_daily_ann") is not None:
                out["sharpe_d_365"] = float(out["sharpe_daily_ann"])
                return out
            m, s = out.get("mean_daily_return"), out.get("vol_daily_return")
            if (m is not None) and (s is not None) and (s not in (0, 0.0)):
                try:
                    out["sharpe_d_365"] = float(m) / float(s) * (ann_factor ** 0.5)
                    return out
                except Exception:
                    pass
            out.setdefault("sharpe_d_365", None)
            return out

        # Normalisation des noms → DB standardisée
        kpi = _ensure_sharpe_365(kpi_raw, ANN_FACTOR_DAILY)

        # ──────────────────────────────────────────────────────────────────────────────
        #  Garde-fous “quant-grade” : robustifier et filtrer les outliers à la source
        #    - n_unique_days : nb. de jours actifs de PnL (qualité d'échantillon)
        #    - std_daily     : écart-type des PnL journaliers (évite Sharpe instable)
        #    - flag_sharpe_outlier / is_valid / invalid_reason
        #  NB: on taggue en DB et on **hard-reject** (is_valid=0) pour ne rien polluer côté GP.
        # ──────────────────────────────────────────────────────────────────────────────

        def _safe_int(x):
            try:
                return int(x) if x is not None and np.isfinite(x) else None
            except Exception:
                return None

        def _safe_float(x):
            try:
                return float(x) if x is not None and np.isfinite(x) else None
            except Exception:
                return None

        def _compute_daily_stats(df: pd.DataFrame, ts_col: str = "timestamp") -> tuple[int | None, float | None]:
            """
            Calcule (n_unique_days, std_daily) à partir d'un DataFrame de trades/logs.
            Essaie d'abord 'pnl'/'pnl_quote'/'pnl_usd'/'daily_pnl' ; fallback sur 0 si non trouvés.
            """
            if df is None or df.empty or ts_col not in df.columns:
                return None, None

            # Colonnes candidates pour un PnL au niveau trade/ligne
            pnl_cols = [c for c in ["daily_pnl", "pnl", "pnl_quote", "pnl_usd", "pnl_net"] if c in df.columns]
            if not pnl_cols:
                return None, None

            try:
                ts = pd.to_datetime(df[ts_col], unit="ms", errors="coerce")  # tes timestamps sont en ms
            except Exception:
                ts = pd.to_datetime(df[ts_col], errors="coerce")

            if ts.isna().all():
                return None, None

            df_tmp = df.copy()
            df_tmp["_date"] = ts.dt.date

            # somme PnL par jour (évite double comptage si plusieurs trades/jour)
            daily = df_tmp.groupby("_date")[pnl_cols].sum(min_count=1)
            # si plusieurs colonnes PnL existent, somme par ligne puis std
            daily_sum = daily.sum(axis=1)

            # nombre de jours avec une observation non nulle / non NaN
            n_days = int(daily_sum.dropna().shape[0])
            std_daily = float(daily_sum.std(ddof=1)) if n_days >= 2 else 0.0

            return n_days, std_daily

        # Compléter n_unique_days / std_daily si absents dans kpi
        if kpi.get("n_unique_days") is None or kpi.get("std_daily") is None:
            n_days_t, std_t = _compute_daily_stats(df_trade, ts_col="timestamp")
            n_days_l, std_l = _compute_daily_stats(df_log,   ts_col="timestamp")
            # priorité trades, sinon logs, sinon None
            kpi["n_unique_days"] = _safe_int(kpi.get("n_unique_days")) or n_days_t or n_days_l
            kpi["std_daily"]     = _safe_float(kpi.get("std_daily"))  or (std_t if std_t not in (None, 0) else std_l)

        # Normaliser défauts
        kpi["n_unique_days"] = _safe_int(kpi.get("n_unique_days"))
        kpi["std_daily"]     = _safe_float(kpi.get("std_daily"))

        # Flags & validation
        kpi["is_valid"] = 1
        kpi["invalid_reason"] = None
        kpi["flag_sharpe_outlier"] = 0

        # Règle 1 : trop peu de jours actifs OU volatilité journalière trop faible
        if (kpi["n_unique_days"] is not None and kpi["n_unique_days"] < MIN_ACTIVE_DAYS) or \
        (kpi["std_daily"] is not None and kpi["std_daily"] < MIN_STD_DAILY):
            kpi["is_valid"] = 0
            kpi["invalid_reason"] = "too_few_days_or_zero_vol"

        # Règle 2 : Sharpe aberrant → on **hard-reject** (recommandé) + tag outlier
        sd365 = _safe_float(kpi.get("sharpe_d_365"))
        if sd365 is not None and np.isfinite(sd365) and abs(sd365) > SHARPE_CAP:
            kpi["flag_sharpe_outlier"] = 1
            kpi["is_valid"] = 0
            kpi["invalid_reason"] = "sharpe_outlier"
            # Optionnel: exposer un Sharpe "winsorisé" si tu veux tracer/diagnostiquer
            kpi["sharpe_d_365_clipped"] = float(np.sign(sd365) * SHARPE_CAP)

        # UPSERT KPI avec les flags
        upsert_kpis(FINAL_DB_PATH, backtest_id, iteration, kpi)


        # ===== 5) Critère de succès : KPI clés calculés (Sharpe 365 désormais) =====
        required = ["sharpe_d_365", "sortino_d_252", "max_drawdown", "profit_factor", "win_rate"]

        def _is_valid_num(x):
            try:
                return (x is not None) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))
            except Exception:
                return False

        has_metrics = all((m in kpi) and _is_valid_num(kpi[m]) for m in required)

        # Hard-reject : si 'is_valid' a été posé à 0 (peu de jours, vol nulle, outlier), on refuse.
        if int(kpi.get("is_valid", 1)) == 0:
            reason = kpi.get("invalid_reason", "invalid")
            print(f"🚫 Backtest invalidé (reason={reason}) — Sharpe365={kpi.get('sharpe_d_365')} — Params: {param_dict}")
            return False, backtest_id, kpi

        if has_metrics:
            return True, backtest_id, kpi
        else:
            missing = [m for m in required if (m not in kpi) or (not _is_valid_num(kpi[m]))]
            print(f"❌ KPI requis manquants/NaN: {missing} — Params: {param_dict}")
            return False, backtest_id, kpi

    except Exception as e:
        # On essaye d'exposer backtest_id si déjà généré, sinon chaîne vide.
        try:
            bid = backtest_id
        except Exception:
            bid = ""
        print(f"[❌ run_backtest_from_params] Erreur: {e} | backtest_id={bid}")
        return False, bid, {}

    finally:
        # Nettoyage mémoire après chaque run
        gc.collect()
"""
=============================================================================================
📁 Fichier : parallel_backtest_launcher.py
🎯 Objectif : Orchestration parallèle de simulations de backtests avec journalisation structurée
=============================================================================================

🧭 Description générale :
Ce script constitue le point d’entrée principal pour exécuter **massivement** et de manière **parallélisée** 
des simulations de stratégies de trading algorithmique. Il combine des **batchs de workers multiprocessing**, 
une **base SQLite transactionnelle**, et une logique de gestion fine des ressources (RAM, cache, I/O), dans le but 
d’effectuer une **exploration systématique d’espace d’hyperparamètres** ou d’un pipeline d’optimisation (AutoML / MetaRL).

Chaque instance de stratégie scalping est lancée dans un sous-processus isolé, avec journalisation des résultats 
dans des tables SQL (`logs`, `trades`) enrichies de KPI quantitatifs et d’identifiants reproductibles.

📦 Fonctions principales :
    - `run_single_backtest(...)` : exécute un backtest unique (stratégie, journalisation, metrics, insertion SQL)
    - `load_or_fetch_market_data(...)` : charge les données depuis cache SQLite ou API Binance
    - `cleanup_after_backtests(...)` : libère ressources mémoire, workers zombies et fichiers temporaires

📊 Résultats produits :
    - Table `logs` : toutes les décisions du bot à chaque timestamp
    - Table `trades` : PnL et exécution de chaque trade (open, win, loss, timeout)
    - KPIs calculés : Sharpe ratio annualisé, max drawdown, profit factor, win rate

🧪 Architecture logicielle :
    - Chargement des données (SQLite ou API)
    - Initialisation des contraintes de marché (tick_size, step_size, etc.)
    - Batching sécurisé avec `tqdm` et `multiprocessing.get_context("spawn")`
    - Chaque backtest → `uuid4`, log, trade, métriques, commit SQL (verrouillé par mutex)
    - Nettoyage intelligent après chaque batch (RAM, Numba, fichiers `.npy`, workers)

🔐 Robustesse intégrée :
    - `try/except` pour chaque worker, sans impacter les autres
    - KPI fallback si erreur de calcul
    - Verrou multiprocessing (`Lock`) pour éviter corruption SQL
    - Garbage collection systématique (`gc.collect()`)
    - Fermeture explicite du `Manager` pour éviter fuite de sémaphores

🔁 Contrôles paramétrables :
    - `chunk_size` : taille d’un batch de simulations
    - `pause_duration` : pause entre batchs pour limiter la pression mémoire
    - `n_workers` : nombre de processus parallèles
    - `n_iterations` : nombre total de scénarios à simuler

Auteur : Moncoucut Brandon  
Version : Juin 2025
"""

# === Imports fondamentaux ===
from scalping_backtest_engine import *
import uuid
import os
import time
from tqdm import tqdm
from time import sleep
from multiprocessing import get_context, Manager
from strategy_params_core import init_params, params_to_dict
from scalping_signal_engine import get_pair_info, play_sound
import gc
import shutil
import tempfile
import multiprocessing
import sqlite3
import sys
import json
import traceback
from automl_backtest_driver import MODE, PAIR, NB_DAYS, INTERVALS, INIT_BALANCE, FEE_RATE, RISK_PER_TRADE, DB_DIR, DATA_PATH, FINAL_DB_PATH
from backtest_db_manager import (
    init_sqlite_database,
    load_market_data_from_sqlite,
    save_market_data_to_sqlite,
    is_sqlite_fusion_empty,
    diagnose_sqlite_backtest_db,
    array_to_clean_dataframe,
    migrate_sqlite_schema,        # migration idempotente
    insert_logs_from_array,       # insert robuste logs
    insert_trades_from_array,     # insert robuste trades
    upsert_kpis,                  # upsert KPI par (backtest_id, iteration)
)
import numpy as np
import pandas as pd  # daily PnL stats (n_unique_days, std_daily)

# Annualisation & garde-fous (crypto 24/7)
try:
    from config import ANN_FACTOR_DAILY, SHARPE_CAP, MIN_ACTIVE_DAYS, MIN_STD_DAILY
except Exception:
    ANN_FACTOR_DAILY = 365
    SHARPE_CAP = 5.0
    MIN_ACTIVE_DAYS = 30
    MIN_STD_DAILY = 1e-6

import metrics_core as mx          # KPI “consistency-first”

# Logging + alertes
import logging
from logging_setup import setup_logging
from alerts_telegram import send_alert
# Annualisation daily pour crypto 24/7 (fallback robuste)
try:
    from config import ANN_FACTOR_DAILY
except Exception:
    ANN_FACTOR_DAILY = 365


logger = logging.getLogger(__name__)

tp_mode_map = {
    "rr_only": 0,
    "min_profit_only": 1,
    "avg_all": 2
}

# Mapping pour tp_mode
TP_MODE_DECODE = {v: k for k, v in tp_mode_map.items()}
DEFAULT_MARKET_CONSTRAINTS = (1e-4, 1e-6, 1e-2)


def _sanitize_kpi_dict(kpi: dict) -> dict:
    """
    Convertit les valeurs NaN / ±Inf en None pour compatibilité SQLite/JSON.

    Args:
        kpi (dict): Dictionnaire de métriques calculées.

    Returns:
        dict: Dictionnaire nettoyé (scalaires simples uniquement).
    """
    clean = {}
    if not isinstance(kpi, dict):
        return clean
    for k, v in kpi.items():
        try:
            if v is None:
                clean[k] = None
            elif isinstance(v, (int,)) and not isinstance(v, bool):
                clean[k] = int(v)
            elif isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    clean[k] = None
                else:
                    clean[k] = float(v)
            elif isinstance(v, str):
                # Autoriser un court texte (invalid_reason) pour diagnostic
                clean[k] = v[:255]
            else:
                # On ne stocke que des scalaires simples ou None
                clean[k] = None
        except Exception:
            clean[k] = None

    return clean

def run_single_backtest(args):
    try:
        # Dépaquetage des arguments nécessaires au backtest
        (
            i,                  # identifiant du backtest (index)
            data,               # données marché multi-timeframe
            pair,               # nom de la paire de trading (ex: BTCUSDC)
            init_balance,       # capital initial
            transaction_fee_rate, 
            risk_per_trade,     # pour gestion du sizing
            db_dir,             # dossier où se trouve la DB
            min_qty, step_size, tick_size,  # contraintes de marché
            final_db_path,      # chemin absolu vers DB SQLite
            db_lock,             # verrou de synchronisation multiprocessing
            custom_values
        ) = args

        # Génère un identifiant unique pour tracer le backtest
        backtest_id = str(uuid.uuid4())

        # 🔧 Initialise un objet contenant tous les hyperparamètres (paramètres scalables)
        try:
            params = init_params(min_qty, step_size, tick_size, custom_values=custom_values)
        except Exception as e:
            print(f"[❌] Erreur dans init_params : {e}")
            raise

        # Clonage des données : chaque process obtient une copie indépendante
        data_clone = {
            interval: {col: arr.copy() for col, arr in interval_data.items()}
            for interval, interval_data in data.items()
        }

        # Lancement du backtest via la stratégie principale
        try:
            trade_history, log_array, early_stop = backtest_strategy(
                data_clone,
                init_balance,
                params,
                transaction_fee_rate,
                risk_per_trade
            )
        except Exception as e:
            print("[❌ ERREUR dans backtest_strategy]", e)
            raise  # remonte l'erreur au process parent

        # ===== Insertion robuste (arrays → DB) =====
        iteration = i + 1

        # (1) Logs & trades : append via helpers (tolèrent schémas partiels)
        with db_lock:
            _ = insert_logs_from_array(final_db_path, backtest_id, iteration, log_array)
            _ = insert_trades_from_array(final_db_path, backtest_id, iteration, trade_history)

        # ===== KPI complets “consistency-first” =====
        df_log   = array_to_clean_dataframe(log_array)
        df_trade = array_to_clean_dataframe(trade_history)

        # Calcul KPI (tolère colonnes manquantes → NaN ciblés)
        try:
            kpi_raw = mx.compute_intraday_consistency_kpis(
                df_trades=df_trade,
                df_logs=df_log,
                price_col="price",
                timestamp_col="timestamp",
                fee_rate=transaction_fee_rate,
                ann_factor=ANN_FACTOR_DAILY,   # ← peut ne pas être supporté selon ta version
            )
        except TypeError:
            kpi_raw = mx.compute_intraday_consistency_kpis(
                df_trades=df_trade,
                df_logs=df_log,
                price_col="price",
                timestamp_col="timestamp",
                fee_rate=transaction_fee_rate,
            )

        # --- Enrichissement KPI utile à la sélection ---
        kpi = dict(kpi_raw)

        # Sharpe 365 (voir logique côté interface unitaire)
        if "sharpe_d_365" not in kpi or kpi.get("sharpe_d_365") is None:
            if "sharpe_daily_ann" in kpi and kpi["sharpe_daily_ann"] is not None:
                kpi["sharpe_d_365"] = float(kpi["sharpe_daily_ann"])
            elif "sharpe_d_252" in kpi and kpi["sharpe_d_252"] is not None:
                try:
                    kpi["sharpe_d_365"] = float(kpi["sharpe_d_252"]) * float(np.sqrt(365.0 / 252.0))
                except Exception:
                    pass

        # %green alias
        if "pct_green_days" not in kpi and "green_days_ratio" in kpi and kpi["green_days_ratio"] is not None:
            kpi["pct_green_days"] = float(kpi["green_days_ratio"])

        # mdd_abs
        if "mdd_abs" not in kpi and "max_drawdown" in kpi and kpi["max_drawdown"] is not None:
            try:
                kpi["mdd_abs"] = float(abs(kpi["max_drawdown"]))
            except Exception:
                kpi["mdd_abs"] = None

        # n_unique_days (fallback léger sans pandas)
        if "n_unique_days" not in kpi or kpi.get("n_unique_days") is None:
            try:
                if df_trade is not None and not df_trade.empty and "timestamp" in df_trade.columns:
                    ts = df_trade["timestamp"].to_numpy()
                    ts = ts[~np.isnan(ts)]
                    if ts.size > 0:
                        denom = 86_400_000.0 if float(np.nanmax(ts)) > 1e12 else 86_400.0
                        days_bucket = (ts // denom).astype(np.int64)
                        kpi["n_unique_days"] = int(np.unique(days_bucket).size)
                    else:
                        kpi["n_unique_days"] = None
                else:
                    kpi["n_unique_days"] = None
            except Exception:
                kpi["n_unique_days"] = None

        # Sanitize puis UPSERT KPI (clé = backtest_id + iteration)
        kpi = _sanitize_kpi_dict(kpi)
        with db_lock:
            upsert_kpis(final_db_path, backtest_id, iteration, kpi)

        def _ensure_sharpe_365(k: dict, ann_factor: int) -> dict:
            """
            Garantit 'sharpe_d_365' (voir logique détaillée dans backtest_interface_code.py).
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

        # Normalisation 365
        kpi_norm = _ensure_sharpe_365(kpi_raw, ANN_FACTOR_DAILY)

        # ─────────────────────────────────────────────────────────────
        #  Garde-fous “quant-grade” (anti-outliers / qualité échantillon)
        #    - n_unique_days : # de jours de PnL actifs
        #    - std_daily     : std des PnL journaliers
        #    - flag_sharpe_outlier / is_valid / invalid_reason
        #  Politique : HARD-REJECT (is_valid=0) pour ne pas polluer le GP
        # ─────────────────────────────────────────────────────────────
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
            Calcule (n_unique_days, std_daily) à partir d’un DataFrame de trades/logs.
            Cherche une colonne PnL plausible et agrège par jour (timestamps en ms par défaut).
            """
            if df is None or df.empty or ts_col not in df.columns:
                return None, None
            pnl_cols = [c for c in ["daily_pnl", "pnl", "pnl_quote", "pnl_usd", "pnl_net"] if c in df.columns]
            if not pnl_cols:
                return None, None
            try:
                ts = pd.to_datetime(df[ts_col], unit="ms", errors="coerce")
            except Exception:
                ts = pd.to_datetime(df[ts_col], errors="coerce")
            if ts.isna().all():
                return None, None
            dtmp = df.copy()
            dtmp["_date"] = ts.dt.date
            daily = dtmp.groupby("_date")[pnl_cols].sum(min_count=1).sum(axis=1)  # somme sur colonnes PnL candidates
            n_days = int(daily.dropna().shape[0])
            std_daily = float(daily.std(ddof=1)) if n_days >= 2 else 0.0
            return n_days, std_daily

        # compléter n_unique_days / std_daily si absents
        if kpi_norm.get("n_unique_days") is None or kpi_norm.get("std_daily") is None:
            n_days_t, std_t = _compute_daily_stats(df_trade, ts_col="timestamp")
            n_days_l, std_l = _compute_daily_stats(df_log,   ts_col="timestamp")
            kpi_norm["n_unique_days"] = _safe_int(kpi_norm.get("n_unique_days")) or n_days_t or n_days_l
            kpi_norm["std_daily"]     = _safe_float(kpi_norm.get("std_daily"))  or (std_t if std_t not in (None, 0) else std_l)

        # normalisation valeurs
        kpi_norm["n_unique_days"] = _safe_int(kpi_norm.get("n_unique_days"))
        kpi_norm["std_daily"]     = _safe_float(kpi_norm.get("std_daily"))

        # flags
        kpi_norm["is_valid"] = 1
        kpi_norm["invalid_reason"] = None
        kpi_norm["flag_sharpe_outlier"] = 0

        # Règle 1 : échantillon trop court / vol journalière quasi nulle
        if (kpi_norm["n_unique_days"] is not None and kpi_norm["n_unique_days"] < MIN_ACTIVE_DAYS) or \
           (kpi_norm["std_daily"] is not None and kpi_norm["std_daily"] < MIN_STD_DAILY):
            kpi_norm["is_valid"] = 0
            kpi_norm["invalid_reason"] = "too_few_days_or_zero_vol"

        # Règle 2 : Sharpe 365 aberrant → hard-reject + tag outlier
        sd365 = _safe_float(kpi_norm.get("sharpe_d_365"))
        if sd365 is not None and np.isfinite(sd365) and abs(sd365) > SHARPE_CAP:
            kpi_norm["flag_sharpe_outlier"] = 1
            kpi_norm["is_valid"] = 0
            kpi_norm["invalid_reason"] = "sharpe_outlier"
            # optionnel: exposer un Sharpe winsorisé à des fins de debug/plots
            kpi_norm["sharpe_d_365_clipped"] = float(np.sign(sd365) * SHARPE_CAP)

        # Sanitize → UPSERT KPI (clé = backtest_id + iteration)
        kpi = _sanitize_kpi_dict(kpi_norm)
        with db_lock:
            upsert_kpis(final_db_path, backtest_id, iteration, kpi)

        # Log utile si invalidé
        if int(kpi.get("is_valid", 1) or 1) == 0:
            logger.info("🚫 KPI invalidé | backtest_id=%s | reason=%s | sharpe365=%s | n_days=%s | std_daily=%s",
                        backtest_id, kpi.get("invalid_reason"), kpi.get("sharpe_d_365"),
                        kpi.get("n_unique_days"), kpi.get("std_daily"))
        else:
            logger.info("✅ KPI upsert | backtest_id=%s | iteration=%s | sharpe365=%s",
                        backtest_id, iteration, kpi.get("sharpe_d_365"))

        logger.info("✅ KPI upsert | backtest_id=%s | iteration=%s | n_kpi=%d", backtest_id, iteration, len(kpi))

        return True  # succès

    except Exception as e:
        bid = locals().get('backtest_id', 'n/a')
        print(f"[❌ Worker #{locals().get('i', 'n/a')}] Erreur: {e} | backtest_id={bid}")
        print(traceback.format_exc())
        try:
            send_alert(f"🚨 Backtest worker crash | id={bid} | {type(e).__name__}: {e}")
        except Exception:
            pass
        return False  # échec

def load_or_fetch_market_data(pair, intervals, nb_days, path_data):
    """
    Charge les données de marché depuis SQLite si disponible, sinon les télécharge via l'API.
    ⚠️ Important : aucune connexion API n'est faite si le cache SQLite existe déjà.
    """
    # 1) Cas cache local présent → lecture directe, zéro dépendance API/clefs
    if os.path.exists(path_data):
        print(f"📥 Base SQLite détectée : chargement depuis '{path_data}' (aucune clé API requise)...")
        return load_market_data_from_sqlite(path_data, intervals)

    # 2) Cas sans cache → téléchargement via API (public endpoints suffisent)
    print("🌐 Aucune base locale détectée. Téléchargement via API Binance (public endpoints)...")
    try:
        # Optionnel : établir un client pour fixer l’API_URL (mainnet/testnet) et tester la connectivité.
        # La version api_mainnet.get_binance_client tolère l’absence de clefs.
        client = connect_wallet(testnet=None)  # lit TESTNET depuis l'env si défini
        set_binance_client(client)            # si votre pipeline en a besoin pour fetch_historical_data
    except Exception as e:
        # On journalise mais on tente quand même fetch_historical_data si elle ne dépend pas d’un client signé.
        print(f"⚠️ Impossible d'initialiser un client Binance (mode public). Raison: {e}")

    data = fetch_historical_data(pair, intervals, nb_days)
    save_market_data_to_sqlite(data, path_data)
    return data


def cleanup_after_backtests(temp_dirs=None, npy_search_dir=None, kill_workers=True):
    """
    Nettoie les ressources mémoire, fichiers temporaires, et éventuellement termine les processus enfants.

    Args:
        temp_dirs (list[str] | None): Répertoires temporaires à supprimer.
        npy_search_dir (str | None): Répertoire dans lequel rechercher les fichiers .npy temporaires.
        kill_workers (bool): Si True, termine les processus enfants (utile à la toute fin uniquement).
    """

    print("🧹 Nettoyage des ressources...")

    # Suppression des répertoires temporaires
    if temp_dirs:
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"🗑️ Répertoire temporaire supprimé : {temp_dir}")
            except Exception as e:
                print(f"⚠️ Erreur suppression temp dir {temp_dir} : {e}")

    # Suppression des fichiers .npy temporaires
    search_dir = npy_search_dir or tempfile.gettempdir()
    try:
        npy_files = [f for f in os.listdir(search_dir) if f.endswith(".npy")]
        for f in npy_files:
            try:
                os.remove(os.path.join(search_dir, f))
                print(f"🗑️ Fichier .npy supprimé : {f}")
            except Exception as e:
                print(f"⚠️ Impossible de supprimer {f} : {e}")
    except Exception as e:
        print(f"⚠️ Erreur en listant {search_dir} : {e}")

    # Libération de la mémoire du processus parent
    gc.collect()
    print("🧠 Mémoire RAM (parent) libérée avec gc.collect()")

    # Terminaison des workers si demandé (uniquement à la fin)
    if kill_workers:
        children = multiprocessing.active_children()
        if children:
            print(f"🔍 {len(children)} processus enfants encore actifs. Tentative de terminaison...")
            for proc in children:
                try:
                    proc.terminate()
                    proc.join(timeout=1)
                    print(f"🔪 Processus {proc.name} terminé.")
                except Exception as e:
                    print(f"⚠️ Impossible de terminer {proc.name} : {e}")
        else:
            print("✅ Aucun processus enfant actif.")
    else:
        print("ℹ Les workers ne sont pas terminés (kill_workers=False).")

    print("✅ Nettoyage finalisé.\n")

_pair_info_cache = {}
def get_pair_info_cached(pair):
    if pair not in _pair_info_cache:
        _pair_info_cache[pair] = get_pair_info(pair)
    return _pair_info_cache[pair]

def run_batch_backtests(
    n_iterations,
    pair,  # paire de trading à simuler
    nb_days,  # nombre de jours d’historique de données marché à utiliser
    intervals,  # unités de temps à charger pour les backtests multi-timeframe
    init_balance,  # capital initial utilisé dans chaque backtest
    fee_rate,  # taux de frais de transaction (ex: 0.1%)
    risk_per_trade,  # risque alloué à chaque trade (en pourcentage du capital)
    db_dir,  # répertoire de la base SQLite
    data_path,  # chemin vers la base contenant les données de marché (ohlcv)
    final_db_path, # chemin vers la base finale de log des résultats
    mode="random",  # ou "custom" pour injection de suggestions précises (ex: GP)
    chunk_size=100,  # taille des batchs (nombre de backtests par groupe)
    pause_duration=60,  # temps de pause entre deux batchs
    n_workers=4,  # nombre de processus multiprocessing à lancer en parallèle
    injected_params_list=None  # liste de dictionnaires d'hyperparamètres, utilisée si mode = "custom"
    ):
    """
    Lance un batch de backtests en mode aléatoire ou via injection d’hyperparamètres.

    Args:
        mode (str): "random" ou "custom"
        injected_params_list (list[dict]): si mode == "custom"
        ...
    """
    # Vérifie que le mode est bien valide
    assert mode in ["random", "custom"], "mode doit être 'random' ou 'custom'"

        # Vérifie que la liste des paramètres injectés est bien présente et suffisante si on est en mode 'custom'
    if mode == "custom":
        assert injected_params_list is not None, "injected_params_list doit être fourni en mode 'custom'"
        assert len(injected_params_list) >= n_iterations, (
            f"injected_params_list doit contenir au moins {n_iterations} éléments "
            f"(actuellement {len(injected_params_list)})"
        )

    # Initialise un verrou multiprocessing global pour sécuriser l’accès concurrent à la base SQLite
    manager = Manager()
    db_lock = manager.Lock()

    # Définit les chemins par défaut si non spécifiés
    data_path = data_path or os.path.join(db_dir, "histo_data.db")
    final_db_path = final_db_path or os.path.join(db_dir, "trade_logs.db")

    # Charge les données de marché OHLCV depuis SQLite ou depuis l’API Binance
    data = load_or_fetch_market_data(pair, intervals, nb_days, data_path)

    # Récupère les contraintes du marché et applique des defaults robustes en BACKTEST
    orig_mq, orig_ss, orig_ts = get_pair_info_cached(pair)
    min_qty, step_size, tick_size = apply_market_defaults(orig_mq, orig_ss, orig_ts)
    if any(x is None or (isinstance(x, (int, float)) and x <= 0) for x in (orig_mq, orig_ss, orig_ts)):
        print(f"[ℹ️] Using default market constraints (BACKTEST): min_qty={min_qty}, step_size={step_size}, tick_size={tick_size}")

    # Initialise + migre la base SQLite de résultats (idempotent)
    os.makedirs(os.path.dirname(final_db_path), exist_ok=True)
    init_sqlite_database(final_db_path)
    migrate_sqlite_schema(final_db_path)  # ✅ ajoute colonnes manquantes si DB existante

    # Timer pour mesurer la durée totale du batch
    start_global = time.time()

    # Calcule le nombre total de batchs en fonction du nombre d’itérations demandées
    total_batches = (n_iterations + chunk_size - 1) // chunk_size

    print(f"\n🚀 Lancement parallèle sur {n_workers} workers ({total_batches} batches de {chunk_size})...")

    # Boucle principale sur chaque batch
    for batch_num in range(total_batches):
        # Détermine les indices de début et fin du batch courant
        start_idx = batch_num * chunk_size
        end_idx = min((batch_num + 1) * chunk_size, n_iterations)

        print(f"\n📦 Batch {batch_num + 1}/{total_batches} → [{start_idx} à {end_idx - 1}]")

        # Prépare la liste des arguments à passer à chaque processus worker
        args_list = [
            (
                i,                      # ID du backtest
                data,                  # Données marché (copiées par chaque worker)
                pair, init_balance, fee_rate, risk_per_trade,
                db_dir, min_qty, step_size, tick_size,
                final_db_path, db_lock,
                injected_params_list[i] if mode == "custom" else None  # injection des params si custom
            )
            for i in range(start_idx, end_idx)
        ]

        # Lance les backtests du batch en parallèle dans un pool multiprocessing sécurisé
        with get_context("spawn").Pool(processes=n_workers, maxtasksperchild=10) as pool:
            with tqdm(total=len(args_list), desc=f"🔁 Batch {batch_num + 1}", ncols=100) as pbar:
                for status in pool.imap_unordered(run_single_backtest, args_list):
                    pbar.update(1)  # met à jour la barre de progression
                    if not status:
                        print(f"⚠️ Une tâche du batch {batch_num + 1} a échoué.")
                    del status  # libère explicitement l’objet
                    gc.collect()  # nettoyage mémoire immédiat

        # Pause entre deux batchs pour relâcher la pression mémoire/disque
        print(f"⏸️ Pause de {pause_duration} secondes après le batch {batch_num + 1}...")
        sleep(pause_duration)
        gc.collect()  # collecte garbage du parent
        cleanup_after_backtests(npy_search_dir="/tmp", kill_workers=False)  # nettoyage temporaire

    # Résumé après tous les backtests
    print(f"\n🏁 Tous les backtests terminés en {time.time() - start_global:.2f} sec.\n")

    # Nettoyage final mémoire et workers restants
    cleanup_after_backtests(npy_search_dir="/tmp", kill_workers=True)
    manager.shutdown()  # arrêt propre du multiprocessing.Manager

    # Analyse de la base SQLite résultante
    diagnose_sqlite_backtest_db(final_db_path)

    # Vérifie si la base est vide avant de jouer le son de fin
    if is_sqlite_fusion_empty(final_db_path):
        print("⚠️ Aucune donnée détectée dans la base fusionnée. Aucun export effectué.")
    else:
        play_sound()  # son de notification si tout s’est bien passé

def apply_market_defaults(min_qty, step_size, tick_size):
    """Remplace None/≤0 par des defaults, caste en float32 et vérifie > 0."""
    mq = DEFAULT_MARKET_CONSTRAINTS[0] if (min_qty is None or (isinstance(min_qty, (int, float)) and min_qty <= 0)) else min_qty
    ss = DEFAULT_MARKET_CONSTRAINTS[1] if (step_size is None or (isinstance(step_size, (int, float)) and step_size <= 0)) else step_size
    ts = DEFAULT_MARKET_CONSTRAINTS[2] if (tick_size is None or (isinstance(tick_size, (int, float)) and tick_size <= 0)) else tick_size
    try:
        mq, ss, ts = np.float32(mq), np.float32(ss), np.float32(ts)
    except Exception:
        mq, ss, ts = map(np.float32, DEFAULT_MARKET_CONSTRAINTS)
    assert mq > 0 and ss > 0 and ts > 0, "Market constraints must be positive"
    return mq, ss, ts


if __name__ == "__main__":
    setup_logging()

    if len(sys.argv) != 2:
        print("❌ Usage: python parallel_backtest_launcher.py <suggestions_json>")
        raise SystemExit(1)

    suggestions_file = sys.argv[1]
    logger.info("🚀 Lancement batch backtests | suggestions=%s", suggestions_file)

    try:
        with open(suggestions_file, "r") as f:
            suggestions = json.load(f)

        run_batch_backtests(
            n_iterations=len(suggestions),
            mode=MODE,
            injected_params_list=suggestions,
            pair=PAIR,
            nb_days=NB_DAYS,
            intervals=INTERVALS,
            init_balance=INIT_BALANCE,
            fee_rate=FEE_RATE,
            risk_per_trade=RISK_PER_TRADE,
            db_dir=DB_DIR,
            data_path=DATA_PATH,
            final_db_path=FINAL_DB_PATH
        )
    except Exception as e:
        logger.exception("Crash batch backtests")
        # Alerte Telegram en cas de crash critique
        send_alert(f"🚨 Crash parallel_backtest_launcher: {type(e).__name__}: {e}")
        raise
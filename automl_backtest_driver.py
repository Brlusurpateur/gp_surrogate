"""
====================================================================================
Fichier : automl_backtest_driver.py
Objectif : Pilotage centralisé du pipeline GP + Backtest + Visualisation SHAP
====================================================================================

Description générale :
Ce fichier orchestre l'optimisation automatique d'une stratégie de trading paramétrique,
en combinant une modélisation bayésienne par processus gaussien (BoTorch), l'exécution
de backtests vectorisés, et l'analyse explicative via SHAP et PDP.

Le pipeline fonctionne en boucle :
    1. Génération de nouvelles suggestions d'hyperparamètres via BoTorch (surrogate model GP),
    2. Évaluation de chaque configuration à l'aide d'un moteur de backtest haute fréquence,
    3. Analyse visuelle de l'espace de décision à l'aide de SHAP, PDP croisés, et UMAP.

Cette architecture s'inspire des pratiques professionnelles chez les fonds quantitatifs
et des approches de recherche opérationnelle comme proposées par López de Prado.

Fonctionnalités clés :
- `sample_params(trial)` : support pour Optuna (optionnel),
- `objective(trial)` : fonction objectif compatible avec Optuna (optionnel),
- `run_pipeline_gp_backtest_visualize()` : pipeline complet GP → backtests → visualisation.

Modules utilisés :
    - `gp_driver.py` : génération intelligente de configurations via Gaussian Process,
    - `backtest_interface_code.py` : exécution et journalisation du backtest,
    - `surrogate_modeling.py` : visualisation explicative de la performance stratégique.

Base de données :
    - SQLite (`trade_logs.db`) pour persistance des logs et résultats,
    - Utilisée comme mémoire commune entre les modules.

Résultats attendus :
    - Backtests enrichis avec métriques de performance (Sharpe, PF, etc.),
    - Visualisations exportées dans `./figs_surrogate/`,
    - Pipeline reproductible et interprétable.

Auteur : Moncoucut Brandon  
Version : Juin 2025
"""

# === Imports fondamentaux ===
import os
import sys
import warnings
import subprocess
import json
import sqlite3
import pandas as pd
import math
import numpy as np
import datetime
from strategy_portfolio import select_strategy_portfolio
from gp_driver_utils import adapt_tr_radius
from scalping_backtest_engine import fetch_historical_data
from backtest_db_manager import (
    save_market_data_to_sqlite,
    load_market_data_from_sqlite,
    has_market_data_coverage,
    ensure_selected_table_exists,
    insert_selected_strategies,
    init_sqlite_database,       
    migrate_sqlite_schema,       
)
import scalping_signal_engine as sse
import time  # NEW: temporisation légère si besoin
from pathlib import Path  #  gestion propre des dossiers
import metrics_core as mx         # ✅ alias standard
import logging
from logging_setup import setup_logging
from alerts_telegram import send_alert
import argparse  # NEW: CLI (target, cadence boucle)
from config import load_thresholds, Thresholds, ANN_FACTOR_DAILY  # seuils + annualisation (crypto 24/7)
try:
    from config import SHARPE_CAP
except Exception:
    SHARPE_CAP = 5.0  # garde-fou si variable absente

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# ⚙️ CONFIG

N_TOTAL_TRIALS = 10 #5000  # Nombre total de suggestions/backtests à exécuter (loop finale = N_TOTAL_TRIALS // BATCH_SIZE)
BATCH_SIZE = 10  # Taille des batchs générés à chaque itération par le GP (nombre de stratégies proposées à chaque boucle)
MODE = "custom"

PAIR = "BTCUSDC"  # Nom de la paire crypto à trader (ici BTC contre USDC)
NB_DAYS = 185  # Nombre de jours d’historique de marché à charger pour les backtests
INTERVALS = ["1m", "5m", "15m", "1h"]  # Unités de temps utilisées pour le backtest multi-timeframes (multi-UT)

INIT_BALANCE = 5000.0  # Capital initial utilisé dans chaque backtest
FEE_RATE = 0.001       # Taux de frais appliqué aux trades (ex. 0.1%)
RISK_PER_TRADE = 0.02  # Fraction du capital risquée par trade (2% de 5000 = 100)

FINAL_DB_PATH = "/Users/brandonmoncoucut/Desktop/Najas_king/log_sqlite/backtest_pipeline/suggest_strat_test.db"
# → Chemin vers la base SQLite principale contenant tous les logs de backtest (stratégies testées)

DB_DIR = "/Users/brandonmoncoucut/Desktop/Najas_king/log_sqlite"
# → Chemin vers le dossier des bases SQLite

DATA_PATH = "/Users/brandonmoncoucut/Desktop/Najas_king/log_sqlite/backtest_pipeline/histo_data_185d.db"
# → Base contenant les données de marché historiques (ohlcv) utilisées pendant les backtests

BEST_STRAT_DB_PATH = "/Users/brandonmoncoucut/Desktop/Najas_king/log_sqlite/backtest_pipeline/filtered_full_copy.db"
# → Base de données contenant uniquement les stratégies ayant des performances jugées acceptables (sharpe_ratio > 0.8)

GOOD_STRAT_DB_PATH = "/Users/brandonmoncoucut/Desktop/Najas_king/log_sqlite/backtest_pipeline/good_iterations.db"
# → Base (facultative) pour stocker uniquement les "bonnes" itérations (win rate > 40% ou Profit factor > 50% sharpe_ratio > 0.5)

tp_mode_map = {"rr_only": 0, "min_profit_only": 1, "avg_all": 2}
# → Encodage des modes de take profit (TP) utilisés dans la stratégie pour paramétrage cohérent

TR_STATE_PATH = "tr_state.json"   # état persistant du rayon TR entre les batchs
DEFAULT_TR_RADIUS = 0.15          # rayon TR initial si pas d’état

ALLOW_DOWNLOAD_IF_MISSING = True   # si False → jamais d’I/O réseau, même si la base manque
TIMESTAMP_SOURCE = "open"          # "open" ou "close" pour le download (si déclenché)

# Répertoire des artefacts explicatifs (aligné avec surrogate_modeling.py)
ARTIFACTS_DIR = Path("/Users/brandonmoncoucut/Desktop/Najas_king/Artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Seuils “consistency-first” centralisés (ENV → defaults) + cible de stratégies
T: Thresholds = load_thresholds()   # min_trades, min_pct_green, max_mdd_abs, etc.
TARGET_GOOD_STRATS_DEFAULT = 100    # peut être surchargé via CLI --target-good

# Objectif AutoML (KPI standardisé calculé par metrics_core → table kpi_by_backtest)
# Crypto 24/7 → annualisation daily sur 365 jours
OBJECTIVE_KPI = "sharpe_d_365"

# === Constantes de pilotage dynamique et EMA SHAP ===
# Lissage EMA des importances SHAP entre batchs
SHAP_EMA_ALPHA = 0.75            # 0<alpha<=1 ; 0.75 = priorité au dernier batch
SHAP_EMA_FILE  = ARTIFACTS_DIR / "shap_importance_ema.json"  # stockage de l'EMA
SHAP_RAW_FILE  = ARTIFACTS_DIR / "shap_importance.json"       # fichier produit par surrogate_modeling.py

# Seuil d'amélioration relative du meilleur Sharpe pour juger "progrès" vs "stagnation"
IMPROV_EPS = 0.01                # +1% minimum

# Paramètres de base pour le GP (dynamiques batch par batch)
# RATIONNEL:
# - q_local ↑ : plus d'essais autour des centres pour mieux “épaissir” l'échantillon local.
# - q_global ~ : garde une petite exploration large pour éviter l'enfermement.
# - distance_thr ↓ : laisse passer davantage de candidats proches (diversité moins stricte).
Q_LOCAL_BASE     = 12            # exploitation locale (était 6 → 12)
Q_GLOBAL_BASE    = 3             # exploration globale (était 2 → 3)
DIST_THR_BASE    = 0.10          # distance min entre candidats (était 0.15 → 0.10)

# Bornes / pas d’adaptation
# Note: on élargit la fenêtre autorisée pour q_local afin de laisser l'adaptation respirer.
Q_LOCAL_MIN, Q_LOCAL_MAX   = 6, 16
Q_GLOBAL_MIN, Q_GLOBAL_MAX = 2,  6
DIST_THR_MIN, DIST_THR_MAX = 0.05, 0.35

# ── Mapping TR→distance & réglages dynamiques (robustes) ─────────────────
TR_RMIN, TR_RMAX = 0.05, 0.40      # bornes utilisées aussi par adapt_tr_radius
DIST_MAP_GAMMA   = 0.70            # courbure pour d_thr(ρ)
DIST_MAP_EPS     = 0.20            # évite un collapse à 0 quand ρ→0
DIST_HYST_FRAC   = 0.08            # hysteresis ±8% : évite les oscillations
COOLDOWN_BATCHES = 1               # 1 batch min avant tout nouveau changement
PACK_WARN_FRAC   = 0.75            # <75% de fill → on relâchera d_thr au prochain batch
PACK_FAIL_FRAC   = 0.50            # <50% → on élargit aussi un peu le rayon TR
PACK_DELTA       = 0.02            # pas unitaire pour relâcher d_thr côté “packing”

# Heuristiques d'adaptation: conservent le comportement existant
# (resserrer si progrès, relâcher si stagnation). On peut les affiner plus tard.
DIST_THR_PROGRESS_STEP     = -0.03   # si progrès: on resserre (→ exploitation)
DIST_THR_STAGNATE_STEP     = +0.05   # si stagnation: on relâche (→ exploration)

# ── Adaptive: table de décision pour logger N_eff (post-HARD) ─────────────────
def _adaptive_ensure_table_path(db_path: str) -> None:
    try:
        with sqlite3.connect(db_path) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_decisions (
                    ts_utc TEXT NOT NULL,
                    source TEXT NOT NULL,
                    n_eff  INTEGER NOT NULL,
                    meta   TEXT
                );
            """)
            c.commit()
    except Exception as e:
        print(f"[ADAPTIVE][WARN] log N_eff (start) échoué: {type(e).__name__}: {e}")

def _adaptive_log_n_eff_dbpath(db_path: str, n_eff: int, source: str, meta: dict | None = None) -> None:
    try:
        _adaptive_ensure_table_path(db_path)
        ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        payload = None
        try:
            payload = json.dumps(meta or {}, ensure_ascii=False)
        except Exception:
            payload = None
        with sqlite3.connect(db_path) as c:
            c.execute(
                "INSERT INTO adaptive_decisions (ts_utc, source, n_eff, meta) VALUES (?,?,?,?)",
                (ts, str(source), int(n_eff), payload)
            )
            c.commit()
    except Exception as e:
        print(f"[ADAPTIVE][WARN] log N_eff (start) échoué: {type(e).__name__}: {e}")
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dirs():
    """Crée au besoin les répertoires requis par la pipeline (artefacts, charts…)."""
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        Path("/Users/brandonmoncoucut/Desktop/Najas_king/Charts/surrogate_modeling").mkdir(parents=True, exist_ok=True)
        Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Impossible de créer certains dossiers ({e}), on continue quand même.")

# Hyperparamètres à optimiser
def sample_params(trial):
    return {
        "ema_short_period": trial.suggest_int("ema_short_period", 5, 28),
        "ema_long_period": trial.suggest_int("ema_long_period", 60, 90),
        "rsi_period": trial.suggest_int("rsi_period", 5, 40),
        "rsi_buy_zone": trial.suggest_int("rsi_buy_zone", 2, 20),
        "rsi_sell_zone": trial.suggest_int("rsi_sell_zone", 50, 84),
        "rsi_past_lookback": trial.suggest_int("rsi_past_lookback", 2, 24),
        "atr_tp_multiplier": trial.suggest_float("atr_tp_multiplier", 2.0, 20.0),
        "atr_sl_multiplier": trial.suggest_float("atr_sl_multiplier", 2.0, 17.0),
        "atr_period": trial.suggest_int("atr_period", 30, 59),
        "macd_signal_period": trial.suggest_int("macd_signal_period", 50, 90),
        "rsi_thresholds_1m": trial.suggest_float("rsi_thresholds_1m", 5.0, 30.0),
        "rsi_thresholds_5m": trial.suggest_float("rsi_thresholds_5m", 5.0, 25.0),
        "rsi_thresholds_15m": trial.suggest_float("rsi_thresholds_15m", 15.0, 30.0),
        "rsi_thresholds_1h": trial.suggest_float("rsi_thresholds_1h", 40.0, 100.0),
        "ewma_period": trial.suggest_int("ewma_period", 2, 20),
        "weight_atr_combined_vol": trial.suggest_float("weight_atr_combined_vol", 0.001, 0.7),
        "threshold_volume": trial.suggest_int("threshold_volume", 0, 24),
        "hist_volum_period": trial.suggest_int("hist_volum_period", 50, 89),
        "detect_supp_resist_period": trial.suggest_int("detect_supp_resist_period", 5, 39),
        "trend_period": trial.suggest_int("trend_period", 5, 59),
        "threshold_factor": trial.suggest_float("threshold_factor", 25.05, 50.0),
        "min_profit_margin": trial.suggest_float("min_profit_margin", 0.01, 10.0),
        "resistance_buffer_margin": trial.suggest_float("resistance_buffer_margin", 1.0, 20.0),
        "risk_reward_ratio": trial.suggest_float("risk_reward_ratio", 1.1, 15.0),
        "confidence_score_params": trial.suggest_float("confidence_score_params", 0, 101),
        "signal_weight_bonus": trial.suggest_float("signal_weight_bonus", -100, 100),
        "penalite_resistance_factor": trial.suggest_float("penalite_resistance_factor", -100, 100.0),
        "penalite_multi_tf_step": trial.suggest_float("penalite_multi_tf_step", -100, 100.0),
        "override_score_threshold": trial.suggest_int("override_score_threshold", 60, 99),
        "rsi_extreme_threshold": trial.suggest_int("rsi_extreme_threshold", 1, 15),
        "signal_pure_threshold": trial.suggest_float("signal_pure_threshold", 0.001, 20.0),
        "signal_pure_weight": trial.suggest_float("signal_pure_weight", 40, 90),
        "tp_mode": tp_mode_map["rr_only"],  # fixé ici pour simplifier
    }

def _load_tr_state(path=TR_STATE_PATH):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_tr_state(state: dict, path=TR_STATE_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)

def _load_json_safe(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json_atomic(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)

def _ema_shap_importance(alpha: float = SHAP_EMA_ALPHA):
    """
    Lisse ARTIFACTS/shap_importance.json dans SHAP_EMA_FILE via EMA
    puis écrase shap_importance.json avec la version lissée (pour le GP suivant).
    """
    raw = _load_json_safe(SHAP_RAW_FILE)
    if not raw:
        print("⚠️ EMA SHAP: aucun shap_importance.json brut à lisser.")
        return

    prev = _load_json_safe(SHAP_EMA_FILE)

    # Format attendu: {"feature_name": importance_float, ...}
    # On harmonise les clés/features
    all_keys = set(raw.keys()) | set(prev.keys())
    smoothed = {}
    for k in all_keys:
        x = float(raw.get(k, 0.0))
        m = float(prev.get(k, x))  # si pas d'EMA précédent → démarre à la valeur courante
        smoothed[k] = alpha * x + (1.0 - alpha) * m

    # Sauvegarde EMA et remplace le shap_importance pour GP
    _save_json_atomic(smoothed, SHAP_EMA_FILE)
    _save_json_atomic(smoothed, SHAP_RAW_FILE)
    print(f"✅ SHAP importance lissée (EMA α={alpha}) → {SHAP_EMA_FILE.name} et {SHAP_RAW_FILE.name}")

def _get_max_rowid(db_path: str, table: str) -> int:
    """
    Retourne le dernier ROWID d'une table SQLite.
    Sert de marqueur pour compter les nouvelles lignes insérées après un événement.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT MAX(rowid) FROM {table};")
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return 0


def _count_new_backtests_since(
    db_path: str,
    logs_rowid_before: "int | None",
    trades_rowid_before: "int | None",
) -> tuple[int, int]:
    """
    Compte le nombre de backtests distincts réellement exécutés depuis des ROWID "avant".
    On renvoie deux comptes :
        - n_logs   : nb de backtest_id distincts apparus dans logs
        - n_trades : nb de backtest_id distincts apparus dans trades

    On utilisera n_logs en priorité (si > 0), sinon n_trades.
    """
    n_logs = 0
    n_trades = 0
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            if logs_rowid_before is not None:
                cur.execute(
                    "SELECT COUNT(DISTINCT backtest_id) FROM logs WHERE rowid > ?;",
                    (int(logs_rowid_before),),
                )
                row = cur.fetchone()
                n_logs = int(row[0]) if row and row[0] is not None else 0
            if trades_rowid_before is not None:
                cur.execute(
                    "SELECT COUNT(DISTINCT backtest_id) FROM trades WHERE rowid > ?;",
                    (int(trades_rowid_before),),
                )
                row = cur.fetchone()
                n_trades = int(row[0]) if row and row[0] is not None else 0
    except Exception:
        pass
    return n_logs, n_trades


def _len_suggestions_file(path: str) -> int:
    """
    Lit le fichier JSON des suggestions GP et renvoie le nombre de candidats.
    Accepte soit un tableau brut, soit un dict contenant 'suggestions'/'candidates'/'params'.
    """
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            for key in ("suggestions", "candidates", "params"):
                if key in payload and isinstance(payload[key], list):
                    return len(payload[key])
        return 0
    except Exception:
        return 0


def _clip(v, vmin, vmax):
    return max(vmin, min(vmax, v))

def _norm_radius(r: float, rmin: float = TR_RMIN, rmax: float = TR_RMAX) -> float:
    """Normalise le rayon TR en ρ∈[0,1]. Robustifie aux valeurs hors bornes."""
    if rmax <= rmin:
        return 0.0
    return _clip((float(r) - rmin) / (rmax - rmin), 0.0, 1.0)

def _decide_gp_knobs(
    *,
    prev_best: float | None,
    new_best: float | None,
    current_tr_radius: float,
    batch_size: int,
    state: dict
) -> tuple[int, int, float, dict]:
    """
    Politique robuste pour (q_local, q_global, distance_thr), couplée au rayon TR:
      - distance_thr(ρ) = clip(D_base * (ε + ρ)^γ, [D_min, D_max])
      - q_local décroît ~ linéairement avec ρ, q_global croît avec ρ
      - nudge +/-1 en cas de progrès / stagnation
      - hysteresis ±DIST_HYST_FRAC et cooldown COOLDOWN_BATCHES
      - sécurité “packing”: si le batch précédent a peu rempli, on relâche d_thr

    Retourne (q_local, q_global, d_thr, state_maj).
    """
    # Étape 0: lecture des infos d'état
    last_dthr   = float(state.get("last_distance_thr", DIST_THR_BASE))
    cooldn      = int(state.get("policy_cooldown", 0))
    last_fill   = int(state.get("last_suggestions_count", 0))
    last_bsize  = int(state.get("last_batch_size", max(1, batch_size)))
    force_relax = bool(state.get("policy_force_relax_once", False))

    # Étape 1: progress signal
    if (prev_best is None) or (new_best is None):
        progress = False
    else:
        if prev_best <= 0 and new_best > 0:
            rel = float('inf')
        elif prev_best == 0:
            rel = 0.0
        else:
            rel = (new_best - prev_best) / abs(prev_best)
        progress = (rel > IMPROV_EPS)

    # Étape 2: mapping ρ -> q_local/q_global (lissé, borné)
    rho = _norm_radius(current_tr_radius, TR_RMIN, TR_RMAX)
    ql = Q_LOCAL_MIN + (Q_LOCAL_MAX - Q_LOCAL_MIN) * (1.0 - rho)
    qg = Q_GLOBAL_MIN + (Q_GLOBAL_MAX - Q_GLOBAL_MIN) * (rho)
    ql = int(round(_clip(ql, Q_LOCAL_MIN, Q_LOCAL_MAX)))
    qg = int(round(_clip(qg, Q_GLOBAL_MIN, Q_GLOBAL_MAX)))

    # nudge selon progrès/stagnation
    if progress:
        ql = _clip(ql + 1, Q_LOCAL_MIN, Q_LOCAL_MAX)
        qg = _clip(qg - 1, Q_GLOBAL_MIN, Q_GLOBAL_MAX)
    else:
        # On garde un peu de global si stagnation
        ql = _clip(ql - 1, Q_LOCAL_MIN, Q_LOCAL_MAX)
        qg = _clip(qg + 1, Q_GLOBAL_MIN, Q_GLOBAL_MAX)

    # Étape 3: distance_thr(ρ) + hysteresis + cooldown
    d_cand = DIST_THR_BASE * pow(DIST_MAP_EPS + rho, DIST_MAP_GAMMA)
    d_cand = float(_clip(d_cand, DIST_THR_MIN, DIST_THR_MAX))

    # Hysteresis: si le nouveau d_thr varie de <±8%, on garde l'ancien
    def _within_hyst(a, b, frac: float) -> bool:
        base = max(1e-9, abs(b))
        return abs(a - b) <= frac * base

    if cooldn > 0:
        # Cooldown actif → on gèle les knobs
        d_new = last_dthr
        ql_new, qg_new = int(state.get("last_q_local", ql)), int(state.get("last_q_global", qg))
        cooldn -= 1
    else:
        # packing safety (retour d'expérience du batch précédent)
        if last_bsize > 0:
            fill_rate = last_fill / float(last_bsize)
        else:
            fill_rate = 1.0
        d_new = d_cand
        ql_new, qg_new = ql, qg

        if force_relax or (fill_rate < PACK_WARN_FRAC):
            d_new = max(DIST_THR_MIN, last_dthr - PACK_DELTA)
            # reset le flag one-shot
            force_relax = False
            cooldn = max(cooldn, COOLDOWN_BATCHES)

        # hysteresis autour de last_dthr
        if _within_hyst(d_new, last_dthr, DIST_HYST_FRAC):
            d_new = last_dthr
        else:
            cooldn = max(cooldn, COOLDOWN_BATCHES)

    # Étape 4: garde-fous
    qg_new = max(int(qg_new), 2)  # exploration plancher
    # capacité: on s'assure que (ql+qg) ≤ batch_size
    if (ql_new + qg_new) > batch_size:
        overflow = (ql_new + qg_new) - batch_size
        # on compresse d'abord le global
        qg_new = max(2, qg_new - overflow)
        if (ql_new + qg_new) > batch_size:
            ql_new = max(1, batch_size - qg_new)

    # Mise à jour d'état
    state["last_q_local"]      = int(ql_new)
    state["last_q_global"]     = int(qg_new)
    state["last_distance_thr"] = float(d_new)
    state["last_batch_size"]   = int(batch_size)
    state["policy_cooldown"]   = int(cooldn)
    state["policy_force_relax_once"] = bool(force_relax)

    return int(ql_new), int(qg_new), float(d_new), state

def _inject_gp_env(q_local: int, q_global: int, distance_thr: float):
    """
    Fournit un fallback universel via variables d’environnement :
      GP_Q_LOCAL, GP_Q_GLOBAL, GP_DISTANCE_THR
    Ton gp_driver.py peut les lire si les flags CLI ne sont pas disponibles.
    """
    os.environ["GP_Q_LOCAL"]      = str(q_local)
    os.environ["GP_Q_GLOBAL"]     = str(q_global)
    os.environ["GP_DISTANCE_THR"] = f"{distance_thr:.6f}"
    print(f"[AUTO-ML] GP params → q_local={q_local}, q_global={q_global}, distance_thr={distance_thr:.3f}")

def _get_best_sharpe(db_path: str) -> float | None:
    """
    Retourne le meilleur Sharpe “365 (capé)” en privilégiant kpi_by_backtest :
      - filtre COALESCE(is_valid,1)=1 si dispo,
      - filtre COALESCE(flag_sharpe_outlier,0)=0 si dispo,
      - filtre ABS(OBJECTIVE_KPI) ≤ SHARPE_CAP,
      - fallback trades.sharpe_ratio si kpi_by_backtest indisponible.
    """
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # kpi_by_backtest présent ?
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='kpi_by_backtest';")
        if cur.fetchone() is not None:
            # Découvre les colonnes
            cur.execute("PRAGMA table_info(kpi_by_backtest);")
            cols = {r[1] for r in cur.fetchall()}
            has_is_valid = "is_valid" in cols
            has_outlier  = "flag_sharpe_outlier" in cols

            f1 = "AND COALESCE(is_valid,1)=1" if has_is_valid else ""
            f2 = "AND COALESCE(flag_sharpe_outlier,0)=0" if has_outlier else ""
            f3 = f"AND ABS({OBJECTIVE_KPI}) <= {float(SHARPE_CAP):.6f}"

            qry = f"""
                SELECT MAX({OBJECTIVE_KPI})
                FROM kpi_by_backtest
                WHERE {OBJECTIVE_KPI} IS NOT NULL
                  {f1} {f2} {f3};
            """
            cur.execute(qry)
            row = cur.fetchone()
            conn.close()
            return float(row[0]) if row and row[0] is not None else None

        # Fallback (héritage)
        cur.execute("SELECT MAX(sharpe_ratio) FROM trades WHERE sharpe_ratio IS NOT NULL;")
        row = cur.fetchone()
        conn.close()
        return float(row[0]) if row and row[0] is not None else None
    except Exception:
        return None

def count_eligible_strategies(db_path: str, thr: Thresholds) -> int:
    """
    Compte le nombre de stratégies “éligibles” selon les seuils stricts 'thr'
    directement dans la table 'kpi_by_backtest'.

    Args:
        db_path: Chemin SQLite final.
        thr: Seuils centralisés (voir config.Thresholds).

    Returns:
        int: nombre de lignes satisfaisant simultanément tous les critères “hard cut”.

    Notes:
        - On travaille sur 'kpi_by_backtest' (KPI standardisés), plus robuste que 'trades'.
        - Les colonnes doivent exister (migration déjà appliquée).
    """
    q = f"""
    SELECT COUNT(*) FROM kpi_by_backtest
    WHERE 
        COALESCE(nb_trades, 0) >= :min_trades
        AND COALESCE(pct_green_days, 0.0) >= :min_pct_green
        AND COALESCE(median_daily_pnl, -1e9) > :min_median_daily_pnl
        AND COALESCE(skew_daily_pnl, -1e9)   > :min_skew_daily_pnl
        AND ABS(COALESCE(max_drawdown, 0.0))       <= :max_mdd_abs
        AND COALESCE(top5_share, 1.0)        <= :max_top5_share
        AND COALESCE(ulcer_index, 0.0)       <= :max_ulcer
        AND COALESCE(sharpe_d_365, -1e9)     >= :min_sharpe_d
        AND COALESCE(profit_factor, -1e9)    >= :min_profit_factor
        AND ABS(COALESCE(sharpe_d_365, 0.0)) <= :cap
        AND COALESCE(is_valid, 1) = 1
        AND COALESCE(flag_sharpe_outlier, 0) = 0;
    """

    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(q, {
                "min_trades": int(thr.min_trades),
                "min_pct_green": float(thr.min_pct_green),
                "min_median_daily_pnl": float(thr.min_median_daily_pnl),
                "min_skew_daily_pnl": float(thr.min_skew_daily_pnl),
                "max_mdd_abs": float(thr.max_mdd_abs),
                "max_top5_share": float(thr.max_top5_share),
                "max_ulcer": float(thr.max_ulcer),
                "min_sharpe_d": float(thr.min_sharpe_d),
                "min_profit_factor": float(thr.min_profit_factor),
                "cap": float(SHARPE_CAP),
            })

            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0
    except Exception as e:
        logger.warning(f"count_eligible_strategies: fallback 0 (err={e})")
        return 0

def _count_pool(db_path: str) -> tuple[int, int]:
    """
    Compte (total_ids, valid_ids) dans kpi_by_backtest d'une base donnée.
    valid_ids = is_valid=1 et flag_sharpe_outlier=0 (cap Sharpe déjà appliqué upstream).
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(DISTINCT backtest_id) FROM kpi_by_backtest")
            total = int(cur.fetchone()[0] or 0)
            cur.execute("""
                SELECT COUNT(DISTINCT backtest_id)
                FROM kpi_by_backtest
                WHERE COALESCE(is_valid,1)=1 AND COALESCE(flag_sharpe_outlier,0)=0
            """)
            valid = int(cur.fetchone()[0] or 0)
            return total, valid
    except Exception:
        return 0, 0

def _fetch_latest_soft_hard(db_path: str) -> tuple[int | None, int | None, dict]:
    """
    Récupère les derniers compteurs N_soft et N_hard depuis la table adaptive_decisions
    du fichier SQLite `db_path`.

    Politique:
      - N_soft: dernière ligne avec source LIKE '%soft%' si dispo, sinon fallback sur dernière ligne.
      - N_hard: dernière ligne avec source LIKE '%post_hard%' si dispo, sinon fallback sur dernière ligne.

    Returns
    -------
    (n_soft, n_hard, meta)
        n_soft : int|None
        n_hard : int|None
        meta   : dict (sources utilisées + flags de fallback)
    """
    meta = {"fallback_soft": True, "fallback_hard": True, "source_soft": None, "source_hard": None}
    def _one(conn, query: str) -> int | None:
        try:
            cur = conn.execute(query)
            row = cur.fetchone()
            if row and row[0] is not None:
                return int(row[0])
        except Exception:
            return None
        return None

    try:
        with sqlite3.connect(db_path) as conn:
            # N_soft
            n_soft = _one(conn, "SELECT n_eff FROM adaptive_decisions WHERE source LIKE '%soft%' ORDER BY ts_utc DESC LIMIT 1;")
            if n_soft is not None:
                meta["fallback_soft"] = False
                meta["source_soft"] = "like:%soft%"
            else:
                n_soft = _one(conn, "SELECT n_eff FROM adaptive_decisions ORDER BY ts_utc DESC LIMIT 1;")
                meta["source_soft"] = "any"

            # N_hard
            n_hard = _one(conn, "SELECT n_eff FROM adaptive_decisions WHERE source LIKE '%post_hard%' ORDER BY ts_utc DESC LIMIT 1;")
            if n_hard is not None:
                meta["fallback_hard"] = False
                meta["source_hard"] = "like:%post_hard%"
            else:
                n_hard = _one(conn, "SELECT n_eff FROM adaptive_decisions ORDER BY ts_utc DESC LIMIT 1;")
                meta["source_hard"] = "any"

            return n_soft, n_hard, meta
    except Exception:
        return None, None, meta

def _sanity_check_kpi_vs_features(db_path: str, objective_col: str = "sharpe_d_365", limit: int = 200) -> None:
    """
    Vérifie qu'il existe bien des features (logs/trades) dans GOOD pour les backtest_id
    considérés 'valid & !outlier & objective non NULL' côté KPI.

    Loggue :
      - #KPI candidats
      - #avec logs, #avec trades
      - Top IDs manquants (s'il y en a)
    """
    try:
        with sqlite3.connect(db_path) as conn:
            # 1) Candidats KPI
            q = f"""
                SELECT DISTINCT backtest_id
                FROM kpi_by_backtest
                WHERE COALESCE(is_valid, 1) = 1
                  AND COALESCE(flag_sharpe_outlier, 0) = 0
                  AND {objective_col} IS NOT NULL
                LIMIT {int(limit)}
            """
            kpi_ids = pd.read_sql_query(q, conn)["backtest_id"].astype(str).tolist()

            if not kpi_ids:
                print("🔎 Sanity-check: aucun candidat KPI (valid & !outlier) dans GOOD.")
                return

            # 2) Présence dans logs / trades
            ids_param = ",".join(["?"] * len(kpi_ids))
            have_logs = pd.read_sql_query(
                f"SELECT DISTINCT backtest_id FROM logs WHERE backtest_id IN ({ids_param})",
                conn, params=kpi_ids
            )["backtest_id"].astype(str).tolist()

            have_trades = pd.read_sql_query(
                f"SELECT DISTINCT backtest_id FROM trades WHERE backtest_id IN ({ids_param})",
                conn, params=kpi_ids
            )["backtest_id"].astype(str).tolist()

            set_kpi   = set(kpi_ids)
            set_logs  = set(have_logs)
            set_tr    = set(have_trades)

            miss_logs = list(set_kpi - set_logs)
            miss_tr   = list(set_kpi - set_tr)

            print(f"🔎 Sanity-check (GOOD): KPI candidats={len(set_kpi)} | "
                  f"avec logs={len(set_logs)} | avec trades={len(set_tr)}")

            if miss_logs[:10]:
                print("   • IDs KPI sans logs (top 10) :", miss_logs[:10])
            if miss_tr[:10]:
                print("   • IDs KPI sans trades (top 10):", miss_tr[:10])

    except Exception as e:
        print(f"⚠️ Sanity-check KPI↔features impossible: {e}")

def load_standardized_kpis(db_path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Charge la table 'kpi_by_backtest' (KPI standardisés) avec un sous-ensemble de colonnes.

    Args:
        db_path (str): Chemin de la base SQLite.
        columns (list[str] | None): Colonnes désirées; si None, charge toutes les colonnes.

    Returns:
        pd.DataFrame: KPI standardisés, colonnes au minimum ['backtest_id','iteration'] si présentes.

    Notes:
        - Utile pour diagnostiquer l'AutoML ou tracer l'évolution de l'objectif OBJECTIVE_KPI.
        - Remplace les ±Inf/NaN par NA pandas pour faciliter les filtres.
    """
    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM kpi_by_backtest", conn)
        except Exception:
            return pd.DataFrame()
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        if keep:
            df = df[keep]
    return df.replace([float("inf"), float("-inf")], pd.NA)

def _select_and_persist_portfolio(db_path: str, batch_id: int, k: int = 100, max_corr: float = 0.30):
    """
    Sélectionne un portefeuille décorrélé à partir de stratégies éligibles
    (filtre 'consistency-first' intra-day) puis persiste en base.
    """
    ensure_selected_table_exists(db_path)

    with sqlite3.connect(db_path) as conn:
        trades = pd.read_sql_query(
            "SELECT backtest_id, exit_time AS timestamp, pnl_net, balance FROM trades",
            conn
        )

    if trades.empty:
        print("⚠️ Sélection portefeuille: aucun trade disponible.")
        return

    # --- KPIs par stratégie ---
    elig = []
    for bid, grp in trades.groupby("backtest_id"):
        try:
            # df_logs peut être None ici (toléré par compute_intraday_consistency_kpis)
            kpi = mx.compute_intraday_consistency_kpis(
                df_trades=grp,
                df_logs=None,
                price_col="price",           # absent ici → certaines métriques seront NaN, c’est acceptable
                timestamp_col="timestamp",
                fee_rate=FEE_RATE,
                ann_factor=ANN_FACTOR_DAILY  # annualisation crypto (365)
            )

            kpi["backtest_id"] = str(bid)
            elig.append(kpi)
        except Exception as e:
            print(f"⚠️ KPIs intraday impossibles pour {bid}: {e}")

    if not elig:
        print("⚠️ Sélection portefeuille: aucun KPI calculable.")
        return

    dfk = pd.DataFrame(elig).replace([float('inf'), float('-inf')], pd.NA).dropna(how="all")

    # Harmonise les noms de colonnes (selon ta version de metrics_core)
    if "pct_green_days" not in dfk.columns and "green_days_ratio" in dfk.columns:
        dfk["pct_green_days"] = dfk["green_days_ratio"]
    # garde-fou si certaines colonnes sont absentes
    for col in ["nb_trades", "pct_green_days", "sharpe_d_365", "median_daily_pnl", "skew_daily_pnl"]:
        if col not in dfk.columns:
            dfk[col] = pd.NA

    # compat héritage : si 365 absent mais 'sharpe_daily_ann' fourni, mappe-le
    if dfk["sharpe_d_365"].isna().all() and "sharpe_daily_ann" in dfk.columns:
        dfk["sharpe_d_365"] = dfk["sharpe_daily_ann"]

    # --- Filtres d'éligibilité (cohérents avec export_good_iteration) ---
    MIN_TRADES = 30
    MIN_PCT_GREEN = 0.55
    MIN_SHARPE_D = 1.0  # même seuil, appliqué au Sharpe 365

    mask = (
        (dfk["nb_trades"].fillna(0) >= MIN_TRADES) &
        (dfk["pct_green_days"].fillna(0.0) >= MIN_PCT_GREEN) &
        (dfk["sharpe_d_365"].fillna(0.0) >= MIN_SHARPE_D) &
        (dfk["median_daily_pnl"].fillna(-1e9) > 0.0) &
        (dfk["skew_daily_pnl"].fillna(-1e9) > 0.0)
    )

    keep_ids = set(dfk.loc[mask, "backtest_id"].dropna().astype(str))

    if not keep_ids:
        print("⚠️ Sélection portefeuille: aucun candidat ne passe les filtres (on utilisera tout le set).")
        filtered_trades = trades.copy()
        filtered_trades["backtest_id"] = filtered_trades["backtest_id"].astype(str)
    else:
        filtered_trades = trades[trades["backtest_id"].astype(str).isin(keep_ids)].copy()

    if filtered_trades.empty:
        print("⚠️ Sélection portefeuille: trades vides après filtre.")
        return

    # --- Sélection gloutonne avec contrainte de corrélation ---
    res = select_strategy_portfolio(
        filtered_trades,
        max_corr=max_corr,
        k=k,
        periods_per_year=ANN_FACTOR_DAILY,  # annualisation 365 pour le Sharpe de portefeuille
        annualize=True
    )

    insert_selected_strategies(
        db_path=db_path,
        selected_ids=res["selected_ids"],
        portfolio_sharpe=res["portfolio_sharpe"],
        mean_pairwise_corr=res["mean_pairwise_corr"],
        batch_id=batch_id,
        stats_df=res["stats"],
    )

    print(f"✅ Portefeuille sélectionné (k≤{k}, max_corr={max_corr:.2f}) : "
        f"{len(res['selected_ids'])} strats, Sharpe_ptf(ann=365)={res['portfolio_sharpe']:.3f}, "
        f"mean|corr|={res['mean_pairwise_corr']:.3f}")

    
def ensure_market_data(pair: str, intervals: list[str], nb_days: int, db_path: str, timestamp_source: str = "open"):
    """
    Garante que la base OHLCV locale est prête pour le backtest.
    - Si couverture suffisante → ne fait rien (offline).
    - Sinon:
        • si ALLOW_DOWNLOAD_IF_MISSING: active LIVE_DEPS (temporaire), télécharge via Binance,
          sauvegarde en base, puis repasse en BACKTEST_DEPS.
        • sinon: lève une erreur explicite.
    """
    ok = has_market_data_coverage(db_path, intervals, nb_days)
    if ok:
        print("✅ Base OHLCV locale OK (couverture suffisante).")
        return

    if not ALLOW_DOWNLOAD_IF_MISSING:
        raise RuntimeError("⛔ Données manquantes et ALLOW_DOWNLOAD_IF_MISSING=False (backtest strict).")

    print("⚠️ Données insuffisantes → téléchargement LIVE (Binance) pour remplir la base...")

    # Active live pour les dépendances du module signal (si tu veux interroger la paire)
    sse.DEFAULT_DEPS = sse.LIVE_DEPS
    try:
        data_dict = fetch_historical_data(pair, intervals, nb_days, timestamp_source=timestamp_source)
        save_market_data_to_sqlite(data_dict, db_path, base_interval='5m')
        print("✅ Données téléchargées et sauvegardées. Re-bascule offline.")
    finally:
        # Revenir en backtest strict (zéro I/O)
        sse.DEFAULT_DEPS = sse.BACKTEST_DEPS

def precompute_shap_importance(db_path: str, timeout_sec: int = 600) -> bool:
    """
    Lance surrogate_modeling.py une fois AVANT le GP pour produire ARTIFACTS/shap_importance.json.
    Retourne True si succès, False sinon. Isolation par subprocess pour éviter les fuites d'état.
    """
    shap_json = ARTIFACTS_DIR / "shap_importance.json"
    # Si un fichier récent existe déjà, on considère que c'est bon.
    if shap_json.exists():
        try:
            age_sec = time.time() - shap_json.stat().st_mtime
            if age_sec < 3600:  # moins d'une heure = OK
                print(f"🧭 SHAP JSON déjà présent (âge {int(age_sec)}s) → réutilisation.")
                return True
        except Exception:
            pass

    print("🧭 Préparation SHAP/importance AVANT GP...")
    try:
        proc = subprocess.run(
            ["python", "surrogate_modeling.py", db_path],
            capture_output=True, text=True, timeout=timeout_sec
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"⚠️ Pré-SHAP a renvoyé un code {proc.returncode} (fallback isotrope/ARD).")
            print(proc.stderr)
            return False
        # Vérifie présence fichier
        if shap_json.exists():
            print(f"✅ SHAP JSON prêt pour GP : {shap_json}")
            return True
        print("⚠️ Pas de shap_importance.json trouvé après surrogate_modeling (fallback isotrope/ARD).")
        return False
    except Exception as e:
        print(f"⚠️ Échec pré-SHAP ({e}) → fallback isotrope/ARD.")
        return False

def run_pipeline_gp_parallel_backtest_visualize(n_total_trials, batch_size):
    """
    Exécute une boucle complète d'optimisation bayésienne, de backtesting
    et d'analyse explicative sur une stratégie de trading.

    Cette fonction orchestre le pipeline suivant :
        1. Génération de `batch_size` suggestions d'hyperparamètres via un
           modèle de processus gaussien (GP) avec BoTorch.
        2. Exécution des backtests pour chaque configuration proposée,
           avec journalisation dans une base SQLite.
        3. Visualisation de la performance stratégique à l'aide de SHAP,
           PDP croisé et UMAP, déclenchée après chaque batch.

    Args:
        n_total_trials (int): Nombre total de suggestions à générer (et de backtests à exécuter).
        batch_size (int): Taille de chaque lot de suggestions générées avant visualisation.

    Comportement :
        - La base de logs est enrichie à chaque itération.
        - Les visualisations sont exportées dans le dossier `./figs_surrogate`.
        - La mémoire est nettoyée entre les runs pour éviter les fuites.

    Exemple :
        run_pipeline_gp_parallel_backtest_visualize(n_total_trials=30, batch_size=5)

    Remarque :
        Cette fonction est conçue pour être le point d'entrée principal du pipeline.
        Elle peut être intégrée dans une boucle d'AutoML ou appelée indépendamment.
    """
    print(f"🎯 OBJECTIVE_KPI = {OBJECTIVE_KPI} (annualisation daily={ANN_FACTOR_DAILY}, cap=±{SHARPE_CAP})")
    print(f"✅ GP/BO source = {GOOD_STRAT_DB_PATH} (pool SOFT), "
      f"export vitrine (HARD) → {BEST_STRAT_DB_PATH}")

    # Vérifie qu'on a bien de la matière (features) pour les IDs KPI sélectionnables
    _sanity_check_kpi_vs_features(GOOD_STRAT_DB_PATH, objective_col=OBJECTIVE_KPI, limit=500)


    # ✅ S'assurer que la base OHLCV est prête (offline prioritaire)
    ensure_market_data(PAIR, INTERVALS, NB_DAYS, DATA_PATH, timestamp_source=TIMESTAMP_SOURCE)

    # ✅ Schéma résultats/export : crée tables si absentes + ajoute colonnes manquantes (idempotent)
    try:
        # Base principale (logs + KPI standardisés)
        init_sqlite_database(FINAL_DB_PATH)
        migrate_sqlite_schema(FINAL_DB_PATH)

        # Bases d’export/filtrage (optionnelles) — idempotentes si déjà prêtes
        try:
            init_sqlite_database(BEST_STRAT_DB_PATH)
            migrate_sqlite_schema(BEST_STRAT_DB_PATH)
        except Exception:
            pass
        try:
            init_sqlite_database(GOOD_STRAT_DB_PATH)
            migrate_sqlite_schema(GOOD_STRAT_DB_PATH)
        except Exception:
            pass
    except Exception as e:
        logger.exception("Échec init/migration DB")
        send_alert(f"🚨 DB init/migration failed: {type(e).__name__}: {e}")
        raise

    # État du trust-region (chargé / initialisé)
    _state = _load_tr_state()
    current_tr_radius = float(_state.get("tr_radius", DEFAULT_TR_RADIUS))
    print(f"🔧 Rayon TR initial: {current_tr_radius:.3f}")

    # Meilleur Sharpe courant AVANT le batch (global, pour mesurer le progrès)
    prev_best_global = _get_best_sharpe(FINAL_DB_PATH)
    print(f"📈 Meilleur Sharpe 365 (capé≤{SHARPE_CAP}) actuel (avant boucle): {prev_best_global}")

    # --- Préparer l'environnement (dossiers) et tenter un pré-SHAP pour anisotropie immédiate ---
    _ensure_dirs()
    _ = precompute_shap_importance(FINAL_DB_PATH)  # On ignore le bool ici : GP gère aussi ARD/isotrope

    for i in range(0, n_total_trials, batch_size):
        batch_start = i + 1
        batch_end = i + batch_size
        print(f"\n📡 Phase GP [{batch_start} → {batch_end}]")

        # Fichier temporaire pour stocker les suggestions
        suggestions_file = f"suggestions_batch_{batch_start}_{batch_end}.json"

        # --- Étape 1 : Génération GP via subprocess
        # 1) on propose q_local/q_global/distance_thr dynamiquement (en fonction du progrès observé depuis le dernier batch)
        prev_best_state = _state.get("prev_best", None)   # best enregistré (TR state)
        q_local, q_global, distance_thr, _state = _decide_gp_knobs(
            prev_best=prev_best_state,
            new_best=prev_best_global,
            current_tr_radius=current_tr_radius,
            batch_size=batch_size,
            state=_state,
        )
        _inject_gp_env(q_local, q_global, distance_thr)

        # Persistance immédiate des knobs appliqués (journalisation)
        _save_tr_state({
            "tr_radius": float(_state.get("tr_radius", current_tr_radius)),
            "prev_best": prev_best_global,
            "zero_gp_streak": int(_state.get("zero_gp_streak", 0)),
            "last_q_local": int(q_local),
            "last_q_global": int(q_global),
            "last_distance_thr": float(distance_thr),
            "policy_cooldown": int(_state.get("policy_cooldown", 0)),
        })

        # 2) on tente de passer aussi en CLI (si ton gp_driver.py les supporte),
        #    sinon il ignorera simplement les arguments supplémentaires et utilisera l'env.
        gp_cmd = [
            "python", "gp_driver.py",
            GOOD_STRAT_DB_PATH,                 # db source pour GP
            str(batch_size),                    # n_suggestions
            suggestions_file,                   # fichier de sortie
            str(current_tr_radius),             # TR radius
            "--q-local",  str(q_local),
            "--q-global", str(q_global),
            "--distance-thr", f"{distance_thr:.6f}",
        ]

        result = subprocess.run(gp_cmd, capture_output=True, text=True)

        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Erreur GP : {result.stderr}")
            continue

        # === DIAGNOSTIC GP : nombre réel de suggestions générées ===
        suggestions_count = _len_suggestions_file(suggestions_file)

        # MAJ télémétrie de faisabilité pour la prochaine policy
        _state["last_suggestions_count"] = int(suggestions_count)
        _state["last_batch_size"]        = int(batch_size)

        if suggestions_count == 0:
            _state["zero_gp_streak"] = int(_state.get("zero_gp_streak", 0)) + 1
            streak = _state["zero_gp_streak"]
            print(
                f"🚨 GP a renvoyé 0 suggestion(s) (streak={streak}). "
                f"Probable hyperparamétrage trop restrictif (TR / distance_thr / bornes)."
            )
            # Persistance immédiate, pour suivi par la boucle principale
            _save_tr_state({
                "tr_radius": current_tr_radius,
                "prev_best": prev_best_global,
                "zero_gp_streak": streak,
                "last_q_local": int(_state.get("last_q_local", Q_LOCAL_BASE)),
                "last_q_global": int(_state.get("last_q_global", Q_GLOBAL_BASE)),
                "last_distance_thr": float(_state.get("last_distance_thr", DIST_THR_BASE)),
                "policy_cooldown": int(_state.get("policy_cooldown", 0)),
                "last_suggestions_count": int(suggestions_count),
                "last_batch_size": int(batch_size),
            })
            # Inutile de lancer les backtests si aucune candidate
            continue
        else:
            if int(_state.get("zero_gp_streak", 0)) > 0:
                print("ℹ️ GP a repris des suggestions → remise à zéro de zero_gp_streak.")
            _state["zero_gp_streak"] = 0

        fill_rate = suggestions_count / float(batch_size)
        if fill_rate < PACK_WARN_FRAC:
            # On relâchera d_thr d'un cran au prochain batch
            _state["policy_force_relax_once"] = True
            print(f"⚠️ Fill-rate partiel ({fill_rate:.2%}<{PACK_WARN_FRAC:.0%}) → détente distance_thr au prochain batch.")

        if fill_rate <= PACK_FAIL_FRAC:
            # Petit élargissement immédiat de TR (secours), borné
            new_r = min(TR_RMAX, current_tr_radius * 1.10)
            if new_r > current_tr_radius:
                print(f"🔧 TR radius élargi pour faisabilité (no-fill sévère): {current_tr_radius:.3f} → {new_r:.3f}")
                current_tr_radius = new_r

        print(f"📦 GP a proposé {suggestions_count} candidate(s).")

        # Marqueurs ROWID avant backtest pour mesurer le volume réellement exécuté
        _logs_rowid_before   = _get_max_rowid(FINAL_DB_PATH, "logs")
        _trades_rowid_before = _get_max_rowid(FINAL_DB_PATH, "trades")

        print(f"⚙️ Lancement backtests parallèles sur {suggestions_count} suggestion(s)...")

        result = subprocess.run([
            "python", "parallel_backtest_launcher.py",  # script externe
            suggestions_file
        ], capture_output=True, text=True)

        # === DIAGNOSTIC BACKTEST : combien de candidats ont vraiment tourné ? ===
        n_logs, n_trades = _count_new_backtests_since(
            FINAL_DB_PATH,
            _logs_rowid_before,
            _trades_rowid_before,
        )
        n_backtested = n_logs if n_logs > 0 else n_trades
        print(
            f"🧪 Backtests réalisés: {n_backtested} "
            f"(logs-based={n_logs}, trades-based={n_trades}) | "
            f"Comparé aux {suggestions_count} suggestion(s) GP."
        )

        # --- Mise à jour du rayon TR selon le progrès observé sur ce batch
        new_best_global = _get_best_sharpe(FINAL_DB_PATH)
        print(f"📉/📈 Meilleur Sharpe 365 (capé≤{SHARPE_CAP}) après batch {batch_start}-{batch_end}: {new_best_global}")

        current_tr_radius = adapt_tr_radius(
            prev_best=prev_best_global,
            new_best=new_best_global,
            tr_radius=current_tr_radius,
            shrink=0.85,   # exploitation si progrès
            expand=1.25,   # exploration si pas de progrès
            rmin=0.05,
            rmax=0.40,
        )

        print(f"🧭 Nouveau rayon TR = {current_tr_radius:.3f} (Sharpe 365 capé ancien={prev_best_global}, nouveau={new_best_global})")

        # Persistance état (rayon + dernier best + streak 0-candidate)
        _save_tr_state({
            "tr_radius": current_tr_radius,
            "prev_best": new_best_global,
            "zero_gp_streak": int(_state.get("zero_gp_streak", 0)),
            "last_q_local": int(_state.get("last_q_local", Q_LOCAL_BASE)),
            "last_q_global": int(_state.get("last_q_global", Q_GLOBAL_BASE)),
            "last_distance_thr": float(_state.get("last_distance_thr", DIST_THR_BASE)),
            "policy_cooldown": int(_state.get("policy_cooldown", 0)),
            "last_suggestions_count": int(_state.get("last_suggestions_count", 0)),
            "last_batch_size": int(_state.get("last_batch_size", 0)),
        })
        _state["prev_best"] = new_best_global

        # Pour la prochaine itération, l'ancien "best" devient le nouveau
        prev_best_global = new_best_global

        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Erreur backtest : {result.stderr}")
            continue

        # --- Étape 3 : SHAP via subprocess
        print("📊 Génération des visualisations SHAP...")

        result = subprocess.run([
            "python", "surrogate_modeling.py",  # script externe
            FINAL_DB_PATH
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.returncode != 0:
            print(f"⚠️ SHAP a planté dans le batch {batch_start} → {batch_end} (exit code {result.returncode})")
        else:
            print(f"✅ SHAP terminé proprement pour le batch {batch_start} → {batch_end}")

        # --- Lissage EMA des importances SHAP (pour guider le GP du prochain batch)
        try:
            _ema_shap_importance(alpha=SHAP_EMA_ALPHA)
        except Exception as e:
            print(f"⚠️ EMA SHAP impossible: {e}")

        # --- Étape 4 : Export double passe (SOFT → good_iterations.db / HARD → filtered_full_copy.db)
        print("Analyse et extraction des stratégies (double passe SOFT/HARD)...")

        # 🛰️ Télémétrie PRE-EXPORT: derniers N_soft/N_hard vus par adaptive_decisions (FINAL_DB_PATH)
        try:
            pre_soft, pre_hard, meta_counts = _fetch_latest_soft_hard(FINAL_DB_PATH)
            print(f"🛰️ PRE-EXPORT | N_soft={pre_soft} (src={meta_counts.get('source_soft')}, fallback={meta_counts.get('fallback_soft')}) | "
                  f"N_hard={pre_hard} (src={meta_counts.get('source_hard')}, fallback={meta_counts.get('fallback_hard')})")
        except Exception as _e:
            print(f"🛰️ PRE-EXPORT | N_soft/N_hard indisponibles: {type(_e).__name__}: {_e}")

        # NOTE: export_good_iteration.py exécute désormais EN INTERNE la double passe :
        #  • SOFT : alimente good_iterations.db (pool GP/BO) avec des seuils larges (via config.SOFT)
        #  • HARD : alimente filtered_full_copy.db (vitrine) avec les seuils “durs” (T + HARD)
        # On lui passe ici les seuils “perf stricts” (T) pour la passe HARD; les seuils SOFT
        # sont pris dans config.py (ENV), donc aucun argument supplémentaire n’est nécessaire.

        result = subprocess.run([
            "python", "export_good_iteration.py",
            FINAL_DB_PATH,                  # src_db
            BEST_STRAT_DB_PATH,             # dst_db
            "--min_trades", str(T.min_trades),
            "--min_pct_green", str(T.min_pct_green),
            "--min_sharpe_d", str(T.min_sharpe_d),
            "--min_median_daily_pnl", str(T.min_median_daily_pnl),
            "--min_skew_daily_pnl", str(T.min_skew_daily_pnl),
            "--max_mdd_abs", str(T.max_mdd_abs),
            "--max_top5_share", str(T.max_top5_share),
            "--max_ulcer", str(T.max_ulcer),
            "--min_profit_factor", str(T.min_profit_factor),
        ], capture_output=True, text=True)

        # Récap des pools après export (utile pour monitorer la progression/diversité)
        try:
            g_tot, g_val = _count_pool(GOOD_STRAT_DB_PATH)
            b_tot, b_val = _count_pool(BEST_STRAT_DB_PATH)
            print(f"📦 Pools mis à jour → GOOD(good_iterations): valid={g_val}/{g_tot} | "
                f"BEST(filtered_full_copy): valid={b_val}/{b_tot}")
        except Exception as e:
            print(f"[WARN] Impossible de compter les pools GOOD/BEST: {e}")

        # 🛰️ Télémétrie POST-EXPORT: N_soft/N_hard après la passe
        try:
            post_soft, post_hard, meta_counts2 = _fetch_latest_soft_hard(FINAL_DB_PATH)
            print(f"🛰️ POST-EXPORT | N_soft={post_soft} (src={meta_counts2.get('source_soft')}, fallback={meta_counts2.get('fallback_soft')}) | "
                  f"N_hard={post_hard} (src={meta_counts2.get('source_hard')}, fallback={meta_counts2.get('fallback_hard')})")
        except Exception as _e:
            print(f"🛰️ POST-EXPORT | N_soft/N_hard indisponibles: {type(_e).__name__}: {_e}")

        # --- Étape 5 : Sélection portefeuille décorrélé & persistance
        try:
            batch_id = batch_end  # ou (i // batch_size) + 1 si tu préfères un compteur simple
            _select_and_persist_portfolio(FINAL_DB_PATH, batch_id=batch_id, k=100, max_corr=0.30)
        except Exception as e:
            print(f"⚠️ Sélection portefeuille impossible sur ce batch : {e}")


if __name__ == "__main__":
    """
    Point d'entrée “main” : boucle AutoML jusqu’à atteindre un quota de stratégies
    éligibles selon les seuils “hard cut” centralisés (config.load_thresholds).

    Comportement :
      - Lit les seuils (ENV → defaults).
      - Compte les stratégies éligibles dans kpi_by_backtest.
      - Tant que le compte < target_good : exécute 1 batch (BATCH_SIZE suggestions)
        via run_pipeline_gp_parallel_backtest_visualize(n_total_trials=BATCH_SIZE).
      - Recompte et boucle, avec une pause courte entre itérations (CLI --sleep-sec).

    Arguments CLI :
      --target-good N   : quota de stratégies éligibles (défaut 100)
      --sleep-sec S     : pause entre itérations (défaut 5s)
    """
    setup_logging()
    try:
        parser = argparse.ArgumentParser(description="AutoML loop until N eligible strategies are found.")
        parser.add_argument("--target-good", type=int, default=TARGET_GOOD_STRATS_DEFAULT,
                            help=f"Quota de stratégies éligibles à atteindre (def: {TARGET_GOOD_STRATS_DEFAULT})")
        parser.add_argument("--sleep-sec", type=int, default=5, help="Pause entre itérations (secondes).")
        args, _unknown = parser.parse_known_args()

        target_good = max(1, int(args.target_good))
        sleep_sec   = max(0, int(args.sleep_sec))

        logger.info(f"🎯 Quota cible: {target_good} stratégies éligibles | Seuils: {T}")
        curr = count_eligible_strategies(FINAL_DB_PATH, T)
        logger.info(f"📊 Stratégies éligibles (Sharpe 365 capé, valid/!outlier) au départ: {curr}/{target_good}")

        # N_eff (post-HARD avec cap) côté driver, pour tracer la trajectoire AutoML
        try:
            _adaptive_log_n_eff_dbpath(FINAL_DB_PATH, curr, source="automl_main:eligible_hard_start",
                                    meta={"cap": float(SHARPE_CAP),
                                            "thr": T.__dict__ if hasattr(T, "__dict__") else str(T)})
        except Exception as e:
            print(f"[ADAPTIVE][WARN] log N_eff (start) échoué: {type(e).__name__}: {e}")

        # Boucle principale : 1 batch par itération → re-comptage
        while curr < target_good:
            logger.info(f"🚀 Lancement 1 batch AutoML (taille={BATCH_SIZE})...")
            try:
                # On exécute volontairement UN seul batch par itération
                run_pipeline_gp_parallel_backtest_visualize(n_total_trials=BATCH_SIZE, batch_size=BATCH_SIZE)
            except Exception as e:
                logger.exception("Crash batch AutoML")
                send_alert(f"🚨 Crash AutoML batch: {type(e).__name__}: {e}")
                # On ne raise pas pour permettre la reprise automatique de la boucle
            # Recompter après le batch
            curr = count_eligible_strategies(FINAL_DB_PATH, T)
            logger.info(f"📈 Stratégies éligibles (Sharpe 365 capé, valid/!outlier) après batch: {curr}/{target_good}")

            try:
                _adaptive_log_n_eff_dbpath(
                    FINAL_DB_PATH,
                    curr,
                    source="automl_main:eligible_hard_after_batch",
                    meta={
                        "cap": float(SHARPE_CAP),
                        "thr": T.__dict__ if hasattr(T, "__dict__") else str(T),
                    },
                )
            except Exception as e:
                print(f"[ADAPTIVE][WARN] log N_eff (after_batch) échoué: {type(e).__name__}: {e}")

            if curr < target_good and sleep_sec > 0:
                time.sleep(sleep_sec)

        logger.info("✅ Quota atteint : assez de stratégies éligibles.")
    except Exception as e:
        logger.exception("Crash pipeline AutoML (main loop)")
        send_alert(f"🚨 Crash AutoML pipeline (main): {type(e).__name__}: {e}")
        raise

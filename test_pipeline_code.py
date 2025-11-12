"""
====================================================================================
Fichier : test_pipeline_code.py
Objectif : Automatisation de la mise à jour des métriques + génération de backtests via BoTorch
====================================================================================

Description :
Ce script assure deux fonctions critiques dans un pipeline d’exploration de stratégies
quantitatives :

1. **Nettoyage et enrichissement de la base de données (`good_iterations.db`)**
   - Vérification de la présence des colonnes nécessaires dans la table `trades`
     (sharpe_ratio, max_drawdown, profit_factor, win_rate)
   - Calcul de ces métriques pour chaque `backtest_id`, sur la base des PnL nets agrégés
   - Propagation cohérente des scores à toutes les lignes du `backtest_id` concerné

2. **Lancement de nouveaux backtests via des suggestions d’hyperparamètres BoTorch**
   - Utilise un modèle bayésien pour générer intelligemment des combinaisons prometteuses
   - Exécute `run_backtest_from_params()` sur chaque combinaison
   - Alimente dynamiquement la base `trades` avec de nouvelles observations

Rôle dans le pipeline :
Ce script peut être vu comme une brique *AutoML supervisée* dans un cadre de
recherche de stratégie algorithmique :
    - Il enrichit les données historiques avec des métriques robustes
    - Il stimule activement de nouvelles explorations via BoTorch (Bayesian Optimization)
    - Il maintient une base SQL exploitable par le RL ou la sélection statistique

Spécificités :
    - Métriques calculées au niveau agrégé (par backtest), évitant les biais ligne par ligne
    - Méthodologie robuste de calcul : Sharpe, drawdown, profit factor, win rate
    - Les suggestions BoTorch permettent une amélioration dirigée, non aléatoire, du front Pareto

À combiner avec :
    - `good_iterations.db` (base centrale du pipeline)
    - `gp_driver.py` (modélisation BoTorch)
    - `run_backtest_from_params()` (exécution d’un backtest isolé)
    - `generate_pca_test_features.py` (préparation post-backtest pour RL)
    - `ppo_train.py` (agent RL utilisant les stratégies les plus robustes)

Utilisation recommandée :
Ce script peut être lancé à intervalles réguliers pour :
    - nettoyer et stabiliser la base SQL
    - déclencher des backtests intelligents (Bayesian Exploration)
    - servir de boucle fermée dans un processus *AutoML RL-based Backtesting*

Auteur : Moncoucut Brandon
Version : Juin 2025
"""

# === Imports fondamentaux ===
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from gp_driver import suggest_from_gp
from backtest_interface_code import run_backtest_from_params

# ✅ KPI standardisés
import metrics_core as mx

# ✅ Helpers DB (init/migration + upsert KPI)
from backtest_db_manager import (
    init_sqlite_database,
    migrate_sqlite_schema,
    upsert_kpis,
)

# === CONFIGURATION ===
USE_BOTORCH = True
N_BOTORCH_TRIALS = 3
DB_PATH = "/Users/brandonmoncoucut/Desktop/Najas_king/log_sqlite/good_iterations.db"
TRADES_TABLE = "trades"

def _sanitize_kpi(kpi: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remplace NaN/Inf par None pour compatibilité SQLite/JSON.
    Conserve uniquement des scalaires numériques (int/float) ou None.

    Args:
        kpi: Dictionnaire de KPI calculés.

    Returns:
        Dict[str, Any]: KPI nettoyés.
    """
    clean: Dict[str, Any] = {}
    if not isinstance(kpi, dict):
        return clean
    for k, v in kpi.items():
        try:
            if v is None:
                clean[k] = None
            elif isinstance(v, (int,)) and not isinstance(v, bool):
                clean[k] = int(v)
            elif isinstance(v, float):
                clean[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
            else:
                clean[k] = None
        except Exception:
            clean[k] = None
    return clean

# === AJOUT DES COLONNES MANQUANTES ===
def ensure_columns_exist():
    """
    Initialise la base résultats et applique une migration idempotente :
      - crée les tables manquantes (logs, trades, kpi_by_backtest),
      - ajoute les nouvelles colonnes si nécessaire (equity/pnl_step/drawdown/fees...).

    Remplace l'ancien ajout manuel de colonnes dans `trades`.
    """
    init_sqlite_database(DB_PATH)
    migrate_sqlite_schema(DB_PATH)

# === CALCULS DES COLONNES MANQUANTES ===
def update_performance_metrics():
    """
    Recalcule les KPI standardisés pour chaque `backtest_id` à partir des tables
    `trades` (obligatoire) et `logs` (optionnelle), puis fait un UPSERT dans
    `kpi_by_backtest` via `upsert_kpis`.

    Détails:
    - Fallback timestamp: utilise 'timestamp' si présent, sinon 'exit_time'.
    - Iteration: prend la valeur la plus fréquente (mode) pour le backtest_id, sinon max().
    - Sanitisation: NaN/Inf → None avant UPSERT (compat SQLite).
    """
    print("📊 Recalcul des KPI standardisés (kpi_by_backtest)...")
    with sqlite3.connect(DB_PATH) as conn:
        trades = pd.read_sql_query("SELECT * FROM trades", conn)
        try:
            logs = pd.read_sql_query("SELECT * FROM logs", conn)
        except Exception:
            logs = pd.DataFrame()

    if trades.empty:
        print("⚠️ Table 'trades' vide — rien à recalculer.")
        return

    # Partition par backtest_id
    for bid, grp in trades.groupby("backtest_id"):
        df_tr = grp.copy()

        # Choix du timestamp de référence
        ts_col = "timestamp" if "timestamp" in df_tr.columns else ("exit_time" if "exit_time" in df_tr.columns else None)
        if ts_col is not None:
            df_tr[ts_col] = pd.to_datetime(df_tr[ts_col], errors="coerce", utc=True)
            df_tr = df_tr.sort_values(ts_col)

        # Sous-ensemble des logs correspondant (si disponible)
        if not logs.empty and "backtest_id" in logs.columns:
            df_lg = logs[logs["backtest_id"] == bid].copy()
            if not df_lg.empty:
                lg_ts = "timestamp" if "timestamp" in df_lg.columns else None
                if lg_ts is not None:
                    df_lg[lg_ts] = pd.to_datetime(df_lg[lg_ts], errors="coerce", utc=True)
                    df_lg = df_lg.sort_values(lg_ts)
            else:
                df_lg = None
        else:
            df_lg = None

        # Calcul KPI (tolère colonnes manquantes → NaN ciblés)
        try:
            kpi_raw = mx.compute_intraday_consistency_kpis(
                df_trades=df_tr,
                df_logs=df_lg,
                price_col="price",
                timestamp_col=ts_col or "timestamp",
                fee_rate=0.0
            )
            kpi = _sanitize_kpi(kpi_raw)
        except Exception as e:
            print(f"[WARN] KPI échoués pour {bid}: {e}")
            continue

        # Iteration robuste (mode; fallback = max; défaut = 0)
        iter_val = 0
        if "iteration" in df_tr.columns:
            try:
                iter_mode = df_tr["iteration"].mode(dropna=True)
                iter_val = int(iter_mode.iloc[0]) if not iter_mode.empty else int(df_tr["iteration"].max())
            except Exception:
                try:
                    iter_val = int(df_tr["iteration"].iloc[0])
                except Exception:
                    iter_val = 0

        # UPSERT en base
        try:
            upsert_kpis(DB_PATH, backtest_id=str(bid), iteration=iter_val, kpi=kpi)
        except Exception as e:
            print(f"[WARN] UPSERT KPI échoué pour {bid}: {e}")

    print("✅ Recalcul KPI terminé.")

def compute_and_propagate_metrics():
    """
    Wrapper conservé pour compatibilité : déclenche simplement le recalcul KPI
    standardisé (kpi_by_backtest). Plus aucune propagation colonne-à-colonne dans
    `trades` (obsolète).
    """
    update_performance_metrics()

# === TEST AUTOML AVEC GP (BoTorch) ===
def run_tests_with_gp():
    """
    Lance quelques backtests proposés par BoTorch et affiche le backtest_id + KPI clés.
    Compatible avec la signature (success, backtest_id, kpi).
    """
    print(f"📡 Test: génération de {N_BOTORCH_TRIALS} points via BoTorch...")
    suggested_params_list = suggest_from_gp(n_trials=N_BOTORCH_TRIALS)

    for i, param_dict in enumerate(suggested_params_list):
        print(f"\n🚀 Test BoTorch Backtest #{i + 1}")
        success, backtest_id, kpi = run_backtest_from_params(param_dict)
        if success:
            print(f"✅ Test #{i + 1} réussi | backtest_id={backtest_id} | sharpe_d_252={kpi.get('sharpe_d_252')}")
        else:
            print(f"❌ Test #{i + 1} échoué | backtest_id={backtest_id}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🔧 Init + migration du schéma SQLite...")
    ensure_columns_exist()  # init + migrate (tables + colonnes)

    print("🧮 Recalcul des KPI standardisés (kpi_by_backtest)...")
    compute_and_propagate_metrics()

    if USE_BOTORCH:
        run_tests_with_gp()
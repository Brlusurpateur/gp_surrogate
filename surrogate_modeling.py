"""
====================================================================================
Fichier : surrogate_modeling.py
Objectif : Modélisation interprétable de la performance stratégique via surrogate model
====================================================================================

Ce module entraîne un modèle substitut (XGBoost) pour approximer une métrique cible
(Sharpe ratio) à partir de l'historique des backtests (SQLite), puis produit :

  • SHAP (beeswarm & bar) + export JSON des importances globales mean(|SHAP|)
  • UMAP sur les valeurs SHAP (structure non linéaire)
  
Les importances exportées (JSON) sont destinées au driver GP pour :
  – construire une Trust-Region anisotrope,
  – pondérer les distances de diversité intra-batch.

Auteur : Moncoucut Brandon
Version : Octobre 2025
"""

# === Imports fondamentaux ===
import os
import sys
import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap
# --- Garde-fou pour la cible (cap du Sharpe annualisé) ---
try:
    import config as cfg
    S_CAP = float(getattr(cfg, "SHARPE_CAP", 10.0)) #performance 5 !!!!!!!!!!
except Exception:
    S_CAP = 10.0  # défaut robuste si config indisponible #performance 5 !!!!!!!!

# === CONFIG ===
# Cible par défaut : Sharpe annualisé daily standardisé (table kpi_by_backtest).
# Fallback automatique sur trades.sharpe_ratio si la table KPI n'existe pas.
TARGET_METRIC = "sharpe_d_365"
EXPORT_DIR = "/Users/brandonmoncoucut/Desktop/Najas_king/Charts/surrogate_modeling"
ARTIFACTS_DIR = "/Users/brandonmoncoucut/Desktop/Najas_king/Artifacts"
MAX_ROWS = 5000  # plafond de lignes chargées (sécurité mémoire)

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# I/O utils
# -----------------------------------------------------------------------------
def _save_plot(name: str) -> None:
    """Sauvegarde la figure matplotlib active avec un nom daté (robuste)."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    filename = f"{name}_{timestamp}.png"
    path = os.path.join(EXPORT_DIR, filename)
    try:
        plt.savefig(path, bbox_inches="tight")
        print(f"✅ Figure sauvegardée : {path}")
    except Exception as e:
        print(f"⚠️ Sauvegarde figure échouée ({path}) : {e}")
    finally:
        plt.close()

def save_shap_importance_json(path: str, feature_names, shap_values: np.ndarray) -> None:
    """
    Sauvegarde un dict {feature: mean(|SHAP|)} en JSON pour usage GP.
    """
    try:
        mean_abs = np.mean(np.abs(shap_values), axis=0)  # shape [d]
        data = {str(name): float(val) for name, val in zip(feature_names, mean_abs)}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"✅ SHAP importance JSON écrit : {path}")
    except Exception as e:
        print(f"⚠️ Échec écriture SHAP JSON ({path}) : {e}")

def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    """
    Retourne True si la table SQLite `name` existe dans la base.
    """
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
        return cur.fetchone() is not None
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_data(db_path: str, limit: int = 1000) -> pd.DataFrame:
    """
    Récupère une ligne par backtest_id (meilleure performance) + dernier log (hyperparams).
    Priorité aux KPI standardisés (table `kpi_by_backtest`, champ TARGET_METRIC).
    Fallback automatique : `trades.sharpe_ratio` si la table KPI n'existe pas.

    Returns
    -------
    pd.DataFrame
        Colonnes : backtest_id, timestamp (UTC), hyperparams..., TARGET_METRIC
    """
    limit = int(min(limit, MAX_ROWS))
    hyperparams = [
        "timestamp", "ema_short_period", "ema_long_period", "rsi_period", "rsi_buy_zone", "rsi_sell_zone",
        "rsi_past_lookback", "atr_tp_multiplier", "atr_sl_multiplier", "atr_period",
        "macd_signal_period", "rsi_thresholds_1m", "rsi_thresholds_5m", "rsi_thresholds_15m",
        "rsi_thresholds_1h", "ewma_period", "weight_atr_combined_vol", "threshold_volume",
        "hist_volum_period", "detect_supp_resist_period", "trend_period", "threshold_factor",
        "min_profit_margin", "resistance_buffer_margin", "risk_reward_ratio", "confidence_score_params",
        "signal_weight_bonus", "penalite_resistance_factor", "penalite_multi_tf_step",
        "override_score_threshold", "rsi_extreme_threshold", "signal_pure_threshold",
        "signal_pure_weight"
    ]
    col_str = ", ".join([f"l.{c}" for c in ["backtest_id"] + hyperparams])

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='kpi_by_backtest';")
        has_kpi = cur.fetchone() is not None
        # Info migration : si des données 252 existent encore, on le signale pour éviter l'ambiguïté.
        if has_kpi and TARGET_METRIC == "sharpe_d_365":
            try:
                cur.execute("SELECT 1 FROM kpi_by_backtest WHERE sharpe_d_252 IS NOT NULL LIMIT 1;")
                if cur.fetchone() is not None:
                    print("ℹ️  kpi_by_backtest contient des valeurs 'sharpe_d_252' alors que TARGET_METRIC='sharpe_d_365'. "
                        "Vérifie la migration/annualisation des run historiques.")
            except Exception:
                # Tolérant : pas bloquant si la colonne n'existe pas
                pass

        if has_kpi:
            # --- Chemin moderne : kpi_by_backtest + best TARGET_METRIC par backtest_id
            # Détecte la présence des colonnes de validation/outliers
            try:
                cur.execute("PRAGMA table_info(kpi_by_backtest);")
                _cols = [r[1] for r in cur.fetchall()]
            except Exception:
                _cols = []
            has_is_valid = ("is_valid" in _cols)
            has_flag_out = ("flag_sharpe_outlier" in _cols)

            # Clause de filtre SQL si colonnes présentes
            filter_clause = ""
            if has_is_valid or has_flag_out:
                # coalesce pour gérer valeurs NULL comme « OK sauf si explicite »
                _f1 = "AND COALESCE(is_valid,1)=1" if has_is_valid else ""
                _f2 = "AND COALESCE(flag_sharpe_outlier,0)=0" if has_flag_out else ""
                filter_clause = f"{_f1} {_f2}"

                # Stats d’exclusion (avant/après)
                try:
                    cur.execute(f"SELECT COUNT(DISTINCT backtest_id) FROM kpi_by_backtest WHERE {TARGET_METRIC} IS NOT NULL;")
                    tot = int(cur.fetchone()[0] or 0)
                    cur.execute(f"SELECT COUNT(DISTINCT backtest_id) FROM kpi_by_backtest WHERE {TARGET_METRIC} IS NOT NULL {_f1} {_f2};")
                    ok = int(cur.fetchone()[0] or 0)
                    if tot > 0 and ok <= tot:
                        drop_pct = 100.0 * (tot - ok) / tot
                        print(f"🧹 Filtre KPI: is_valid/!outlier — gardés={ok}/{tot} ({100.0 - drop_pct:.1f}% kept, {drop_pct:.1f}% dropped)")
                except Exception:
                    pass

            query = f"""
                WITH ranked AS (
                    SELECT
                        backtest_id,
                        iteration,
                        {TARGET_METRIC} AS metric,
                        ROW_NUMBER() OVER (
                            PARTITION BY backtest_id
                            ORDER BY {TARGET_METRIC} DESC
                        ) AS rn
                    FROM kpi_by_backtest
                    WHERE {TARGET_METRIC} IS NOT NULL
                    {filter_clause}
                ),
                top AS (
                    SELECT backtest_id, iteration, metric
                    FROM ranked
                    WHERE rn = 1
                    ORDER BY metric DESC
                    LIMIT {limit}
                ),
                mx AS (
                    SELECT l.backtest_id, MAX(l.timestamp) AS ts
                    FROM logs l
                    JOIN top t ON l.backtest_id = t.backtest_id
                    GROUP BY l.backtest_id
                )
                SELECT {col_str}, t.metric AS {TARGET_METRIC}, t.iteration
                FROM logs l
                JOIN mx  ON l.backtest_id = mx.backtest_id AND l.timestamp = mx.ts
                JOIN top t ON l.backtest_id = t.backtest_id
            """
            df = pd.read_sql_query(query, conn)
            # Si la table KPI existe mais ne retourne rien (toutes valeurs NULL), on fera fallback ci-dessous.
            if df.empty:
                has_kpi = False

        if not has_kpi:
            # --- Fallback hérité : best trades.sharpe_ratio (par backtest_id + iteration)
            query = f"""
                WITH ranked AS (
                    SELECT
                        backtest_id,
                        iteration,
                        sharpe_ratio AS metric,
                        ROW_NUMBER() OVER (
                            PARTITION BY backtest_id
                            ORDER BY sharpe_ratio DESC
                        ) AS rn
                    FROM trades
                    WHERE sharpe_ratio IS NOT NULL
                ),
                top AS (
                    SELECT backtest_id, iteration, metric
                    FROM ranked
                    WHERE rn = 1
                    ORDER BY metric DESC
                    LIMIT {limit}
                ),
                mx AS (
                    SELECT l.backtest_id, MAX(l.timestamp) AS ts
                    FROM logs l
                    JOIN top t ON l.backtest_id = t.backtest_id
                    GROUP BY l.backtest_id
                )
                SELECT {col_str}, t.metric AS sharpe_ratio, t.iteration
                FROM logs l
                JOIN mx  ON l.backtest_id = mx.backtest_id AND l.timestamp = mx.ts
                JOIN top t ON l.backtest_id = t.backtest_id
            """
            df = pd.read_sql_query(query, conn)
    finally:
        if conn is not None:
            conn.close()

    # Harmonisation UTC
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Harmonisation de la colonne cible (assure TARGET_METRIC dans tous les cas)
    if TARGET_METRIC not in df.columns:
        # Fallback hérité : mappe depuis trades.sharpe_ratio si présent.
        # ⚠️ Attention : l'échelle peut différer de l'annualisation 365 ; c'est un repli, pas la voie nominale.
        if "sharpe_ratio" in df.columns:
            print(f"⚠️  Fallback: utilisation de 'trades.sharpe_ratio' pour alimenter '{TARGET_METRIC}'. "
                "Vérifie l'unité/annualisation.")
            df[TARGET_METRIC] = df["sharpe_ratio"]
        else:
            raise ValueError(f"❌ La colonne cible '{TARGET_METRIC}' est absente des résultats.")

    # Nettoyage numérique (évite NaN/Inf côté modeling)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    return df

# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

def print_dataset_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    y_name: str = TARGET_METRIC,
    alert_std_eps: float = 1e-8,
    alert_unique_k: int = 3,
) -> None:
    """
    Affiche des diagnostics "qualité signal" pour (X, y).

    Paramètres
    ----------
    X : pd.DataFrame
        Matrice d'hyperparamètres (features numériques).
    y : pd.Series
        Cible (par défaut, Sharpe annualisé `TARGET_METRIC`).
    y_name : str
        Nom lisible de la cible pour les logs.
    alert_std_eps : float
        Seuil sous lequel l'écart-type est considéré comme quasi nul.
    alert_unique_k : int
        Seuil sous lequel le nombre de valeurs uniques de y est jugé trop faible.

    Logs produits
    -------------
    • Taille et taux de NaN de X et y
    • Nombre de valeurs uniques de y + stats (moyenne / écart-type / quantiles)
    • Variance par feature (triée, pour repérer les colonnes quasi constantes)
    • Alertes si std(y) ~ 0 ou si y a très peu de valeurs uniques
    """
    # --- Taille / NaN ---
    n, d = X.shape
    x_nan_ratio = X.isna().mean().mean() if n * d > 0 else float("nan")
    y_nan_ratio = float(y.isna().mean()) if len(y) > 0 else float("nan")
    print(f"🧪 DIAG: X shape={X.shape}, NaN_rate≈{x_nan_ratio:.4f} | y len={len(y)}, NaN_rate≈{y_nan_ratio:.4f}")

    # --- y : uniques + stats ---
    y_clean = y.replace([np.inf, -np.inf], np.nan).dropna()
    nunique_y = int(y_clean.nunique())
    y_mean = float(y_clean.mean()) if len(y_clean) else float("nan")
    y_std  = float(y_clean.std(ddof=1)) if len(y_clean) > 1 else 0.0
    q = y_clean.quantile([0.05, 0.25, 0.50, 0.75, 0.95]) if len(y_clean) else pd.Series(dtype=float)

    print(f"🧪 DIAG: y='{y_name}' | uniques={nunique_y} | mean={y_mean:.6f} | std={y_std:.6f}")
    if not q.empty:
        print("🧪 DIAG: y quantiles (5/25/50/75/95%): " +
              ", ".join([f"{int(p*100)}%={q.loc[p]:.6f}" for p in [0.05, 0.25, 0.50, 0.75, 0.95]]))

    # --- Variance par feature (pour colonnes quasi constantes) ---
    # On remplit temporairement les NaN par la médiane pour ne pas biaiser la variance.
    X_num = X.copy()
    for c in X_num.columns:
        if X_num[c].isna().any():
            med = X_num[c].median(skipna=True)
            X_num[c] = X_num[c].fillna(med)
    variances = X_num.var(ddof=0).sort_values()  # ddof=0 -> variance population
    low_var = variances.head(min(10, len(variances)))
    print("🧪 DIAG: plus faibles variances (top 10) →")
    for feat, v in low_var.items():
        print(f"   • {feat}: var={v:.6e}")

    # --- Alertes ---
    if y_std <= alert_std_eps:
        print(f"🚨 ALERTE: std(y)≈0 (std={y_std:.3e}) → cible quasi constante / mal construite.")
    if nunique_y < alert_unique_k:
        print(f"🚨 ALERTE: y a très peu de valeurs uniques (nunique={nunique_y}) → cible quasi constante / mal construite.")

def train_xgb(df: pd.DataFrame):
    """
    Entraîne un XGBoost régularisé pour approximer TARGET_METRIC à partir des hyperparams.
    Retourne : (model, X, y)
    """
    exclude = ["timestamp", "backtest_id", "iteration", TARGET_METRIC]
    param_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    X = df[param_cols].replace([np.inf, -np.inf], np.nan).dropna()
    y_raw = df.loc[X.index, TARGET_METRIC].astype(float)

    # Cap de sécurité (anti-valeurs aberrantes)
    y = np.clip(y_raw, -S_CAP, S_CAP)
    try:
        clip_ratio = float((np.abs(y_raw.values) > S_CAP).mean())
        if clip_ratio > 0.0:
            print(f"🛡️  Target clip: |y|>S_CAP sur {clip_ratio*100:.1f}% des points (S_CAP={S_CAP})")
    except Exception:
        pass

    # Modèle un peu régularisé + seed fixe (stabilité et généralisation)
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=0
    )
    model.fit(X, y)
    return model, X, y

# -----------------------------------------------------------------------------
# SHAP & viz
# -----------------------------------------------------------------------------
def compute_shap(model, X: pd.DataFrame):
    """
    Calcule les valeurs SHAP (numpy) + tracés globaux.
    Retourne (shap_values: np.ndarray).
    """
    # Explainer & values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # np.ndarray [n, d]

    # Expected value compat (scalaire ou vecteur)
    try:
        base_vals = explainer.expected_value
        if isinstance(base_vals, (list, tuple, np.ndarray)):
            base_vals = np.array(base_vals).ravel()
            base_vals = float(np.mean(base_vals))
    except Exception:
        base_vals = 0.0

    # Beeswarm
    try:
        shap_obj = shap.Explanation(
            values=shap_values,
            base_values=np.full((shap_values.shape[0],), base_vals, dtype=float),
            data=X.values,
            feature_names=X.columns
        )
        shap.plots.beeswarm(shap_obj, max_display=20, show=False)
        plt.title(f"SHAP beeswarm — {TARGET_METRIC} (capé modèle ±{S_CAP})")
        _save_plot(f"shap_beeswarm_{TARGET_METRIC}")

    except Exception as e:
        print(f"⚠️ Beeswarm non généré : {e}")

    # Bar
    try:
        shap.plots.bar(shap_obj, max_display=20, show=False)
        plt.title(f"SHAP importance — {TARGET_METRIC} (capé modèle ±{S_CAP})")
        _save_plot(f"shap_bar_{TARGET_METRIC}")

    except Exception as e:
        print(f"⚠️ Bar plot non généré : {e}")

    return shap_values

def save_pdp_2d(*args, **kwargs):
    """
    PDP désactivé : no-op (gardé pour compat ascendante).
    """
    return

def umap_on_shap(shap_values: np.ndarray, y: pd.Series):
    """
    UMAP des SHAP values (après standardisation) coloré par y (Sharpe).
    """
    try:
        scaler = StandardScaler()
        shap_scaled = scaler.fit_transform(shap_values)

        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(shap_scaled)

        df_emb = pd.DataFrame(emb, columns=["UMAP1", "UMAP2"])
        df_emb["Sharpe"] = y.values

        sns.scatterplot(data=df_emb, x="UMAP1", y="UMAP2", hue="Sharpe", palette="coolwarm")
        plt.title(f"UMAP des SHAP — {TARGET_METRIC} (capé modèle ±{S_CAP})")
        _save_plot(f"shap_umap_{TARGET_METRIC}")

    except Exception as e:
        print(f"⚠️ UMAP non généré : {e}")

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def run_surrogate_pipeline(db_path: str, top_k: int = 1000):
    """
    1) Chargement (top_k backtests uniques, meilleurs Sharpe / id)
    2) Entraînement XGB (SHARPE ~ hyperparams)
    3) SHAP + export JSON (Artifacts/shap_importance.json)
    4) UMAP sur SHAP
    """
    print("📥 Chargement...")
    print(f"🎯 Target = {TARGET_METRIC} (capé modèle ±{S_CAP}) — filtres data: is_valid=1 & !outlier (pas de filtre cap).")
    df = load_data(db_path, limit=top_k)

    print("🧠 Entraînement modèle XGBoost...")
    model, X, y = train_xgb(df)

    # --- Export de la liste des features (pour le GP) ---
    try:
        feature_list = list(map(str, X.columns))
        feats_path = os.path.join(ARTIFACTS_DIR, "surrogate_features.json")
        with open(feats_path, "w") as f:
            json.dump({"feature_list": feature_list}, f)
        print(f"✅ Feature list exportée : {feats_path} ({len(feature_list)} features)")
    except Exception as e:
        print(f"⚠️ Impossible d'écrire surrogate_features.json : {e}")

    # 🧪 Diagnostics qualité signal (X/y)
    print("🧪 Diagnostics X/y (qualité signal)...")
    print_dataset_diagnostics(X, y, y_name=TARGET_METRIC)

    # Vérifier que y n’est pas dégénérée (trop peu de valeurs / variance quasi nulle)
    y_clean = pd.Series(y).replace([np.inf, -np.inf], np.nan).dropna()
    y_std = float(y_clean.std(ddof=1)) if len(y_clean) > 1 else 0.0
    y_nuniq = int(y_clean.nunique())

    if (y_std <= 1e-12) or (y_nuniq < 3):
        print(f"🚫 SHAP sauté: y dégénérée (std={y_std:.3e}, nunique={y_nuniq}).")
        shap_values = None
    else:
        print("🔍 Analyse SHAP...")
        shap_values = compute_shap(model, X)

        # Export importances globales (consommées par gp_driver.py)
        shap_json_path = os.path.join(ARTIFACTS_DIR, "shap_importance.json")
        save_shap_importance_json(shap_json_path, X.columns, shap_values)

        print("🌌 UMAP sur SHAP values...")
        umap_on_shap(shap_values, y)

    print("\n🎯 Analyse terminée. Résultats dans :", EXPORT_DIR)

# -----------------------------------------------------------------------------
# Entrée CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❌ Usage: python surrogate_modeling.py <db_path>")
        sys.exit(1)
    DB_PATH = sys.argv[1]
    run_surrogate_pipeline(DB_PATH)

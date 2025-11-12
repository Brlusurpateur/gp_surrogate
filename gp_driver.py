"""
====================================================================================
Fichier : gp_driver.py
Objectif : Modélisation bayésienne de la performance stratégique par processus gaussien
====================================================================================

Description générale :
Ce module implémente une modélisation probabiliste de la performance d’une stratégie de
trading en fonction de ses hyperparamètres, à l’aide d’un modèle de régression gaussienne
(Gaussian Process Regression) via BoTorch, une librairie construite sur PyTorch.

Cette approche permet d’exploiter un historique de backtests pour approximer la fonction
coût (le Sharpe ratio) par un modèle substitut probabiliste — dit modèle de surrogate.
Elle sert de base à une optimisation bayésienne plus économe en ressources que l’Optuna naïf.

Étapes principales :
    1. Chargement des logs : jointure des paramètres (logs) et scores (trades) depuis SQLite.
    2. Préparation des données : sélection, normalisation, conversion en tenseurs.
    3. Entraînement d’un modèle GP : ajustement des hyperparamètres du modèle par maximum de vraisemblance marginale.
    4. Définition d’une fonction d’acquisition (LogExpectedImprovement).
    5. Génération de nouvelles configurations d’hyperparamètres jugées prometteuses.

Motivation quantitative :
Le processus gaussien fournit une distribution a posteriori sur la fonction d’évaluation (Sharpe ratio),
permettant à chaque itération de cibler des zones à fort potentiel en tenant compte de
l’incertitude. Cette stratégie d’exploration-exploitation est standard en recherche opérationnelle
et chez les asset managers institutionnels modernes.

Technologies :
    - `botorch`, `gpytorch` : GP bayésiens modernes avec gradients automatiques
    - `pandas` + `sqlite3` : pour accéder aux résultats des backtests
    - `torch` : manipulation efficace de tenseurs numériques

Sortie attendue :
    - `List[Dict[str, float]]` : suggestions de nouveaux points dans l’espace des hyperparamètres
      (à passer ensuite à `backtest_interface_code.py` pour évaluation réelle)

Bonnes pratiques intégrées :
    - Chargement filtré des hyperparamètres via une fonction centralisée load_data
    - Normalisation stricte en [0,1] de l’espace d’entrée
    - Stabilisation des calculs via float64 et LogExpectedImprovement

Auteur : Moncoucut Brandon
Version : Juin 2025
"""

# === Imports fondamentaux ===
import torch                          # Tensor computation, indispensable pour BoTorch
from botorch.models import SingleTaskGP        # Modèle de régression Gaussienne simple
from botorch.models.transforms import Normalize
from botorch.models.transforms.outcome import Standardize as _Standardize
from botorch.fit import fit_gpytorch_mll       # Fonction de fitting du GP
# ---------------------------------------------------------------------------
# Acquisition function (BoTorch) — Compatibilité de versions :
# - Si qLogExpectedImprovement est disponible (>= certaines versions BoTorch),
#   on l'utilise (meilleure stabilité numérique).
# - Sinon, on retombe proprement sur qExpectedImprovement.
# On référence ACQ_CLASS partout pour éviter tout mismatch futur.
# ---------------------------------------------------------------------------
try:  # BoTorch récent (recommandé)
    from botorch.acquisition.monte_carlo import qLogExpectedImprovement as _QLOGEI
    ACQ_CLASS = _QLOGEI
except Exception:
    # Fallback BoTorch stable (versions plus anciennes)
    from botorch.acquisition.monte_carlo import qExpectedImprovement as _QEI
    ACQ_CLASS = _QEI
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf        # Optimiseur d’acquisition BoTorch
from gpytorch.mlls import ExactMarginalLogLikelihood  # Fonction de vraisemblance marginale
import numpy as np
import pandas as pd
import sqlite3
import sys
import json
import gpytorch
import os, json
# ── GP model with SafeStandardize & noise floor ──
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit as _botorch_fit

# Garde-fous config (cap Sharpe)
try:
    import config as _cfg
    SHARPE_CAP: float = float(getattr(_cfg, "SHARPE_CAP", 5.0))
except Exception:
    SHARPE_CAP: float = 5.0

from hyperparam_domain import DOMAIN_BOUNDS
from gp_driver_utils import (
        restrict_bounds_around,          # boîte anisotrope centrée
        restrict_bounds_feature_aware,   # fallback si besoin
        make_distance_weights,           # pondère les distances
        far_enough,                      # accepte weights=
        greedy_poisson_selection,        # accepte weights=, k_max=
        enforce_constraints_for_gp,
        scale_TR_per_dimension,          # calcule les échelles par dimension
        apply_soft_box_prior,            # soft prior ciblé ATR (tp/sl)
        gen_multi_start_inits,           # init multi-start robuste (acqf)
        compute_rank_score_from_df,      # <<< NEW : score multi-objectif par rangs
        _safe_std,
        _safe_corr,
        _winsorize_series,
        _adaptive_log_wfcv
    )

try:
    from config import score_weights as _score_weights
except Exception:
    def _score_weights(normalize: bool = True) -> dict:
        # Fallback raisonnable si config indisponible
        w = dict(SHARPE=0.35, PF=0.20, GREEN=0.10, MDD=0.20, ULCER=0.10, TOP5=0.05)
        s = sum(w.values())
        return {k: v / s for k, v in w.items()} if normalize and s > 0 else w

class SafeStandardize(_Standardize):
    """
    Variante tolérante : si std(y) ~ 0 sur le fold, n'applique pas la standardisation.
    Empêche les NaN quand la cible est (quasi) constante.
    """
    def __init__(self, m=1, eps: float = 1e-8):
        super().__init__(m=m, batch_shape=None)
        self._eps = float(eps)

    def forward(self, Y):
        import torch
        Yf = Y if isinstance(Y, torch.Tensor) else torch.as_tensor(Y)
        std = Yf.std()
        if torch.isnan(std) or (std.abs() < self._eps):
            # Pas de standardisation : identité
            return Y
        return super().forward(Y)

    def untransform(self, Y):
        # fallback identique si pas de std
        return super().untransform(Y)
    
# =============================================================================
# PARAMÈTRES / CONSTANTES (réglables au même endroit)
# =============================================================================

# Données / chargement
DB_TOP_LIMIT: int = 1000                # nb max de backtests chargés
TEST_RATIO_DEFAULT: float = 0.20        # part test dans le split temporel
WFCV_N_FOLDS: int = 3                   # nb de folds pour la WFCV
WFCV_MIN_PER_SPLIT: int = 20            # min d’observations par split

# Artifacts (SHAP -> GP)
SHAP_ARTIFACT_DIR: str = "/Users/brandonmoncoucut/Desktop/Najas_king/Artifacts"
SHAP_JSON_NAME: str = "shap_importance.json"  # laissé modulable

# Modèle / acquisition
TORCH_DEFAULT_DTYPE = torch.double      # stabilité num.
MC_SAMPLE_SHAPE: int = 256              # échantillons MC pour qEI

# q-batch & Trust-Region
Q_BATCH: int = 8
LOCAL_SHARE: float = 0.70
# Rayon de base de la TR en [0,1].
# RATIONNEL:
# - On élargit légèrement la TR par défaut (0.15 → 0.22) pour que le volume exploré
#   soit plus conséquent autour des centres, ce qui augmente le nombre de candidats
#   “pre” puis “after_mask”.
TR_RADIUS_DEFAULT: float = 0.22
MIN_L2_DEFAULT: float = 0.02            # distance minimale en L2
TOPK_CENTERS_DEFAULT: int = 3           # nb de centres TR
MAX_RETRIES_DEFAULT: int = 5            # retries si batch vide
# Multiplicateurs du rayon de base par rang de centre (du + étroit au + large).
# RATIONNEL:
# - On “pousse” un peu le rayon des centres 2/3 (1.00→1.25, 1.25→1.50) pour
#   garantir >2–3 candidats avant filtrage dans tes stats [DBG][CANDS].
CENTER_RADIUS_SCHEDULE = (1.00, 1.25, 1.50)

# Diversité anisotrope (distance pondérée)
DIVERSITY_WEIGHT_FLOOR: float = 0.50    # plancher de tolérance dims peu importantes
DIVERSITY_WEIGHT_CEIL: float  = 2.00    # plafond (plus strict) dims importantes
DISTANCE_THR_OVERRIDE: float | None = None  # impose un seuil (sinon auto)

# Répartition local/global si on veut forcer (sinon auto via LOCAL_SHARE)
Q_LOCAL_OVERRIDE: int | None = None
Q_GLOBAL_OVERRIDE: int | None = None

# Optimisation (multi-start BoTorch)
ACQ_LOCAL_NUM_RESTARTS: int = 16
ACQ_LOCAL_RAW_SAMPLES: int  = 768
ACQ_GLOBAL_NUM_RESTARTS: int = 20
ACQ_GLOBAL_RAW_SAMPLES: int  = 1024
ACQ_MAXITER: int = 300

# Re-ranking soft prior (stabilité mid-frequency intraday)
RERANK_ALPHA: float = 0.70              # poids mean(GP)
RERANK_BETA:  float = 0.30              # poids soft prior
SOFT_PRIOR_CENTER: float = 0.50         # centre préféré en [0,1]

# Cible de modélisation (KPI crypto annualisé 365)
# Les centres TR (top-K) seront donc déterminés sur 'sharpe_d_365'.
TARGET = "sharpe_d_365"

# --- Anisotropie & soft-box prior spécifiques ---
ANISO_MIN_SCALE: float = 0.35         # borne basse d’échelle TR par dimension
ANISO_MAX_SCALE: float = 1.65         # borne haute d’échelle TR par dimension

# Prior doux pour ATR (centrage autour de 0.5 en espace normalisé)
BOX_PRIOR_FEATURES: tuple[str, ...] = ("atr_tp_multiplier", "atr_sl_multiplier")
BOX_PRIOR_CENTER: float = 0.50        # centre en [0,1]
BOX_PRIOR_GAMMA: float = 0.35         # intensité (0.2–0.5 raisonnable)

# Génération d'initialisations multi-start (si dispo dans utils / version BoTorch)
USE_GEN_MULTI_START_INITS: bool = True

ARTIFACTS_DIR = "/Users/brandonmoncoucut/Desktop/Najas_king/Artifacts"  # adapte si besoin
SURR_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "surrogate_features.json")

# Applique le dtype par défaut strictement
torch.set_default_dtype(TORCH_DEFAULT_DTYPE)


# =============================================================================
# FEATURE IMPORTANCE pour distances & Trust-Region
# =============================================================================

def _load_surrogate_feature_list() -> list:
    try:
        with open(SURR_FEATURES_PATH, "r") as f:
            payload = json.load(f)
        feats = list(map(str, payload.get("feature_list", [])))
        return feats
    except Exception:
        return []
    
def _load_shap_importance_json(json_path: str, feature_names: list[str]) -> dict | None:
    """
    Charge un JSON {feature: importance} et renvoie un dict normalisé sur les features présentes.
    Retourne None si indisponible.
    """
    try:
        with open(json_path, "r") as f:
            raw = json.load(f)
        v = np.array([float(raw.get(k, 0.0)) for k in feature_names], dtype=float)
        s = float(v.sum())
        if s <= 0:
            return None
        v = v / s
        return {k: float(v[i]) for i, k in enumerate(feature_names)}
    except Exception:
        return None


def _gp_ard_importance(model: SingleTaskGP, feature_names: list[str]) -> dict | None:
    """
    Fallback robuste : importance ~ 1/lengthscale^2 si ARD dispo.
    Retourne None si le kernel ne fournit pas de lengthscales par dimension.
    """
    try:
        # SingleTaskGP -> ScaleKernel(base_kernel=RBF/Matern avec ard_num_dims=d)
        ls = model.covar_module.base_kernel.lengthscale  # [1, 1, d] ou [*, d]
        ls = ls.detach().cpu().view(-1).numpy()
        if ls.size != len(feature_names):
            return None
        inv = 1.0 / (ls * ls + 1e-12)
        s = inv.sum()
        if s <= 0:
            return None
        inv /= s
        return {k: float(inv[i]) for i, k in enumerate(feature_names)}
    except Exception:
        return None


def _get_feature_importance_for_gp(model: SingleTaskGP, feature_names: list[str]) -> dict | None:
    """
    1) Essaye SHAP (Artifacts/shap_importance.json)
    2) Sinon fallback ARD
    3) Sinon None => isotrope
    """
    shap_json = os.path.join(SHAP_ARTIFACT_DIR, SHAP_JSON_NAME)
    imp = _load_shap_importance_json(shap_json, feature_names)
    if imp is not None:
        print("[GP] Using SHAP importance for distances/TR.")
        return imp
    imp = _gp_ard_importance(model, feature_names)
    if imp is not None:
        print("[GP] Using ARD lengthscales for distances/TR (fallback).")
        return imp
    print("[GP] No feature importance available → isotropic distances/TR.")
    return None

# === PARAMÈTRES DE LA PIPELINE ===
TARGET = "sharpe_d_365"       # Variable cible pour la modélisation GP (Sharpe KPI 365)
torch.set_default_dtype(torch.double)   # Par défaut, calculs en float64

# =============================================================================
# UTILITAIRES : SCALER MIN-MAX
# =============================================================================
class MinMaxScalerDict:
    """
    Implémente un scaler MinMax adapté aux DataFrames pandas.
    - Fit : calcule min/max sur le jeu d'entraînement uniquement
    - Transform : applique ces bornes à d'autres DataFrames
    Permet de garder une cohérence train/test et d’éviter le data leakage.
    """

    def __init__(self, bounds=None):
        self.bounds = bounds or {}

    def fit(self, df, cols):
        self.bounds = {c: (float(df[c].min()), float(df[c].max())) for c in cols}
        # ⚠️ Ne PAS dropper les colonnes constantes : forcer une petite largeur (1.0)
        for c, (lo, hi) in list(self.bounds.items()):
            if not np.isfinite([lo, hi]).all():
                self.bounds[c] = (0.0, 1.0)
            elif hi <= lo:
                self.bounds[c] = (lo, lo + 1.0)
        return self

    def transform(self, df, cols):
        X = df[cols].copy()
        cols_ok = [c for c in cols if c in self.bounds]
        for c in cols_ok:
            lo, hi = self.bounds[c]
            denom = hi - lo
            if denom <= 0 or not np.isfinite(denom):
                denom = 1.0
            X[c] = (X[c] - lo) / (denom + 1e-12)
        return X[cols_ok], cols_ok

# =============================================================================
# DATA LOADER
# =============================================================================
def load_data(db_path, limit: int = DB_TOP_LIMIT):
    """
    Charge uniquement les meilleurs backtests selon le Sharpe ratio,
    avec leurs hyperparamètres depuis 'logs', sans surcharger la mémoire.

    Args:
        db_path (str): Chemin vers la base SQLite.
        limit (int): Nombre maximum de stratégies à charger.

    Returns:
        pd.DataFrame: Données prêtes pour la modélisation GP.
    """

    # Liste des colonnes d'hyperparamètres à récupérer
    hyperparams = [
        "ema_short_period", "ema_long_period", "rsi_period", "rsi_buy_zone", "rsi_sell_zone",
        "rsi_past_lookback", "atr_tp_multiplier", "atr_sl_multiplier", "atr_period",
        "macd_signal_period", "rsi_thresholds_1m", "rsi_thresholds_5m", "rsi_thresholds_15m",
        "rsi_thresholds_1h", "ewma_period", "weight_atr_combined_vol", "threshold_volume",
        "hist_volum_period", "detect_supp_resist_period", "trend_period", "threshold_factor",
        "min_profit_margin", "resistance_buffer_margin", "risk_reward_ratio", "confidence_score_params",
        "signal_weight_bonus", "penalite_resistance_factor", "penalite_multi_tf_step",
        "override_score_threshold", "rsi_extreme_threshold", "signal_pure_threshold",
        "signal_pure_weight"
    ]
    try:
        print("-> Création d'un index pour accélérer la recherche (si absent)...")
        conn = sqlite3.connect(db_path)
        # Index legacy (trades) pour compat
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_backtest_sharpe
            ON trades(backtest_id, sharpe_ratio);
        """)
        # Index KPI 365 si la table/colonne existent
        try:
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_kpi_backtest_sharpe365
                ON kpi_by_backtest(backtest_id, sharpe_d_365);
            """)
        except Exception:
            # Table/colonne pas encore migrée → pas bloquant
            pass
        conn.commit()

        # Détection de la dispo KPI 365
        cur = conn.cursor()
        def _table_exists(name: str) -> bool:
            try:
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
                return cur.fetchone() is not None
            except Exception:
                return False

        def _col_exists(table: str, col: str) -> bool:
            try:
                cur.execute(f"PRAGMA table_info({table});")
                cols = [row[1] for row in cur.fetchall()]
                return col in cols
            except Exception:
                return False

        has_kpi_365 = _table_exists("kpi_by_backtest") and _col_exists("kpi_by_backtest", "sharpe_d_365")

        print("-> Sélection des meilleurs Sharpe (priorité KPI 'sharpe_d_365', fallback trades)...")

        # 2️⃣ Charger uniquement la DERNIÈRE ligne de logs pour ces backtests (déterministe)
        col_str = ", ".join([f"l.{col}" for col in hyperparams])

        if has_kpi_365:
            # Détecte présence des colonnes de validation/outlier
            try:
                cur.execute("PRAGMA table_info(kpi_by_backtest);")
                _cols = [r[1] for r in cur.fetchall()]
            except Exception:
                _cols = []

            has_is_valid = ("is_valid" in _cols)
            has_flag_out = ("flag_sharpe_outlier" in _cols)

            # Clauses optionnelles
            _f1 = "AND COALESCE(is_valid,1)=1" if has_is_valid else ""
            _f2 = "AND COALESCE(flag_sharpe_outlier,0)=0" if has_flag_out else ""

            # Stats avant/après (sans cap SQL)
            try:
                cur.execute("SELECT COUNT(DISTINCT backtest_id) FROM kpi_by_backtest WHERE sharpe_d_365 IS NOT NULL;")
                _tot = int(cur.fetchone()[0] or 0)
                cur.execute(f"SELECT COUNT(DISTINCT backtest_id) FROM kpi_by_backtest WHERE sharpe_d_365 IS NOT NULL {_f1} {_f2};")
                _ok = int(cur.fetchone()[0] or 0)
                if _tot > 0:
                    _drop = 100.0 * (_tot - _ok) / _tot
                    print(f"🧹 Filtre KPI: valid & !outlier (cap = clipping modèle) {_ok}/{_tot} ({100.0 - _drop:.1f}% kept, {_drop:.1f}% dropped)")

            except Exception:
                pass

            # SQL: **AUCUNE** présélection top-Sharpe / pas d'ORDER BY/LIMIT ici.
            # On prend UN enregistrement KPI = meilleur Sharpe par backtest_id pour joindre aux logs (dernière ligne de logs).
            select_extra = []
            if has_is_valid:   select_extra.append("k.is_valid")
            if has_flag_out:   select_extra.append("k.flag_sharpe_outlier")
            extra_cols = (", " + ", ".join(select_extra)) if select_extra else ""

            query_logs = f"""
                WITH pool AS (
                    SELECT backtest_id, MAX(sharpe_d_365) AS best_sharpe365
                    FROM kpi_by_backtest
                    WHERE sharpe_d_365 IS NOT NULL
                    {_f1} {_f2}
                    GROUP BY backtest_id
                ),
                kbest AS (
                    SELECT k.backtest_id, k.iteration, k.sharpe_d_365{extra_cols}
                    FROM kpi_by_backtest k
                    JOIN pool p
                      ON k.backtest_id = p.backtest_id
                     AND k.sharpe_d_365 = p.best_sharpe365
                ),
                mx AS (
                    SELECT l.backtest_id, MAX(l.timestamp) AS ts
                    FROM logs l
                    JOIN pool p ON l.backtest_id = p.backtest_id
                    GROUP BY l.backtest_id
                )
                SELECT
                    l.backtest_id,
                    l.timestamp,
                    {col_str},
                    kbest.sharpe_d_365 AS sharpe_d_365
                    {", kbest.is_valid" if has_is_valid else ""}
                    {", kbest.flag_sharpe_outlier" if has_flag_out else ""}
                FROM logs l
                JOIN mx    ON l.backtest_id = mx.backtest_id AND l.timestamp = mx.ts
                JOIN kbest ON l.backtest_id = kbest.backtest_id
            """
            df = pd.read_sql_query(query_logs, conn)

        else:
            # Fallback legacy (avant migration KPI) : trades.sharpe_ratio
            query_top = f"""
                SELECT backtest_id, MAX(sharpe_ratio) AS best_sharpe
                FROM trades
                WHERE sharpe_ratio IS NOT NULL
                GROUP BY backtest_id
                ORDER BY best_sharpe DESC
                LIMIT {limit}
            """
            query_logs = f"""
                WITH top AS (
                    {query_top}
                ),
                mx AS (
                    SELECT l.backtest_id, MAX(l.timestamp) AS ts
                    FROM logs l
                    JOIN top t ON l.backtest_id = t.backtest_id
                    GROUP BY l.backtest_id
                )
                SELECT
                    l.backtest_id,
                    l.timestamp,
                    {col_str},
                    t.best_sharpe AS sharpe_ratio
                FROM logs l
                JOIN mx    ON l.backtest_id = mx.backtest_id AND l.timestamp = mx.ts
                JOIN top t ON l.backtest_id = t.backtest_id
            """
            df = pd.read_sql_query(query_logs, conn)
            if "sharpe_d_365" not in df.columns:
                print("⚠️  KPI 'sharpe_d_365' indisponible — fallback sur trades.sharpe_ratio (échelle possiblement différente).")
                df["sharpe_d_365"] = df["sharpe_ratio"]

    finally:
        conn.close()

    # Conversion du timestamp
    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"),
        unit="ms",
        errors="coerce",
        utc=True
    )
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filtre Python de sûreté (au cas où) — **pas de cap ici** (cap = clipping modèle)
    n0 = len(df)
    if "is_valid" in df.columns:
        df = df[df["is_valid"].fillna(1) == 1]
    if "flag_sharpe_outlier" in df.columns:
        df = df[df["flag_sharpe_outlier"].fillna(0) == 0]
    n1 = len(df)
    if n0 and (n1 < n0):
        print(f"🛡️  Post-filtre Python: valid & !outlier — gardés={n1}/{n0} ({(n1/n0)*100:.1f}%).")

    print(f"-> Chargement terminé : {len(df)} stratégies chargées")
    return df

# =============================================================================
# PRÉPARATION DES DONNÉES
# =============================================================================
def preprocess_with_scaler(train_df, test_df, feature_cols, target):
    """
    Prépare les tenseurs Torch pour train/test :
    - Fit MinMaxScaler sur train
    - Transform train/test avec les mêmes bornes
    - Conversion en tenseurs Torch (float64)
    """

    scaler = MinMaxScalerDict().fit(train_df, feature_cols)
    X_train_df, feat_used = scaler.transform(train_df, feature_cols)
    X_test_df, _ = scaler.transform(test_df, feature_cols)

    # Conversion en tenseurs Torch
    X_train = torch.tensor(X_train_df.to_numpy(), dtype=torch.double)

    y_tr_raw = train_df[target].astype(float).to_numpy()
    y_tr = np.clip(y_tr_raw, -SHARPE_CAP, SHARPE_CAP)
    if np.any(np.abs(y_tr_raw) > SHARPE_CAP):
        ratio = float((np.abs(y_tr_raw) > SHARPE_CAP).mean()) * 100.0
        print(f"🛡️  Clip target (train): {ratio:.1f}% |S|>{SHARPE_CAP}")

    y_train = torch.tensor(y_tr, dtype=torch.double).unsqueeze(-1)

    X_test = torch.tensor(X_test_df.to_numpy(), dtype=torch.double)
    y_te_raw = test_df[target].astype(float).to_numpy()
    y_te = np.clip(y_te_raw, -SHARPE_CAP, SHARPE_CAP)
    y_test = torch.tensor(y_te, dtype=torch.double).unsqueeze(-1)

    return X_train, y_train, X_test, y_test, feat_used, scaler.bounds

def preprocess_train_only(df, feature_cols, target):
    """
    Prépare uniquement le jeu d’entraînement :
    - Fit MinMaxScaler
    - Transform
    - Conversion en tenseurs Torch
    """

    scaler = MinMaxScalerDict().fit(df, feature_cols)
    X_df, feat_used = scaler.transform(df, feature_cols)

    X = torch.tensor(X_df.to_numpy(), dtype=torch.double)

    y_raw = df[target].astype(float).to_numpy()
    y_np = np.clip(y_raw, -SHARPE_CAP, SHARPE_CAP)
    if np.any(np.abs(y_raw) > SHARPE_CAP):
        ratio = float((np.abs(y_raw) > SHARPE_CAP).mean()) * 100.0
        print(f"🛡️  Clip target (fit): {ratio:.1f}% |S|>{SHARPE_CAP}")

    y = torch.tensor(y_np, dtype=torch.double).unsqueeze(-1)

    return X, y, feat_used, scaler.bounds

def stratified_sample_by_quantiles(
    df: pd.DataFrame,
    target: str,
    max_n: int,
    q_low: float = 0.20,
    q_high: float = 0.80,
    fracs: tuple[float, float, float] = (0.20, 0.60, 0.20),
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Réduit df par échantillonnage stratifié sur le target :
    - bins: <=q_low (low), (q_low, q_high) (mid), >=q_high (high)
    - allocation: fracs (low, mid, high)
    - respecte la taille dispo par bin; complète si besoin.
    """
    if len(df) <= max_n:
        return df.copy()

    rng = np.random.default_rng(random_state)
    lo_thr = df[target].quantile(q_low)
    hi_thr = df[target].quantile(q_high)

    low = df[df[target] <= lo_thr]
    mid = df[(df[target] > lo_thr) & (df[target] < hi_thr)]
    high = df[df[target] >= hi_thr]

    n_low = max(1, int(round(max_n * fracs[0])))
    n_mid = max(1, int(round(max_n * fracs[1])))
    n_high = max(1, max_n - n_low - n_mid)

    def _sample(block: pd.DataFrame, n: int) -> pd.DataFrame:
        n = min(n, len(block))
        if n <= 0:
            return block.iloc[0:0]
        idx = rng.choice(block.index.to_numpy(), size=n, replace=False)
        return block.loc[idx]

    s_low  = _sample(low,  n_low)
    s_mid  = _sample(mid,  n_mid)
    s_high = _sample(high, n_high)

    kept = pd.concat([s_low, s_mid, s_high], axis=0)
    # si on est court (bins clairsemés), compléter aléatoirement
    if len(kept) < max_n:
        remain = df.drop(kept.index, errors="ignore")
        n_more = max_n - len(kept)
        s_more = _sample(remain, n_more)
        kept = pd.concat([kept, s_more], axis=0)

    kept = kept.sample(frac=1.0, random_state=rng.integers(0, 1 << 32)).sort_values("timestamp")
    print(
        f"[GP] Stratified sampling {len(df)} → {len(kept)} "
        f"(low={len(s_low)}/{len(low)}, mid={len(s_mid)}/{len(mid)}, high={len(s_high)}/{len(high)})"
    )
    return kept

# =============================================================================
# MODÈLE GP
# =============================================================================
def fit_gp_model(train_X, train_Y):
    """
    Entraîne un modèle GP avec vraisemblance marginale exacte.

    Args:
        train_X (Tensor): Entrées normalisées [n, d]
        train_Y (Tensor): Sortie cible [n, 1]

    Returns:
        model (SingleTaskGP): Modèle entraîné
    """
    # Check variance avant fit (évite outcome standardize sur cible ~ constante)
    if _safe_std(train_Y) == 0.0:
        raise ValueError("Degenerate fold: std(y_train) ~ 0 → skip GP fit for this fold")

    model = SingleTaskGP(
        train_X, train_Y,
        outcome_transform=SafeStandardize(m=1)  # ← robust to near-constant targets
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _botorch_fit.fit_gpytorch_model(mll)

    # Noise floor minimal
    try:
        noise = model.likelihood.noise
        min_noise = torch.tensor(1e-4, dtype=noise.dtype, device=noise.device)
        with torch.no_grad():
            model.likelihood.noise = torch.clamp(noise, min=min_noise)
    except Exception:
        pass

    model.eval()

    return model

def gp_posterior_mean(model, X):
    """
    Retourne la moyenne postérieure du GP sur un jeu X.
    Utilise torch.no_grad() et fast_pred_var pour accélérer.
    """
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return model.posterior(X).mean.squeeze(-1)

# ================================================================
# CHECK MAPPING (sécurité) & SOFT PRIOR (re-ranking)
# ================================================================
def _check_feature_mapping(feat_used: list[str], bounds_dict: dict):
    """
    Vérifie que l'ordre des colonnes (feat_used) correspond bien aux bornes
    utilisées pour la dénormalisation. Loggue un avertissement si anomalie.
    """
    missing = [c for c in feat_used if c not in bounds_dict]
    extra   = [c for c in bounds_dict.keys() if c not in feat_used]
    if missing or extra:
        print(f"[GP][WARN] Feature mapping mismatch → missing={missing}, extra={extra}")
    # Vérif “ordinale”: on ne peut pas forcer, mais on log l’index attendu
    for i, c in enumerate(feat_used):
        lo, hi = bounds_dict.get(c, (None, None))
        if lo is None or hi is None or not np.isfinite([lo, hi]).all():
            print(f"[GP][WARN] bounds for '{c}' are invalid: {bounds_dict.get(c)}")

def _soft_prior_score(
    X_norm: torch.Tensor,
    feature_names: list[str],
    importance: dict | None,
    center_bias: float = SOFT_PRIOR_CENTER,
):
    """
    Soft prior générique (invariant au scale) :
    - Préfère des valeurs 'centrales' (0.5) sur les dimensions importantes
      pour éviter les extrêmes instables (typique en intraday mid-freq).
    - importance = dict(feature -> poids normalisé). Si None → isotrope.

    Score retourné de taille [n] (plus grand = mieux).
    """
    if X_norm.numel() == 0:
        return torch.empty((0,), dtype=X_norm.dtype, device=X_norm.device)

    d = X_norm.shape[-1]
    if importance is None:
        w = torch.ones(d, dtype=X_norm.dtype, device=X_norm.device) / d
    else:
        w_np = np.array([float(importance.get(f, 0.0)) for f in feature_names], dtype=float)
        if w_np.sum() <= 0:
            w_np = np.ones_like(w_np) / max(1, len(w_np))
        else:
            w_np = w_np / w_np.sum()
        w = torch.tensor(w_np, dtype=X_norm.dtype, device=X_norm.device)

    # pénalisation quadratique de l’éloignement du centre (0.5)
    penalty = ((X_norm - center_bias) ** 2) @ w  # [n]
    score = -penalty  # plus proche du centre ⇒ score plus haut
    return score

def _nn_mo_prior(X_norm_batch: torch.Tensor) -> torch.Tensor:
    """
    Prior 'multi-objectif' pour le re-ranking des candidats :
    approxime le mo_score des points candidats par **plus-proche voisin**
    dans la base entraînement (X_train_norm) pondérée par mo_score.
    Attend que globals()['MO_PRIOR_DB'] soit défini sous la forme:
       {"X": X_train_norm [n,d], "score": torch.tensor([n]) in [0,1]}
    Retourne un tensor [m] avec des valeurs ∈ [0,1] (plus haut = mieux).
    """
    db = globals().get("MO_PRIOR_DB", None)
    if not db:
        return torch.zeros(X_norm_batch.shape[0], dtype=X_norm_batch.dtype, device=X_norm_batch.device)

    Xdb: torch.Tensor = db.get("X", None)
    s: torch.Tensor   = db.get("score", None)
    if Xdb is None or s is None or Xdb.numel() == 0:
        return torch.zeros(X_norm_batch.shape[0], dtype=X_norm_batch.dtype, device=X_norm_batch.device)

    # Distances Euclides normales (espace [0,1]^d)
    # NB : torch.cdist est suffisamment rapide pour ~O(10^3)
    d = torch.cdist(X_norm_batch, Xdb, p=2)
    idx = torch.argmin(d, dim=1)  # plus proche voisin
    prior = s.to(device=X_norm_batch.device, dtype=X_norm_batch.dtype)[idx]
    # Safety clamp
    prior = prior.clamp(0.0, 1.0)
    return prior


# =============================================================================
# WALK-FORWARD CROSS-VALIDATION
# =============================================================================
def make_time_folds(df, date_col: str, n_folds: int):
    """
    Découpe le DataFrame en folds temporels robustes pour WFCV (Walk-Forward CV).

    Règles de robustesse :
    - Trie par `date_col` (datetime), supprime les NaT éventuels;
    - Garantit l'absence de fuite temporelle : train = [start : test_start), test = [test_start : test_end);
    - Impose des tailles minimales : min_test >= min_per_test, min_train >= min_per_train;
    - Réduit automatiquement le nombre de folds si nécessaire (au lieu de produire des folds vides);
    - Évite les folds « crantés » par timestamps dupliqués en coupant sur des indices (et non sur valeurs de date).

    Paramètres
    ----------
    df : pd.DataFrame
        Données d'origine (non triées).
    date_col : str
        Nom de la colonne date/temps (sera convertie en datetime si besoin).
    n_folds : int
        Nombre de folds souhaités (sera réduit si insuffisant).

    Retour
    ------
    list[tuple[pd.DataFrame, pd.DataFrame]]
        Liste de (train_df, test_df) chronologiques, sans fuite ni fold vide.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("make_time_folds: 'df' doit être un pandas.DataFrame")

    if date_col not in df.columns:
        raise ValueError(f"make_time_folds: '{date_col}' est introuvable dans df.columns")

    # 1) Normalisation & tri
    df_local = df.copy()
    if not np.issubdtype(df_local[date_col].dtype, np.datetime64):
        df_local[date_col] = pd.to_datetime(df_local[date_col], errors="coerce", utc=True)
    df_local = df_local.dropna(subset=[date_col]).sort_values(date_col, kind="mergesort")  # stable

    n = len(df_local)
    if n < 10:
        # Trop court : renvoyer au moins 1 pseudo-fold 80/20 si possible
        split = max(1, int(round(0.8 * n)))
        train_df = df_local.iloc[:split].copy()
        test_df = df_local.iloc[split:].copy()
        return [(train_df, test_df)] if len(test_df) > 0 else []

    # 2) Tailles minimales robustes (train/test)
    #    - min par défaut : environ n_folds « équilibrés » mais avec plancher
    min_per_test = max(20, n // (n_folds * 3))   # test suffisamment grand pour avoir de la variance
    min_per_train = max(40, n // (n_folds * 2))  # train plus grand que test

    # 3) Si n_folds est trop ambitieux, on le réduit
    #    Test: faut-il au moins min_per_train + min_per_test pour le premier fold ?
    max_folds_theorique = max(1, (n - min_per_train) // max(1, min_per_test))
    n_f = int(min(n_folds, max_folds_theorique))
    if n_f < 1:
        # fallback 1 fold 80/20
        split = max(1, int(round(0.8 * n)))
        train_df = df_local.iloc[:split].copy()
        test_df = df_local.iloc[split:].copy()
        return [(train_df, test_df)] if len(test_df) > 0 else []

    # 4) Construction des bornes de test « égales » en indices (quantiles d'index)
    #    On fabrique n_f segments test contigus, le train = tout avant chaque segment.
    #    Les coupes se font en indices pour éviter les problèmes de timestamps dupliqués.
    #    On s'assure de respecter min_per_train/min_per_test; sinon on saute le fold.
    idx = np.arange(n, dtype=int)
    # points de coupe en indices pour n_f segments ~égaux
    cuts = np.linspace(0, n, num=n_f + 1, endpoint=True).astype(int)

    folds = []
    for k in range(n_f):
        test_start = cuts[k]
        test_end = cuts[k + 1]

        # Impose min tailles
        # - test
        if (test_end - test_start) < min_per_test:
            continue
        # - train
        if test_start < min_per_train:
            # Pas assez de train pour ce fold → on essaie de « pousser » le test plus loin
            # pour libérer plus de train; si ce n'est pas possible, on skip.
            delta = min_per_train - test_start
            if (test_end + delta) <= n:
                test_start += delta
                test_end += delta
            else:
                continue

        # Définit train = [0 : test_start), test = [test_start : test_end)
        train_df = df_local.iloc[:test_start].copy()
        test_df = df_local.iloc[test_start:test_end].copy()

        # Sécurité : pas de fuite ni fold vide
        if len(train_df) < min_per_train or len(test_df) < min_per_test:
            continue
        if train_df[date_col].max() >= test_df[date_col].min():
            # Dans le doute (timestamps dupliqués exactement à la frontière), on pousse d'un cran
            # pour garantir la flèche du temps
            test_df = test_df.iloc[1:].copy()
            if len(test_df) < min_per_test:
                continue

        folds.append((train_df, test_df))

    # 5) Dernier filet de sécurité : si rien n'a passé, on fabrique 1 fold 80/20
    if not folds:
        split = max(min_per_train, int(round(0.8 * n)))
        split = min(split, n - 1)  # laisser au moins 1 obs en test
        train_df = df_local.iloc[:split].copy()
        test_df = df_local.iloc[split:].copy()
        if len(test_df) >= 1:
            folds = [(train_df, test_df)]

    return folds

def evaluate_gp_model_wfcv(df, feature_cols, target=TARGET, date_col="timestamp", n_folds: int = WFCV_N_FOLDS):
    """
    Évalue le GP via Walk-Forward Cross-Validation **robuste** :
    - Split temporel en n_folds
    - Winsorisation (1–99 %) du target **uniquement** pour l'évaluation WFCV
      (on évite d’écraser la variance par le clip dur ±SHARPE_CAP)
    - Garde-fous variance (train/test) + corrélation sûre
    - Télémétrie best-effort dans adaptive_decisions
    Retourne une moyenne de corrélations **toujours définie** (jamais NaN).
    """
    folds = make_time_folds(df, date_col, n_folds)
    scores = []
    n_total = 0
    n_valid = 0
    n_skipped_train = 0
    n_skipped_test = 0
    n_const_pred = 0

    for train_df, test_df in folds:
        n_total += 1

        # 1) Prépare X via le scaler existant (cohérence de normalisation)
        Xtr, ytr_clip, Xte, yte_clip, feat_used, _ = preprocess_with_scaler(
            train_df, test_df, feature_cols, target
        )

        # 2) Winsorise la cible pour la WFCV (préserve la variance intra-fold)
        y_tr_np = _winsorize_series(train_df[target].astype(float).to_numpy(), 0.01, 0.99)
        y_te_np = _winsorize_series(test_df[target].astype(float).to_numpy(),  0.01, 0.99)

        # Remplace y_train tensor par la version winsorisée
        ytr = torch.tensor(y_tr_np, dtype=torch.double).unsqueeze(-1)

        # 3) Garde-fous : pas de fit si train quasi-constant / trop pauvre
        if Xtr.shape[1] == 0 or _safe_std(ytr) == 0.0 or np.unique(y_tr_np).size < 3:
            scores.append(0.0)         # score neutre au lieu de NaN
            n_skipped_train += 1
            continue

        # 4) Fit GP robuste (SafeStandardize + noise floor déjà dans fit_gp_model)
        try:
            model = fit_gp_model(Xtr, ytr)
        except Exception:
            scores.append(0.0)
            n_skipped_train += 1
            continue

        # 5) Prédiction sur test
        try:
            y_pred = gp_posterior_mean(model, Xte).cpu().numpy()
        except Exception:
            scores.append(0.0)
            n_const_pred += 1
            continue

        # 6) Score robuste (corrélation sûre). On n'utilise PAS le yte clipé.
        if _safe_std(y_te_np) == 0.0:
            scores.append(0.0)
            n_skipped_test += 1
            continue

        s = _safe_corr(y_te_np, y_pred)
        if s == 0.0 and _safe_std(y_pred) == 0.0:
            n_const_pred += 1
        else:
            n_valid += 1
        scores.append(float(s))

    # 7) Agrégation **toujours définie**
    mean_score = float(np.mean(scores)) if len(scores) else 0.0
    if not np.isfinite(mean_score):
        mean_score = 0.0

    # 8) Télémétrie (best-effort) — utilise BACKTEST_DB_PATH si présent
    try:
        db_path = os.environ.get("BACKTEST_DB_PATH", "")
        if db_path:
            _adaptive_log_wfcv(db_path, dict(
                n_total=int(n_total),
                n_valid=int(n_valid),
                n_skipped_train=int(n_skipped_train),
                n_skipped_test=int(n_skipped_test),
                n_const_pred=int(n_const_pred),
                mean_corr=mean_score,
            ))
    except Exception:
        pass

    return mean_score


# =============================================================================
# ACQUISITION BAYÉSIENNE
# =============================================================================

def _describe_bounds(bounds: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    """
    Résume une boîte [2, d] en espace normalisé.

    Returns
    -------
    avg_width : float
        Largeur moyenne sur les d dimensions.
    volume : float
        Produit des largeurs (calculé en log-sum-exp pour éviter les underflows).
    widths : torch.Tensor
        Largeurs par dimension, shape [d].
    """
    w = (bounds[1] - bounds[0]).clamp_min(1e-12)
    avg_w = float(w.mean().item())
    vol = float(torch.exp(torch.log(w).sum()).item())
    return avg_w, vol, w


def _warn_restrictive(stage: str, **kv):
    """
    Imprime un avertissement lisible lorsque l’ensemble de candidats s’effondre (~0).
    Les paramètres clés (ex: distance_thr, base_r) sont imprimés pour post-mortem.
    """
    payload = ", ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[WARN][GP] Hyperparamètres trop restrictifs @ {stage} → aucun / quasi-aucun candidat. {payload}")

def _summarize_weights(_weights, _names):
    # imprime les 5 dimensions les plus “penalisées” par la métrique de diversité
    try:
        if _weights is None:
            print("[GP] Diversity weights: isotropic (no importance provided)")
            return
        w = _weights.detach().cpu().numpy()
        order = np.argsort(w)[::-1][:5]
        top = [(str(_names[i]), float(w[i])) for i in order]
        print(f"[GP] Diversity weights (top5): {top}")
    except Exception:
        # on ne casse jamais le flux d’optim, ce log est purement informatif
        pass

# ------------------------------------------------------------
# Fallback si trop peu de points pour fitter un GP de façon stable
# ------------------------------------------------------------
GP_MIN_TRAIN: int = int(os.getenv("GP_MIN_TRAIN", "12"))  # seuil 10–15 recommandé

def _fallback_suggest_local_random(
    *,
    X_train_norm: torch.Tensor,
    y_train: torch.Tensor,
    feat_used: list[str],
    bounds_dict: dict[str, tuple[float, float]],
    n_trials: int,
    tr_radius: float = 0.22,
    local_share: float = 0.7,
    seed: int | None = None,
):
    """
    Génère des suggestions **sans modèle GP** lorsque n_train est trop faible :
      - ~local : petites perturbations autour du meilleur point observé
      - ~global : quasi-random uniforme dans [0,1]^d
    Puis dénormalise avec bounds_dict et applique les contraintes légères.

    Paramètres
    ----------
    X_train_norm : Tensor [n, d]  — features déjà **normalisées** en [0,1]
    y_train      : Tensor [n, 1]  — cible clipée
    feat_used    : ordre des features (colonnes)
    bounds_dict  : dict(feature -> (min, max)) appris sur le train
    n_trials     : nombre total de suggestions à produire
    tr_radius    : rayon local en espace normalisé
    local_share  : proportion locale (reste → global)
    seed         : graine optionnelle
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = X_train_norm.device
    dtype  = X_train_norm.dtype
    d = X_train_norm.shape[-1]

    # centre = meilleur y_train (déjà clipé) en **espace normalisé**
    best_idx = torch.argmax(y_train.squeeze(-1))
    center   = X_train_norm[best_idx].detach().clone()  # [d]

    q_local  = max(1, int(round(local_share * n_trials)))
    q_global = max(0, n_trials - q_local)

    # --- LOCAL : échantillonne dans une TR hyper-rectangle autour du centre
    lo_loc = (center - tr_radius).clamp(0.0, 1.0)
    hi_loc = (center + tr_radius).clamp(0.0, 1.0)
    if q_local > 0:
        U = torch.rand((q_local, d), dtype=dtype, device=device)
        X_loc = lo_loc + U * (hi_loc - lo_loc + 1e-12)
    else:
        X_loc = torch.empty((0, d), dtype=dtype, device=device)

    # --- GLOBAL : quasi-random uniforme
    if q_global > 0:
        X_glo = torch.rand((q_global, d), dtype=dtype, device=device)
    else:
        X_glo = torch.empty((0, d), dtype=dtype, device=device)

    X_batch = torch.vstack([X_loc, X_glo]).clamp(0.0, 1.0)

    # --- DÉNORMALISATION + contraintes
    def _denorm_row(row_np):
        return {
            k: float(row_np[j]) * (bounds_dict[k][1] - bounds_dict[k][0]) + bounds_dict[k][0]
            for j, k in enumerate(feat_used)
            if k in bounds_dict and np.isfinite(bounds_dict[k]).all()
        }

    suggestions = []
    for i in range(X_batch.shape[0]):
        row = X_batch[i].detach().cpu().numpy()
        den  = _denorm_row(row)
        den  = enforce_constraints_for_gp(den, bounds_dict, domain_bounds=DOMAIN_BOUNDS)
        suggestions.append(den)

    return suggestions[:n_trials]

def suggest_candidates(
    model,
    X_train_norm,               # << nouveau : X d'entraînement normalisé [n,d]
    y_train,
    feat_used,
    bounds_dict,
    n_trials=10,
    q_batch: int = Q_BATCH,           # << taille du lot
    local_share: float = LOCAL_SHARE,   # ~70% local (trust-region), ~30% global
    tr_radius: float = TR_RADIUS_DEFAULT,    # rayon initial de trust-region (espace normalisé)
    min_l2: float = MIN_L2_DEFAULT,       # distance minimale L2 en [0,1]
    topk_centers: int = TOPK_CENTERS_DEFAULT,      # nb de centres TR (meilleures configs)
    max_retries: int = MAX_RETRIES_DEFAULT,       # réessais en cas de diversité insuffisante
    seed: int | None = None,
    X_evaluated_norm=None,      # points déjà évalués (par défaut = X_train_norm)
    X_pending_norm=None,        # points déjà “lancés” mais non scorés (optionnel)
    distance_thr: float | None = DISTANCE_THR_OVERRIDE,      # override du seuil de distance
    q_local_override: int | None = Q_LOCAL_OVERRIDE,    # override q_local
    q_global_override: int | None = Q_GLOBAL_OVERRIDE,   # override q_global
    center_radius_schedule: tuple[float, ...] = CENTER_RADIUS_SCHEDULE,  # shrink/expand selon le rang
    rerank_alpha: float = RERANK_ALPHA,              # poids mean(GP)
    rerank_beta: float = RERANK_BETA,               # poids soft-prior
):
    """
    Génère des suggestions par lots (q-batch) en combinant :
      - acquisition locale sur des trust-regions autour des meilleurs points
      - acquisition globale sur l’espace complet
      - anti-clustering (vs évalué, vs pending, et intra-batch)
      - X_pending pour décourager les collisions
    Retourne une liste de dicts d’hyperparamètres (dénormalisés).
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = X_train_norm.device
    dtype = X_train_norm.dtype
    dim = X_train_norm.shape[-1]

    # --- Features & importance (optionnel) ---
    # feat_used vient de preprocess_* et correspond à l'ordre des colonnes (d features)
    feature_names = list(feat_used)

    # Si tu exposes un dict global de SHAP importances (ex: {feature: mean_abs_shap}),
    # on l'utilise; sinon None -> fallback isotrope.
    importance = globals().get("SHAP_IMPORTANCE_DICT", None)

    # Poids de distance: >1 => dimension "importante" (distance plus stricte)
    #                    <1 => dimension "faible"   (distance plus tolérante)
    weights = make_distance_weights(
        feature_names=feature_names,
        importance=importance,     # dict|None (fallback isotrope si None)
        device=device,
        dtype=dtype,
        floor=DIVERSITY_WEIGHT_FLOOR,                 # plancher de tolérance pour dims peu importantes
        ceil=DIVERSITY_WEIGHT_CEIL,                  # plus strict sur dims importantes
    )

    # --- Seuil de distance L2 (cohérent avec TR)  # NEW ↓↓↓
    if distance_thr is None:
        distance_thr = max(min_l2, tr_radius * 0.6)
    else:
        distance_thr = float(distance_thr)

    _summarize_weights(weights, feature_names)

    # --- Bornes globales [0,1]^d
    BOUNDS = torch.stack([torch.zeros(dim, dtype=dtype, device=device),
                          torch.ones(dim,  dtype=dtype, device=device)])

    # --- Acquisition qEI (Monte Carlo)
    best_f = y_train.max().detach()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLE_SHAPE]))  # 128–512 OK selon budget
    # ACQ_CLASS est soit qLogExpectedImprovement (si dispo), soit qExpectedImprovement (fallback)
    acq_global = ACQ_CLASS(model=model, best_f=best_f, sampler=sampler)

    # --- Centres de trust-region = top-k points selon y_train
    n = y_train.shape[0]
    topk = min(topk_centers, max(1, n))
    idx_top = torch.topk(y_train.squeeze(-1), k=topk).indices
    centers = X_train_norm[idx_top]  # [k, d]

    # tensors utilitaires
    if X_evaluated_norm is None:
        X_evaluated_norm = X_train_norm
    if X_pending_norm is None:
        X_pending_norm = torch.empty((0, dim), dtype=dtype, device=device)

    # Sélection finale en NORMALISÉ
    selected_norm = torch.empty((0, dim), dtype=dtype, device=device)

    # Helpers ---------------------------------------------------------

    def denorm_row(row_np):
        return {
            k: float(row_np[j]) * (bounds_dict[k][1] - bounds_dict[k][0]) + bounds_dict[k][0]
            for j, k in enumerate(feat_used)
        }

    # ---------------------------------------------------------------

    suggestions = []
    needed_total = n_trials
    # --- Cibles local/global par batch
    if q_local_override is not None and q_global_override is not None:
        q_local_target = int(q_local_override)
        q_global_target = int(q_global_override)
    else:
        q_local_target = int(round(local_share * q_batch))
        q_global_target = q_batch - q_local_target

    # On boucle par “batches” jusqu’à atteindre n_trials
    while len(suggestions) < needed_total:
        # 1) ----- LOCAL : optimisations sur TR autour des meilleurs centres -----
        local_selected = torch.empty((0, dim), dtype=dtype, device=device)
        q_local_remaining = q_local_target

        # On itère sur les centres pour récolter des sous-batchs
        for rank, c in enumerate(centers):
            if q_local_remaining <= 0:
                break
            sub_q = min(q_local_remaining, max(1, q_batch // max(1, centers.shape[0])))

            # facteur de radius par rang
            scale_rank = center_radius_schedule[min(rank, len(center_radius_schedule) - 1)]
            base_r = float(tr_radius) * float(scale_rank)

            # >>> anisotropie par dimension : on produit un vecteur d’échelles
            try:
                scales = scale_TR_per_dimension(
                    feature_names=feature_names,
                    importance=importance,              # dict|None
                    base_radius=base_r,
                    min_scale=ANISO_MIN_SCALE,
                    max_scale=ANISO_MAX_SCALE,
                    device=device,
                    dtype=dtype,
                )
                bounds_tr = restrict_bounds_around(center=c, scales=scales)
            except Exception:
                # Fallback feature-aware (moins fin mais robuste)
                bounds_tr = restrict_bounds_feature_aware(
                    center=c,
                    base_radius=base_r,
                    feature_names=feature_names,
                    importance=importance,
                    min_scale=ANISO_MIN_SCALE,
                    max_scale=ANISO_MAX_SCALE,
                )

            # --- Diagnostics TR / espace explorable ---
            _avg_w, _vol, _w = _describe_bounds(bounds_tr)
            try:
                _minw = float(_w.min().item()); _maxw = float(_w.max().item())
            except Exception:
                _minw = _maxw = float('nan')
            print(
                f"[DBG][TR] center={rank+1}/{centers.shape[0]} | "
                f"base_r={base_r:.3f} | avg_width={_avg_w:.4f} | "
                f"min/max_width=({_minw:.4f},{_maxw:.4f}) | "
                f"volume≈{_vol:.3e} | distance_thr={distance_thr:.3f} | q_target={sub_q}"
            )

            # Même principe : on passe toujours par ACQ_CLASS pour rester version-agnostique.
            acq_local = ACQ_CLASS(model=model, best_f=best_f, sampler=sampler)

            X_pending_all = torch.vstack([X_pending_norm, selected_norm, local_selected])
            if X_pending_all.shape[0] > 0:
                acq_local.set_X_pending(X_pending_all)

            # multi-start “robuste” : plus de redémarrages + raw_samples élevés
            if USE_GEN_MULTI_START_INITS:
                try:
                    inits_loc = gen_multi_start_inits(
                        acq_function=acq_local,
                        bounds=bounds_tr,
                        q=sub_q,
                        num_restarts=ACQ_LOCAL_NUM_RESTARTS,
                        raw_samples=ACQ_LOCAL_RAW_SAMPLES,
                        device=device,
                        dtype=dtype,
                    )
                    # Certaines versions de BoTorch acceptent batch_initial_conditions :
                    try:
                        cand_loc, _ = optimize_acqf(
                            acq_function=acq_local,
                            bounds=bounds_tr,
                            q=sub_q,
                            batch_initial_conditions=inits_loc,
                            options={"maxiter": ACQ_MAXITER},
                        )
                    except TypeError:
                        # fallback : on récupère des (num_restarts, raw_samples) proposés par utils
                        cand_loc, _ = optimize_acqf(
                            acq_function=acq_local,
                            bounds=bounds_tr,
                            q=sub_q,
                            num_restarts=ACQ_LOCAL_NUM_RESTARTS,
                            raw_samples=ACQ_LOCAL_RAW_SAMPLES,
                            options={"maxiter": ACQ_MAXITER},
                        )
                except Exception:
                    # fallback brut si inits indisponibles
                    cand_loc, _ = optimize_acqf(
                        acq_function=acq_local,
                        bounds=bounds_tr,
                        q=sub_q,
                        num_restarts=ACQ_LOCAL_NUM_RESTARTS,
                        raw_samples=ACQ_LOCAL_RAW_SAMPLES,
                        options={"maxiter": ACQ_MAXITER},
                    )
            else:
                cand_loc, _ = optimize_acqf(
                    acq_function=acq_local,
                    bounds=bounds_tr,
                    q=sub_q,
                    num_restarts=ACQ_LOCAL_NUM_RESTARTS,
                    raw_samples=ACQ_LOCAL_RAW_SAMPLES,
                    options={"maxiter": ACQ_MAXITER},
                )

            # Anti-clustering vs évalués, vs pending, et intra-batch local
            _pre = int(cand_loc.shape[0])
            mask = far_enough(
                cand_loc,
                torch.vstack([X_evaluated_norm, X_pending_norm, selected_norm, local_selected]),
                thr=distance_thr,
                weights=weights
            )
            _after_mask = int(mask.sum().item())
            cand_loc = cand_loc[mask]
            cand_loc = greedy_poisson_selection(
                cands=cand_loc,
                existing=torch.vstack([X_evaluated_norm, X_pending_norm, selected_norm]),
                thr=distance_thr,
                weights=weights,
                k_max=q_batch,  # évite de "sur-remplir" une niche
            )
            _after_div = int(cand_loc.shape[0])
            print(f"[DBG][CANDS][local] pre={_pre} | after_mask={_after_mask} | after_diversity={_after_div} | thr={distance_thr:.3f}")
            if _after_div <= 1:
                _warn_restrictive(stage="local", base_r=round(base_r, 3), distance_thr=round(distance_thr, 3))

            if cand_loc.shape[0] > 0:
                local_selected = torch.vstack([local_selected, cand_loc])
                q_local_remaining -= cand_loc.shape[0]

                # 2) ----- GLOBALE : compléter sur l’espace complet -----
        global_selected = torch.empty((0, dim), dtype=dtype, device=device)
        # Si le local n'a pas rempli sa cible, on “reporte” le manque sur le global
        q_global_remaining = q_global_target + max(0, q_local_target - local_selected.shape[0])

        if q_global_remaining > 0:
            X_pending_all = torch.vstack([X_pending_norm, selected_norm, local_selected])
            if X_pending_all.shape[0] > 0:
                acq_global.set_X_pending(X_pending_all)

            # Initialisations multi-start pour la passe GLOBALE (sur [0,1]^d)
            if USE_GEN_MULTI_START_INITS:
                try:
                    inits_glob = gen_multi_start_inits(
                        acq_function=acq_global,
                        bounds=BOUNDS,
                        q=q_global_remaining,
                        num_restarts=ACQ_GLOBAL_NUM_RESTARTS,
                        raw_samples=ACQ_GLOBAL_RAW_SAMPLES,
                        device=device,
                        dtype=dtype,
                    )
                    try:
                        cand_glob, _ = optimize_acqf(
                            acq_function=acq_global,
                            bounds=BOUNDS,
                            q=q_global_remaining,
                            batch_initial_conditions=inits_glob,
                            options={"maxiter": ACQ_MAXITER},
                        )
                    except TypeError:
                        cand_glob, _ = optimize_acqf(
                            acq_function=acq_global,
                            bounds=BOUNDS,
                            q=q_global_remaining,
                            num_restarts=ACQ_GLOBAL_NUM_RESTARTS,
                            raw_samples=ACQ_GLOBAL_RAW_SAMPLES,
                            options={"maxiter": ACQ_MAXITER},
                        )
                except Exception:
                    cand_glob, _ = optimize_acqf(
                        acq_function=acq_global,
                        bounds=BOUNDS,
                        q=q_global_remaining,
                        num_restarts=ACQ_GLOBAL_NUM_RESTARTS,
                        raw_samples=ACQ_GLOBAL_RAW_SAMPLES,
                        options={"maxiter": ACQ_MAXITER},
                    )
            else:
                cand_glob, _ = optimize_acqf(
                    acq_function=acq_global,
                    bounds=BOUNDS,
                    q=q_global_remaining,
                    num_restarts=ACQ_GLOBAL_NUM_RESTARTS,
                    raw_samples=ACQ_GLOBAL_RAW_SAMPLES,
                    options={"maxiter": ACQ_MAXITER},
                )

            # Anti-clustering vs évalués, vs pending, et intra-batch global
            _pre_g = int(cand_glob.shape[0])
            mask = far_enough(
                cand_glob,
                torch.vstack([X_evaluated_norm, X_pending_norm, selected_norm, local_selected]),
                thr=distance_thr,
                weights=weights
            )
            _after_mask_g = int(mask.sum().item())
            cand_glob = cand_glob[mask]
            cand_glob = greedy_poisson_selection(
                cands=cand_glob,
                existing=torch.vstack([X_evaluated_norm, X_pending_norm, selected_norm]),
                thr=distance_thr,
                weights=weights,
                k_max=q_batch,
            )
            _after_div_g = int(cand_glob.shape[0])
            print(f"[DBG][CANDS][global] pre={_pre_g} | after_mask={_after_mask_g} | after_diversity={_after_div_g} | thr={distance_thr:.3f}")
            if _after_div_g <= 1:
                _warn_restrictive(stage="global", distance_thr=round(distance_thr, 3))

            if cand_glob.shape[0] > 0:
                global_selected = torch.vstack([global_selected, cand_glob])

        # 3) ----- Fusion du batch courant -----
        batch_selected = torch.vstack([local_selected, global_selected])

        # --- Soft-prior re-ranking (améliore la stabilité mid-freq)
        if batch_selected.shape[0] > 0:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mu = model.posterior(batch_selected).mean.squeeze(-1)  # [m]

            # Prior multi-objectif basé sur **mo_score** (NN sur X_train_norm)
            try:
                prior = _nn_mo_prior(batch_selected)  # [m] in [0,1]
                # NB : pas de center_bias ici — le prior MO remplace le prior isotrope
                #      pour guider vers des zones respectant mieux le multi-objectif.
            except Exception:
                # Fallback robuste si NN prior indisponible
                prior = _soft_prior_score(
                    X_norm=batch_selected,
                    feature_names=feature_names,
                    importance=importance,
                    center_bias=SOFT_PRIOR_CENTER,
                )

            # score composite : alpha * mean(GP) + beta * prior(MO)
            score = rerank_alpha * mu + rerank_beta * prior

            # >>> bonus ATR (tp/sl) : prior doux “box” sur features ciblées
            try:
                bonus_box = apply_soft_box_prior(
                    X_norm=batch_selected,
                    feature_names=feature_names,
                    target_features=list(BOX_PRIOR_FEATURES),
                    center=BOX_PRIOR_CENTER,
                    gamma=BOX_PRIOR_GAMMA,
                    importance=importance,
                )
                score = score + bonus_box
            except Exception:
                pass

            order = torch.argsort(score, descending=True)
            batch_selected = batch_selected[order]

        if batch_selected.shape[0] == 0:
            _warn_restrictive(stage="batch", distance_thr=round(distance_thr, 3))
            # on élargit la TR et on réessaie (quelques fois)
            retries = 0

            while batch_selected.shape[0] == 0 and retries < max_retries:
                retries += 1
                tr_radius = min(0.5, tr_radius * 1.5)
                # relance une passe globale “secours”
                X_pending_all = torch.vstack([X_pending_norm, selected_norm])
                if X_pending_all.shape[0] > 0:
                    acq_global.set_X_pending(X_pending_all)

                if USE_GEN_MULTI_START_INITS:
                    try:
                        inits_glob = gen_multi_start_inits(
                            acq_function=acq_global,
                            bounds=BOUNDS,
                            q=q_global_remaining,
                            num_restarts=ACQ_GLOBAL_NUM_RESTARTS,
                            raw_samples=ACQ_GLOBAL_RAW_SAMPLES,
                            device=device,
                            dtype=dtype,
                        )
                        try:
                            cand_glob, _ = optimize_acqf(
                                acq_function=acq_global,
                                bounds=BOUNDS,
                                q=q_global_remaining,
                                batch_initial_conditions=inits_glob,
                                options={"maxiter": ACQ_MAXITER},
                            )
                        except TypeError:
                            cand_glob, _ = optimize_acqf(
                                acq_function=acq_global,
                                bounds=BOUNDS,
                                q=q_global_remaining,
                                num_restarts=ACQ_GLOBAL_NUM_RESTARTS,
                                raw_samples=ACQ_GLOBAL_RAW_SAMPLES,
                                options={"maxiter": ACQ_MAXITER},
                            )
                    except Exception:
                        cand_glob, _ = optimize_acqf(
                            acq_function=acq_global,
                            bounds=BOUNDS,
                            q=q_global_remaining,
                            num_restarts=ACQ_GLOBAL_NUM_RESTARTS,
                            raw_samples=ACQ_GLOBAL_RAW_SAMPLES,
                            options={"maxiter": ACQ_MAXITER},
                        )
                else:
                    cand_glob, _ = optimize_acqf(
                        acq_function=acq_global,
                        bounds=BOUNDS,
                        q=q_global_remaining,
                        num_restarts=ACQ_GLOBAL_NUM_RESTARTS,
                        raw_samples=ACQ_GLOBAL_RAW_SAMPLES,
                        options={"maxiter": ACQ_MAXITER},
                    )

                mask = far_enough(
                    cand_glob,
                    torch.vstack([X_evaluated_norm, X_pending_norm, selected_norm]),
                    thr=distance_thr * 0.8,
                    weights=weights
                )
                batch_selected = greedy_poisson_selection(
                    cands=cand_glob[mask],
                    existing=torch.vstack([X_evaluated_norm, X_pending_norm, selected_norm]),
                    thr=distance_thr * 0.8,
                    weights=weights,
                    k_max=q_batch,
                )

            if batch_selected.shape[0] == 0:
                # On remplit en Sobol-like (uniforme) si vraiment rien
                m = min(q_batch, needed_total - len(suggestions))
                rnd = torch.rand((m, dim), dtype=dtype, device=device)
                mask = far_enough(
                    rnd,
                    torch.vstack([X_evaluated_norm, X_pending_norm, selected_norm]),
                    thr=distance_thr * 0.5,
                    weights=weights
                )
                batch_selected = rnd[mask]

        # 4) ----- Mise à jour pending + sélection finale -----
        if batch_selected.numel() > 0:
            selected_norm = torch.vstack([selected_norm, batch_selected])
            X_pending_norm = torch.vstack([X_pending_norm, batch_selected])

        # 5) ----- Dénormalisation & contraintes légères -----
        # on clippe si besoin (déjà [0,1] normalement)
        batch_selected = batch_selected.clamp(0.0, 1.0)  

        for i in range(batch_selected.shape[0]):          
            if len(suggestions) >= needed_total:
                break
            row = batch_selected[i].detach().cpu().numpy()
            denorm = denorm_row(row)
            denorm = enforce_constraints_for_gp(denorm, bounds_dict, domain_bounds=DOMAIN_BOUNDS)
            suggestions.append(denorm)

        # Si on n'a pas encore atteint n_trials, on boucle : le pending actuel
        # est communiqué aux acquisitions des prochains sous-batchs (acq.set_X_pending)
    return suggestions[:n_trials]

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def suggest_with_gp_pipeline(
    db_path,
    n_trials: int = 10,
    test_ratio: float = TEST_RATIO_DEFAULT,
    tr_radius: float = TR_RADIUS_DEFAULT,
    *,
    q_local_override: int | None = None,
    q_global_override: int | None = None,
    distance_thr: float | None = None,
):
    """
    Pipeline complet :
    1. Chargement & tri temporel des données
    2. Split train/test
    3. Sélection des features
    4. Walk-Forward CV pour valider le GP
    5. Entraînement final du GP sur train
    6. Génération de suggestions d’hyperparamètres

    Args:
        db_path: Chemin SQLite source (logs/trades).
        n_trials: Nombre de suggestions à produire.
        test_ratio: Ratio temporel réservé au test (WFCV).
        tr_radius: Rayon de Trust-Region en espace normalisé [0,1].
        q_local_override: (optionnel) force le nombre de candidats “locaux”.
        q_global_override: (optionnel) force le nombre de candidats “globaux”.
        distance_thr: (optionnel) seuil de distance minimale (anti-clustering).
    """

    # Chargement
    df = load_data(db_path)

    # Sûreté: si des colonnes existent encore dans df (selon SELECT), on filtre ici aussi
    n_all = len(df)
    if "is_valid" in df.columns:
        df = df[df["is_valid"].fillna(1) == 1]
    if "flag_sharpe_outlier" in df.columns:
        df = df[df["flag_sharpe_outlier"].fillna(0) == 0]
    if n_all and len(df) < n_all:
        print(
            f"🧹 Dataset GP filtré: {len(df)}/{n_all} restants (valid & !outlier). "
            f"(Sharpe cap={SHARPE_CAP} appliqué par **clipping** dans le modèle, pas en filtre)"
        )

    split_idx = int(len(df) * (1 - test_ratio))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    # --- Réduction par STRATIFICATION si train trop gros ou trop “top-heavy”
    GP_MAX_TRAIN = int(os.getenv("GP_MAX_TRAIN", "500"))   # plafond raisonnable
    if len(train_df) > GP_MAX_TRAIN:
        train_df = stratified_sample_by_quantiles(
            df=train_df,
            target=TARGET,
            max_n=GP_MAX_TRAIN,
            q_low=0.20, q_high=0.80,
            fracs=(0.20, 0.60, 0.20),
            random_state=None,
        )

    # Features
    feature_cols = [
        c for c in train_df.columns
        if c not in [TARGET, "timestamp", "backtest_id"]
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    EXPECTED_FEATS = [
        "ema_short_period","ema_long_period","rsi_period","rsi_buy_zone","rsi_sell_zone",
        "rsi_past_lookback","atr_tp_multiplier","atr_sl_multiplier","atr_period",
        "macd_signal_period","rsi_thresholds_1m","rsi_thresholds_5m","rsi_thresholds_15m",
        "rsi_thresholds_1h","ewma_period","weight_atr_combined_vol","threshold_volume",
        "hist_volum_period","detect_supp_resist_period","trend_period","threshold_factor",
        "min_profit_margin","resistance_buffer_margin","risk_reward_ratio","confidence_score_params",
        "signal_weight_bonus","penalite_resistance_factor","penalite_multi_tf_step",
        "override_score_threshold","rsi_extreme_threshold","signal_pure_threshold","signal_pure_weight",
    ]

    missing = [c for c in EXPECTED_FEATS if c not in feature_cols]
    extra   = [c for c in feature_cols if c not in EXPECTED_FEATS]
    print(f"[DBG][FEATS] p={len(feature_cols)} | missing={missing} | extra={extra}")

    # --- Forcer le même set de features que le surrogate ---
    allowed = _load_surrogate_feature_list()
    if allowed:
        drop_cols = {"is_valid", "flag_sharpe_outlier"}
        allowed_final = [c for c in allowed if c not in drop_cols]
        # restreint la liste de colonnes utilisables
        missing = [c for c in allowed_final if c not in train_df.columns]
        extra   = [c for c in feature_cols if c not in allowed_final]
        if missing:
            print(f"[GP][WARN] Features manquantes vs surrogate: {missing}")
        if extra:
            print(f"[GP][INFO] Colonnes ignorées (non dans surrogate): {extra}")
        feature_cols = [c for c in allowed_final if c in train_df.columns]

    # --- Logs PRE-FIT (sur DataFrame brut, pas sur des tensors inexistants) ---
    p_pre = len(feature_cols)
    y_std_pre = float(
        train_df[TARGET].astype(float).clip(-SHARPE_CAP, SHARPE_CAP).std(ddof=1)
    ) if len(train_df) > 1 else 0.0
    print(f"[DBG] pre-fit: p={p_pre} | n_train={len(train_df)} | std(y)={y_std_pre:.6f}")

    # (⚠️ supprimer/neutraliser le return avec `...` ; ne pas faire de fallback ici)

    # WFCV
    mean_corr = evaluate_gp_model_wfcv(train_df, feature_cols, target=TARGET, date_col="timestamp", n_folds=3)
    print(f"📊 Corrélation moyenne WFCV (train): {mean_corr:.4f}")

    # Fit final (+ fallback si trop peu d'exemples)
    X_train, y_train, feat_used, bounds = preprocess_train_only(train_df, feature_cols, TARGET)
    n_train = int(X_train.shape[0])
    p = int(X_train.shape[1]) if X_train.ndim == 2 else 0
    y_std = float(y_train.std().item()) if y_train.numel() > 0 else 0.0

    # 🔎 Diagnostics avant fit
    print(f"[DBG] X_train.shape = {tuple(X_train.shape)} | p={p} | n_train={n_train} | std(y)={y_std:.6f}")
    try:
        # 10 features avec le moins de diversité + leurs nunique()
        if len(feat_used) > 0:
            nunq = train_df[list(feat_used)].nunique(dropna=False).sort_values()
            print("[DBG] nunique() par feature (10 plus faibles) :")
            print(nunq.head(10).to_string())
            # Aperçu des 10 premières colonnes (si dispo)
            first10 = list(feat_used)[:10]
            print("[DBG] Aperçu (head) des 10 premières features :")
            print(train_df[first10].head(3).to_string(index=False))
    except Exception as e:
        print(f"[DBG] nunique/head indisponible: {e}")

    # 🛑 Garde-fous de stabilité : trop peu de points, pas de features, ou y constant
    if n_train < GP_MIN_TRAIN or p == 0 or y_std < 1e-12:
        reason = []
        if n_train < GP_MIN_TRAIN: reason.append(f"n_train={n_train}<{GP_MIN_TRAIN}")
        if p == 0:                  reason.append("p=0 (aucune feature exploitable)")
        if y_std < 1e-12:           reason.append("std(y)=0 (cible constante)")
        print(f"⚠️  Guard → fallback local/random (pas de fit GP) — " + " | ".join(reason))

        suggestions = _fallback_suggest_local_random(
            X_train_norm=X_train,
            y_train=y_train,
            feat_used=list(feat_used),
            bounds_dict=bounds,
            n_trials=n_trials,
            tr_radius=tr_radius,
            local_share=LOCAL_SHARE,
            seed=None,
        )
        return suggestions

    # ✅ Fit GP seulement si les garde-fous sont passés
    model = fit_gp_model(X_train, y_train)

    # Sécurité mapping : ordre des features vs bornes
    _check_feature_mapping(list(feat_used), bounds)

    # --- Soft-prior multi-objectif : on calcule un score par rangs sur le train
    try:
        W = _score_weights(normalize=True)
        mo_series = compute_rank_score_from_df(
            df=train_df,
            weights=W,
            sharpe_cap=SHARPE_CAP,  # clip de sûreté, cohérent avec le reste
        )
        # Normalisation 0..1 (stable) pour usage direct comme prior
        mo_min, mo_max = float(mo_series.min()), float(mo_series.max())
        denom = (mo_max - mo_min) if (mo_max > mo_min) else 1.0
        mo_norm = ((mo_series - mo_min) / denom).astype(float)
        # Publie dans un 'DB' global pour le re-ranking batch (NN prior)
        globals()["MO_PRIOR_DB"] = {
            "X": X_train,  # normalisé, aligné à feat_used
            "score": torch.tensor(mo_norm.to_numpy(), dtype=torch.double),
        }
        print("[GP] Soft-prior: MO rank score computed on train (published as MO_PRIOR_DB).")
    except Exception as e:
        globals()["MO_PRIOR_DB"] = None
        print(f"[GP][WARN] MO prior unavailable (fallback to center prior): {e}")

    # --- Importance des features pour distances & TR
    feature_names_for_imp = list(feat_used)
    importance_dict = _get_feature_importance_for_gp(model, feature_names_for_imp)
    # On rend dispo pour suggest_candidates via globals() (contrainte de signature actuelle)
    globals()["SHAP_IMPORTANCE_DICT"] = importance_dict

    # Petit résumé (top5) pour visuel
    if importance_dict is not None:
        arr = np.array([importance_dict.get(k, 0.0) for k in feature_names_for_imp], dtype=float)
        order = np.argsort(arr)[::-1][:5]
        top5 = [(feature_names_for_imp[i], float(arr[i])) for i in order]
        print(f"[GP] Feature importance (top5): {top5}")
    else:
        print("[GP] Feature importance: isotropic (no SHAP/ARD).")

    # --- Diagnostics rapides du GP sur le train ---
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = model.posterior(X_train)           # X_train est un Tensor
        mean = post.mean.squeeze(-1)
        var  = post.variance.squeeze(-1)

    yhat = mean.detach().cpu().numpy()
    print("[DBG] var(yhat) =", float(np.var(yhat)))
    print("[DBG] mean(var_post) =", float(var.mean().detach().cpu().item()))

    # Hyperparamètres du kernel / bruit
    try:
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().view(-1).numpy()
        outputscale = model.covar_module.outputscale.detach().cpu().item()
        nv = model.likelihood.noise.detach().cpu().item()
        print(f"[DBG][GP] lengthscales (d={ls.size}):", ls[:10], "...")  # tronque si très long
        print(f"[DBG][GP] outputscale={outputscale:.4g} | noise={nv:.4g}")
    except Exception as e:
        print("[DBG][GP] param dump error:", e)

    # Suggestions (q-batch + TR + anti-clustering)
    suggestions = suggest_candidates(
        model=model,
        X_train_norm=X_train,
        y_train=y_train,
        feat_used=feat_used,
        bounds_dict=bounds,
        n_trials=n_trials,
        q_batch=Q_BATCH,
        local_share=LOCAL_SHARE,
        tr_radius=tr_radius,
        min_l2=MIN_L2_DEFAULT,
        topk_centers=TOPK_CENTERS_DEFAULT,
        max_retries=MAX_RETRIES_DEFAULT,
        seed=None,
        X_evaluated_norm=X_train,
        X_pending_norm=None,
        # Utilise les overrides de la fonction si fournis, sinon laisse None (fallback interne)
        distance_thr=distance_thr,
        q_local_override=q_local_override,
        q_global_override=q_global_override,
        center_radius_schedule=CENTER_RADIUS_SCHEDULE,
        rerank_alpha=RERANK_ALPHA,
        rerank_beta=RERANK_BETA,
    )

    return suggestions

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    """
    Entrée CLI robuste pour le driver GP.
    Supporte la forme historique :
        python gp_driver.py <db_path> <n_trials> <output_json_path> [tr_radius]
    et des flags optionnels (ignorés si absents) :
        --q-local INT        (fallback ENV: GP_Q_LOCAL)
        --q-global INT       (fallback ENV: GP_Q_GLOBAL)
        --distance-thr FLOAT (fallback ENV: GP_DISTANCE_THR)
    """
    import argparse

    parser = argparse.ArgumentParser(description="Gaussian Process suggestion generator")
    parser.add_argument("db_path", type=str, help="Chemin SQLite source (good_iterations / logs)")
    parser.add_argument("n_trials", type=int, help="Nombre de suggestions à produire")
    parser.add_argument("output_json_path", type=str, help="Fichier JSON de sortie")
    parser.add_argument("tr_radius", nargs="?", type=float, default=TR_RADIUS_DEFAULT,
                        help=f"Trust-Region radius (def={TR_RADIUS_DEFAULT})")
    # Overrides optionnels (compatibles avec automl_backtest_driver.py)
    parser.add_argument("--q-local", dest="q_local", type=int, default=None,
                        help="Force nombre de candidats locaux (fallback ENV GP_Q_LOCAL)")
    parser.add_argument("--q-global", dest="q_global", type=int, default=None,
                        help="Force nombre de candidats globaux (fallback ENV GP_Q_GLOBAL)")
    parser.add_argument("--distance-thr", dest="distance_thr", type=float, default=None,
                        help="Seuil distance anti-clustering (fallback ENV GP_DISTANCE_THR)")

    args, _unknown = parser.parse_known_args()

    # Fallback ENV si flags non fournis
    def _env_int(name: str) -> int | None:
        try:
            v = os.getenv(name, "").strip()
            return int(v) if v else None
        except Exception:
            return None

    def _env_float(name: str) -> float | None:
        try:
            v = os.getenv(name, "").strip()
            return float(v) if v else None
        except Exception:
            return None

    q_local_override = args.q_local if args.q_local is not None else _env_int("GP_Q_LOCAL")
    q_global_override = args.q_global if args.q_global is not None else _env_int("GP_Q_GLOBAL")
    distance_thr = args.distance_thr if args.distance_thr is not None else _env_float("GP_DISTANCE_THR")

    # Appel pipeline avec overrides (si fournis)
    suggestions = suggest_with_gp_pipeline(
        db_path=args.db_path,
        n_trials=int(args.n_trials),
        tr_radius=float(args.tr_radius),
        q_local_override=q_local_override,
        q_global_override=q_global_override,
        distance_thr=distance_thr,
    )

    # Écriture JSON atomique
    out_path = args.output_json_path
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(suggestions, f)
    os.replace(tmp_path, out_path)

"""
====================================================================================
Fichier : strategy_params_core.py
Objectif : Définition des structures et paramètres compatibles Numba pour backtest RL
====================================================================================

Description :
Ce module constitue l’infrastructure technique nécessaire à la vectorisation et compilation
just-in-time (JIT) des opérations du moteur de backtest, en particulier dans le contexte 
de l'entraînement ou de l’évaluation d'agents PPO.

Il repose sur l’utilisation intensive de **Numba** pour assurer :
    - des performances équivalentes à C dans les boucles critiques (logique de trading)
    - une compatibilité avec des environnements à haut débit de simulation (AutoML, RL)
    - un typage strict pour un passage fluide des données vers la JIT pipeline

Contenu principal :
    1. Définition d’un `dtype` structuré `log_dtype` pour journaliser les états du moteur
    2. Spécification du `jitclass StrategyParams`, encapsulant tous les hyperparamètres d'une stratégie
    3. Générateur de paramètres `init_params(...)` pour injection aléatoire ou guidée (Optuna/BoTorch)
    4. Mapping bidirectionnel `tp_mode` ⇄ int pour compatibilité JIT
    5. Utilitaires (e.g. `truncated_normal`, `params_to_dict`) pour conversion ou échantillonnage

Contexte d'utilisation :
- Appelé dans la **boucle de décision temps réel** de l’agent RL
- Utilisé pour initialiser des populations de stratégies en **exploration bayésienne**
- Sert de couche d’abstraction Numba pour des modules plus complexes (`backtest_loop`, `env.step()`...)

Cas d’usage typique :
    - `params = init_params(...)` → injection dans le moteur RL ou le backend d’un simulateur Numba
    - `log_array = create_empty_array(n, log_dtype)` → préallocation de mémoire pour logs
    - `params_to_dict(params)` → rendu exportable pour analyse ou logging SQLite

Références conceptuelles :
    - López de Prado, M. *Advances in Financial Machine Learning*, Wiley
    - NumPy structured arrays : vectorisation mémoire des simulations
    - Numba jitclass : alternative compilée aux dataclasses pour traitement intensif

Contraintes :
    - Typage statique strict (e.g. `float32` vs `float64`) requis par `@jitclass`
    - Ne pas modifier l’ordre des champs dans `strategy_spec` sans mise à jour associée du moteur
    - Les chaînes `U256` utilisées pour journaux de raisons de refus sont limitées à usage affichage

Auteur : Moncoucut Brandon
Version : Juin 2025
"""

# === Imports fondamentaux ===
from numba import float32, int32, int8, int64, float64
from numba.experimental import jitclass
import numpy as np
from numba.typed import List
import time  # pour simuler les timestamps UNIX
import pandas as pd
from scipy.stats import truncnorm
from hyperparam_domain import DOMAIN_BOUNDS

# 🎯 Mapping Numba-compatible pour tp_mode
tp_mode_map = {"rr_only": 0, "min_profit_only": 1, "avg_all": 2}


# Schéma des trades (par ligne "open" puis complétée à la clôture)
# Champs ajoutés pour la qualité d'exécution et l'audit:
#   - fee                  : frais totaux (entrée + sortie), même unité que pnl_net
#   - slippage_bps         : slippage estimé (basis points) du trade (moteur bar-based → 0.0 par défaut)
#   - spread_bps_captured  : spread “capturé” (bps, négatif = coût) sur entrée+sortie si disponible
#   - fills                : nombre de fills cumulés pour ce trade
#   - orders_sent          : nombre d’ordres envoyés (incluant OCO)
#   - notional             : notionnel à l’entrée (qty * entry_price)
trade_array_columns = np.dtype([
    # Identité du trade & PnL
    ('num_trade',     np.int64),
    ('balance',       np.float32),
    ('status',        'U64'),
    ('entry_price',   np.float32),
    ('entry_time',    'datetime64[ms]'),
    ('exit_price',    np.float32),
    ('exit_time',     'datetime64[ms]'),
    ('tp',            np.float32),
    ('sl',            np.float32),
    ('quantity',      np.float32),
    ('pnl_net',       np.float32),

    # ====== Qualité d’exécution / audit ======
    ('fee',                 np.float32),   # frais totaux (entrée+sortie), même unité que pnl_net
    ('slippage_bps',        np.float32),   # slippage estimé en bps (si pas de LOB → approx bar-based)
    ('spread_bps_captured', np.float32),   # spread “capturé” (bps, négatif = coût) entrée+sortie
    ('fills',               np.float32),   # nb de fills
    ('orders_sent',         np.float32),   # nb d’ordres envoyés (incluant OCO)
    ('notional',            np.float32),   # notionnel à l’entrée (qty * entry_price)

    # ====== Early stop (facultatif — alignement DB) ======
    ('early_stop_reason',   'U64'),        # raison d’arrêt anticipé (si activé)
    ('early_stop_step',     np.float32),   # pas/étape au moment de l’arrêt anticipé
])

# Schéma des logs par pas de backtest.
# Champs ajoutés (equity-ready) pour permettre le calcul robuste des métriques daily :
#   - balance      : solde courant (base equity)
#   - equity       : alias explicite de balance (mêmes valeurs)
#   - pnl_step     : variation de balance sur le pas courant
#   - drawdown     : drawdown instantané (= balance / peak - 1)  ⇒ <= 0
#   - mkt_quote_vol: proxy volume notionnel de la bougie (close * volume_base)
#   - price        : prix (close) au pas courant (diagnostics)
#   - signal_stability_var / drift : stabilité du score de signal sur fenêtre récente
#   - regime_tag   : tag 'trend' / 'range' / 'vol' (U16)
log_dtype = np.dtype([
    ('timestamp', np.int64),

    # Hyperparamètres
    ('ema_short_period', np.int32),
    ('ema_long_period', np.int32),
    ('rsi_period', np.int32),
    ('rsi_buy_zone', np.int32),
    ('rsi_sell_zone', np.int32),
    ('rsi_past_lookback', np.int32),
    ('atr_tp_multiplier', np.float32),
    ('atr_sl_multiplier', np.float32),
    ('atr_period', np.int32),
    ('macd_signal_period', np.int32),
    ('rsi_thresholds_1m', np.float32),
    ('rsi_thresholds_5m', np.float32),
    ('rsi_thresholds_15m', np.float32),
    ('rsi_thresholds_1h', np.float32),
    ('ewma_period', np.int32),
    ('weight_atr_combined_vol', np.float32),
    ('threshold_volume', np.int32),
    ('hist_volum_period', np.int32),
    ('detect_supp_resist_period', np.int32),
    ('trend_period', np.int32),
    ('threshold_factor', np.float32),
    ('min_qty', np.float32),
    ('step_size', np.float32),
    ('tick_size', np.float32),
    ('min_profit_margin', np.float32),
    ('resistance_buffer_margin', np.float32),
    ('risk_reward_ratio', np.float32),
    ('confidence_score_params', np.float32),
    ('signal_weight_bonus', np.float32),
    ('penalite_resistance_factor', np.float32),
    ('penalite_multi_tf_step', np.float32),
    ('override_score_threshold', np.int32),
    ('rsi_extreme_threshold', np.int32),
    ('signal_pure_threshold', np.float32),
    ('signal_pure_weight', np.float32),

    # Indicateurs et stratégie
    ('atr_15m', np.float32),
    ('avg_atr', np.float32),
    ('required_confirmations', np.int32),
    ('adjusted_rsi_1m', np.float32),
    ('adjusted_rsi_5m', np.float32),
    ('adjusted_rsi_15m', np.float32),
    ('adjusted_rsi_1h', np.float32),
    ('rsi_1m', np.float32),
    ('rsi_1m_threshold', np.float32),
    ('rsi_5m', np.float32),
    ('rsi_5m_threshold', np.float32),
    ('rsi_15m', np.float32),
    ('rsi_15m_threshold', np.float32),
    ('rsi_1h', np.float32),
    ('rsi_1h_threshold', np.float32),
    ('total_confirmations', np.float32),
    ('multi_tf_confirmation', np.int8),
    ('ema_short', np.float32),
    ('ema_long', np.float32),
    ('rsi', np.float32),
    ('macd', np.float32),
    ('macd_signal', np.float32),
    ('atr', np.float32),
    ('volatilite_combinee', np.float32),
    ('volume_signal', np.float32),
    ('support', np.float32),
    ('resistance', np.float32),
    ('vwap', np.float32),
    ('prix_actuel', np.float32),
    ('volume_avg', np.float32),
    ('rsi_past', np.float32),
    ('condition_marche', np.int32),
    ('confidence_score_real', np.float32),
    ('score_signal_pure', np.float32),
    ('signal_ema', np.int8),
    ('signal_rsi', np.int8),
    ('signal_macd', np.int8),
    ('penalite_resistance', np.int8),
    ('penalite_multi_tf', np.int8),
    ('distance_resistance_pct', np.float32),
    ('raison_refus', 'U256'),
    ('buy_reasons', 'U256'),
    ('refus_critique_structurel', np.int8),
    ('refus_critique_technique', np.int8),
    ('nb_signaux_valides', np.int32),
    ('position', np.int32),

    # ====== Nouveaux champs equity/exécution (optionnels) ======
    ('balance',              np.float32),
    ('equity',               np.float32),
    ('pnl_step',             np.float32),
    ('drawdown',             np.float32),
    ('mkt_quote_vol',        np.float32),
    ('price',                np.float32),

    # ====== Nouveaux champs stabilité/régime (optionnels) ======
    ('signal_stability_var',   np.float32),
    ('signal_stability_drift', np.float32),
    ('regime_tag',             'U16'),
])

# log_dtype doit être défini exactement comme dans ton message précédent

def create_empty_array(data_shape, dtype):
    """
    Crée un tableau structuré vide pour le logging du backtest.

    Paramètres :
    - data_shape : tuple ou int, nombre de lignes
    - dtype : dtype structuré NumPy (ex: log_dtype)

    Retourne :
    - log_array : tableau structuré NumPy
    """
    if isinstance(data_shape, tuple):
        n_rows = data_shape[0]
    else:
        n_rows = data_shape

    return np.zeros(n_rows, dtype=dtype)

# ✅ Spécification de StrategyParams (doit correspondre au jitclass)
strategy_spec = [
    ("ema_short_period", int32),
    ("ema_long_period", int32),
    ("rsi_period", int32),
    ("rsi_buy_zone", int32),
    ("rsi_sell_zone", int32),
    ("rsi_past_lookback", int32),
    ("atr_tp_multiplier", float32),
    ("atr_sl_multiplier", float32),
    ("atr_period", int32), 
    ("macd_signal_period", int32),
    ("rsi_thresholds_1m", float32),
    ("rsi_thresholds_5m", float32),
    ("rsi_thresholds_15m", float32),
    ("rsi_thresholds_1h", float32),
    ("ewma_period", int32),
    ("weight_atr_combined_vol", float32),
    ("threshold_volume", int32),
    ("hist_volum_period", int32),
    ("detect_supp_resist_period", int32),
    ("trend_period", int32),
    ("threshold_factor", float32),
    ("min_qty", float32),
    ("step_size", float32),
    ("tick_size", float32),
    ("min_profit_margin", float32),
    ("resistance_buffer_margin", float32),
    ("risk_reward_ratio", float32),
    ("confidence_score_params", float32),
    ("signal_weight_bonus", float32),
    ("penalite_resistance_factor", float32),
    ("penalite_multi_tf_step", float32),
    ("override_score_threshold", int32),
    ("rsi_extreme_threshold", int32),
    ("signal_pure_threshold", float32),
    ("signal_pure_weight", float32),
    ("tp_mode", int8),
    ("enable_early_stop", int8),
    ("mdd_stop_pct", float32),
    ("max_consecutive_losses", int32),
    ("max_steps_no_trade", int32),
]

@jitclass(strategy_spec)
class StrategyParams:
    def __init__(
        self,
        ema_short_period,
        ema_long_period,
        rsi_period,
        rsi_buy_zone,
        rsi_sell_zone,
        rsi_past_lookback,
        atr_tp_multiplier,
        atr_sl_multiplier,
        atr_period,
        macd_signal_period,
        rsi_thresholds_1m,
        rsi_thresholds_5m,
        rsi_thresholds_15m,
        rsi_thresholds_1h,
        ewma_period,
        weight_atr_combined_vol,
        threshold_volume,
        hist_volum_period,
        detect_supp_resist_period,
        trend_period,
        threshold_factor,
        min_qty,
        step_size,
        tick_size,
        min_profit_margin,
        resistance_buffer_margin,
        risk_reward_ratio,
        confidence_score_params,
        signal_weight_bonus,
        penalite_resistance_factor,
        penalite_multi_tf_step,
        override_score_threshold,
        rsi_extreme_threshold,
        signal_pure_threshold,
        signal_pure_weight,
        tp_mode,
        enable_early_stop,
        mdd_stop_pct,
        max_consecutive_losses,
        max_steps_no_trade,
    ):
        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period
        self.rsi_period = rsi_period
        self.rsi_buy_zone = rsi_buy_zone
        self.rsi_sell_zone = rsi_sell_zone
        self.rsi_past_lookback = rsi_past_lookback
        self.atr_tp_multiplier = atr_tp_multiplier
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_period = atr_period
        self.macd_signal_period = macd_signal_period
        self.rsi_thresholds_1m = rsi_thresholds_1m
        self.rsi_thresholds_5m = rsi_thresholds_5m
        self.rsi_thresholds_15m = rsi_thresholds_15m
        self.rsi_thresholds_1h = rsi_thresholds_1h
        self.ewma_period = ewma_period
        self.weight_atr_combined_vol = weight_atr_combined_vol
        self.threshold_volume = threshold_volume
        self.hist_volum_period = hist_volum_period
        self.detect_supp_resist_period = detect_supp_resist_period
        self.trend_period = trend_period
        self.threshold_factor = threshold_factor
        self.min_qty = min_qty
        self.step_size = step_size
        self.tick_size = tick_size
        self.min_profit_margin = min_profit_margin
        self.resistance_buffer_margin = resistance_buffer_margin
        self.risk_reward_ratio = risk_reward_ratio
        self.confidence_score_params = confidence_score_params
        self.signal_weight_bonus = signal_weight_bonus
        self.penalite_resistance_factor = penalite_resistance_factor
        self.penalite_multi_tf_step = penalite_multi_tf_step
        self.override_score_threshold = override_score_threshold
        self.rsi_extreme_threshold = rsi_extreme_threshold
        self.signal_pure_threshold = signal_pure_threshold
        self.signal_pure_weight = signal_pure_weight
        self.tp_mode = tp_mode
        self.enable_early_stop = enable_early_stop
        self.mdd_stop_pct = mdd_stop_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_steps_no_trade = max_steps_no_trade

def truncated_normal(mean, std, lower, upper):
    """Échantillonnage d'une loi normale tronquée"""
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std)

def _normalize_market_constraints(min_qty, step_size, tick_size):
    """Garantit des float32 non nuls pour le jitclass (fallbacks si besoin)."""
    DEFAULTS = (1e-4, 1e-6, 1e-2)
    mq = DEFAULTS[0] if (min_qty is None or (isinstance(min_qty, (int, float)) and min_qty <= 0)) else min_qty
    ss = DEFAULTS[1] if (step_size is None or (isinstance(step_size, (int, float)) and step_size <= 0)) else step_size
    ts = DEFAULTS[2] if (tick_size is None or (isinstance(tick_size, (int, float)) and tick_size <= 0)) else tick_size
    mq, ss, ts = np.float32(mq), np.float32(ss), np.float32(ts)
    if not (mq > 0 and ss > 0 and ts > 0):
        raise ValueError("Market constraints must be positive (after normalization).")
    return mq, ss, ts

# --- Contraintes structurelles (init) -----------------------------------------
def _enforce_constraints_params(d: dict) -> dict:
    """
    Applique des contraintes structurelles AVANT la construction du jitclass.
    - Arrondit les entiers et impose des bornes métier strictes
    - Garantit les invariants (ema_short < ema_long, buy < sell, etc.)
    """

    # 1) EMA: short < long (robuste aux arrondis)
    if "ema_short_period" in d and "ema_long_period" in d:
        s = int(round(d["ema_short_period"]))
        l = int(round(d["ema_long_period"]))
        s = max(1, s)
        l = max(2, l)
        if s >= l:
            # Assure l'ordre strict après arrondi
            l = max(l, s + 1)
            s = max(1, min(s, l - 1))
        d["ema_short_period"] = s
        d["ema_long_period"]  = l

    # 2) RSI global : 0..100 et buy < sell (écart minimum implicite)
    if "rsi_buy_zone" in d and "rsi_sell_zone" in d:
        b = float(d["rsi_buy_zone"])
        s = float(d["rsi_sell_zone"])
        b = max(0.0, min(100.0, b))
        s = max(0.0, min(100.0, s))
        if b >= s:
            # Ecart minimal 10 pts pour éviter la dégénérescence
            mid = 0.5 * (b + s)
            b, s = max(0.0, mid - 10.0), min(100.0, mid + 10.0)
        d["rsi_buy_zone"], d["rsi_sell_zone"] = int(round(b)), int(round(s))

    # 3) RSI seuils par TF : clamp 0..100
    for k in ("rsi_thresholds_1m","rsi_thresholds_5m","rsi_thresholds_15m","rsi_thresholds_1h"):
        if k in d:
            d[k] = float(max(0.0, min(100.0, d[k])))

    # 4) Périodes entières >= 1
    for k in ("atr_period","ewma_period","rsi_period","macd_signal_period",
              "trend_period","hist_volum_period","detect_supp_resist_period"):
        if k in d:
            d[k] = max(1, int(round(d[k])))

    # 5) Multiplicateurs & ratios > 0 (avec clips ATR pro-BO)
    if "atr_tp_multiplier" in d:
        d["atr_tp_multiplier"] = float(d["atr_tp_multiplier"])
        d["atr_tp_multiplier"] = min(max(d["atr_tp_multiplier"], 2.2), 4.3)  # << bornes actives
    if "atr_sl_multiplier" in d:
        d["atr_sl_multiplier"] = float(d["atr_sl_multiplier"])
        d["atr_sl_multiplier"] = min(max(d["atr_sl_multiplier"], 3.0), 5.0)  # << bornes actives
    if "risk_reward_ratio" in d:
        # évite valeurs quasi-nulles
        d["risk_reward_ratio"] = max(0.1, float(d["risk_reward_ratio"]))

    # 6) Poids/coeffs usuels
    if "weight_atr_combined_vol" in d:
        d["weight_atr_combined_vol"] = max(0.0, min(1.0, float(d["weight_atr_combined_vol"])))
    for k in ("signal_weight_bonus","penalite_resistance_factor","penalite_multi_tf_step"):
        if k in d:
            d[k] = float(max(-1e6, min(1e6, d[k])))

    # 7) Seuils & marges (non négatifs, plafond raisonnable)
    if "override_score_threshold" in d:
        d["override_score_threshold"] = max(0, int(round(d["override_score_threshold"])))
    if "rsi_extreme_threshold" in d:
        v = int(round(d["rsi_extreme_threshold"]))
        d["rsi_extreme_threshold"] = max(0, min(100, v))
    for k in ("min_profit_margin","resistance_buffer_margin","threshold_factor","signal_pure_threshold"):
        if k in d:
            v = float(d[k])
            d[k] = max(0.0, min(10.0, v))  # plafond large mais évite les aberrations

    # 8) Volumes (int >= 0)
    if "threshold_volume" in d:
        d["threshold_volume"] = max(0, int(round(d["threshold_volume"])))

    # 9) Early-stop
    if "enable_early_stop" in d:
        d["enable_early_stop"] = 1 if int(round(d["enable_early_stop"])) != 0 else 0
    if "mdd_stop_pct" in d:
        d["mdd_stop_pct"] = float(max(0.0, min(0.99, d["mdd_stop_pct"])))
    if "max_consecutive_losses" in d:
        d["max_consecutive_losses"] = max(1, int(round(d["max_consecutive_losses"])))
    if "max_steps_no_trade" in d:
        d["max_steps_no_trade"] = max(1, int(round(d["max_steps_no_trade"])))

    return d

# -----------------------------------------------------------------------------

# ✅ Fonction de génération Numpy-compatible
def init_params(min_qty, step_size, tick_size, custom_values=None):
    """
    Initialise les hyperparamètres de stratégie pour le backtest.

    Cette fonction peut générer les paramètres de manière :
    - aléatoire (recherche exploratoire)
    - ou déterministe (via injection BoTorch, Optuna, etc.) grâce à l'argument `custom_values`.

    Args:
        min_qty (float): Quantité minimale de la paire (issue de la plateforme)
        step_size (float): Pas de quantité autorisé par la paire
        tick_size (float): Granularité des prix
        custom_values (dict, optional): Hyperparamètres à injecter directement (clé=nom du paramètre)

    Returns:
        StrategyParams: Objet encapsulant tous les hyperparamètres pour le moteur Numba
    """

    # Normalise et sécurise les contraintes marché pour le jitclass
    min_qty, step_size, tick_size = _normalize_market_constraints(min_qty, step_size, tick_size)

    # --- Tirage/Injection basé sur DOMAIN_BOUNDS ------------------------------
    def scale_domain(name: str, is_int: bool = False):
        """
        Génère/injecte la valeur d’un hyperparamètre à partir des bornes 'de design'
        définies dans DOMAIN_BOUNDS[name].

        Règles :
        - Si custom_values[name] est fourni :
            • si 0.0 <= val <= 1.0  → interprété comme valeur **normalisée**,
              on l'interpole dans [lo, hi] (bornes métier).
            • sinon → interprété comme **dénormalisé**, injecté tel quel.
        - Sinon (pas de custom_values) → tirage aléatoire **uniforme** dans [lo, hi].
        - is_int=True → arrondi entier (randint-like).

        Avantages :
        - Une **seule** source de vérité des bornes (hyperparam_domain.DOMAIN_BOUNDS)
        - Plus de divergences entre GP et tirages aléatoires

        Notes :
        - Si `name` est absent de DOMAIN_BOUNDS → on lève un KeyError explicite.
        """
        if name not in DOMAIN_BOUNDS:
            raise KeyError(f"{name} absent de DOMAIN_BOUNDS; ajoute-le ou passe une valeur via custom_values.")

        lo, hi = DOMAIN_BOUNDS[name]

        # Valeur injectée par l'appelant ?
        if custom_values and name in custom_values:
            val = float(custom_values[name])
            # Valeur normalisée ?
            if 0.0 <= val <= 1.0:
                scaled = lo + val * (hi - lo)
            else:
                scaled = val
        else:
            # Tirage aléatoire (uniforme) dans le domaine de design
            scaled = np.random.uniform(lo, hi)

        return int(round(scaled)) if is_int else float(scaled)


    # =====================================================================
    # === 1. Moyennes Mobiles Exponentielles (EMA) ========================
    # ---------------------------------------------------------------------
    # ema_short_period : période de l’EMA “rapide” (réactivité du signal)
    # ema_long_period  : période de l’EMA “lente”  (filtrage du bruit)
    # → Règle structurelle imposée ensuite : ema_short_period < ema_long_period
    # → Domaines : DOMAIN_BOUNDS["ema_short_period"], DOMAIN_BOUNDS["ema_long_period"]
    ema_short_period = scale_domain("ema_short_period", is_int=True)
    ema_long_period  = scale_domain("ema_long_period",  is_int=True)

    # =====================================================================
    # === 2. RSI (Relative Strength Index) ================================
    # ---------------------------------------------------------------------
    # rsi_period        : longueur de fenêtre du RSI (sensibilité/lag)
    # rsi_buy_zone      : zone basse indicative d’un “survente” (buy context)
    # rsi_sell_zone     : zone haute indicative d’un “surachat” (sell context)
    # rsi_past_lookback : barre de “mémoire” pour confirmations RSI passées
    # → Règle structurelle imposée : rsi_buy_zone < rsi_sell_zone (et clamp 0..100)
    # → Domaines : DOMAIN_BOUNDS["rsi_*"]
    rsi_period        = scale_domain("rsi_period", is_int=True)
    rsi_buy_zone      = scale_domain("rsi_buy_zone", is_int=True)
    rsi_sell_zone     = scale_domain("rsi_sell_zone", is_int=True)
    rsi_past_lookback = scale_domain("rsi_past_lookback", is_int=True)

    # =====================================================================
    # === 3. ATR (Average True Range) =====================================
    # ---------------------------------------------------------------------
    # atr_tp_multiplier : multiplicateur d’ATR pour le Take Profit
    # atr_sl_multiplier : multiplicateur d’ATR pour le Stop Loss
    # atr_period        : période de calcul ATR (volatilité de fond)
    # → Domaines : DOMAIN_BOUNDS["atr_*"]
    atr_tp_multiplier = scale_domain("atr_tp_multiplier")
    atr_sl_multiplier = scale_domain("atr_sl_multiplier")
    atr_period        = scale_domain("atr_period", is_int=True)

    # =====================================================================
    # === 4. MACD (Moving Average Convergence Divergence) =================
    # ---------------------------------------------------------------------
    # macd_signal_period : période de l’EMA appliquée à la ligne MACD
    # (les EMA fast/slow du MACD sont induites via ema_short/ema_long)
    # → Domaine : DOMAIN_BOUNDS["macd_signal_period"]
    macd_signal_period = scale_domain("macd_signal_period", is_int=True)

    # =====================================================================
    # === 5. RSI multi-timeframe (confirmations de contexte) ==============
    # ---------------------------------------------------------------------
    # rsi_thresholds_{TF} : seuil RSI (0..100) utilisé pour la TF correspondante
    # (utile pour pondérer/filtrer le score selon la concordance multi-TF)
    # → Domaines : DOMAIN_BOUNDS["rsi_thresholds_*"] (tous 0..100)
    rsi_thresholds_1m  = scale_domain("rsi_thresholds_1m")
    rsi_thresholds_5m  = scale_domain("rsi_thresholds_5m")
    rsi_thresholds_15m = scale_domain("rsi_thresholds_15m")
    rsi_thresholds_1h  = scale_domain("rsi_thresholds_1h")

    # =====================================================================
    # === 6. Volatilité & structure de marché =============================
    # ---------------------------------------------------------------------
    # ewma_period             : période de l’EMA appliquée aux |log-returns|
    # weight_atr_combined_vol : poids de l’ATR% vs l’EMA(|ret|) dans la vol. combinée [0..1]
    # threshold_volume        : seuil minimum de volume (filtre d’activité)
    # hist_volum_period       : fenêtre de lissage du volume
    # detect_supp_resist_period : fenêtre d’analyse S/R (swing points)
    # trend_period            : fenêtre de détection de tendance (slope/EMA)
    # threshold_factor        : facteur générique de seuils (scaling global)
    # → Domaines : DOMAIN_BOUNDS[...]
    ewma_period               = scale_domain("ewma_period", is_int=True)
    weight_atr_combined_vol   = scale_domain("weight_atr_combined_vol")
    threshold_volume          = scale_domain("threshold_volume", is_int=True)
    # option: si tu souhaites un léger adoucissement, conserve la transformation :
    threshold_volume          = max(1, int(round(threshold_volume * 0.5)))
    hist_volum_period         = scale_domain("hist_volum_period", is_int=True)
    detect_supp_resist_period = scale_domain("detect_supp_resist_period", is_int=True)
    trend_period              = scale_domain("trend_period", is_int=True)
    threshold_factor          = scale_domain("threshold_factor")

    # =====================================================================
    # === 7. Money Management & Risk Control ==============================
    # ---------------------------------------------------------------------
    # min_profit_margin        : marge minimale (en unité relative) pour valider un TP
    # resistance_buffer_margin : buffer autour des résistances détectées (réduction risque cassure)
    # risk_reward_ratio        : ratio R:R cible (structure SL/TP)
    # → Domaines : DOMAIN_BOUNDS["min_profit_margin"], etc.
    min_profit_margin        = scale_domain("min_profit_margin")
    resistance_buffer_margin = scale_domain("resistance_buffer_margin")
    risk_reward_ratio        = scale_domain("risk_reward_ratio")

    # =====================================================================
    # === 8. Scores, pénalités et overrides ===============================
    # ---------------------------------------------------------------------
    # confidence_score_params    : intensité du score de confiance par régime
    # signal_weight_bonus        : bonus additif au score (peut être négatif)
    # penalite_resistance_factor : pénalité liée à la proximité des résistances
    # penalite_multi_tf_step     : pénalité si confirmations multi-TF insuffisantes
    # override_score_threshold   : seuil (entier) au-delà duquel on force un “go”
    # rsi_extreme_threshold      : seuil RSI “extrême” (entier) pour overrides
    # signal_pure_threshold      : seuil d’activation du “signal pur”
    # signal_pure_weight         : poids du “signal pur” dans l’agrégation
    # → Domaines : DOMAIN_BOUNDS[...]
    confidence_score_params    = scale_domain("confidence_score_params")
    signal_weight_bonus        = scale_domain("signal_weight_bonus")
    penalite_resistance_factor = scale_domain("penalite_resistance_factor")
    penalite_multi_tf_step     = scale_domain("penalite_multi_tf_step")
    override_score_threshold   = scale_domain("override_score_threshold", is_int=True)
    rsi_extreme_threshold      = scale_domain("rsi_extreme_threshold", is_int=True)
    signal_pure_threshold      = scale_domain("signal_pure_threshold")
    signal_pure_weight         = scale_domain("signal_pure_weight")

    # =====================================================================
    # === 9. Mode Take Profit (TP) — catégoriel ===========================
    # ---------------------------------------------------------------------
    # tp_mode : stratégie d’agrégation du TP
    #  - "rr_only"         : TP basé uniquement sur le risk:reward
    #  - "min_profit_only" : TP sur marge minimale uniquement
    #  - "avg_all"         : moyenne/agrégation des signaux TP
    # (pondération de sampling configurable via x/y/z ci-dessous)
    x = 1; y = 0; z = 0
    tp_mode_str = np.random.choice(["rr_only"] * x + ["min_profit_only"] * y + ["avg_all"] * z)
    tp_mode = tp_mode_map[tp_mode_str]

    # =====================================================================
    # === 10. Early-stop (coupe-coûts) ====================================
    # ---------------------------------------------------------------------
    # enable_early_stop : 1 = on, 0 = off (int8)
    enable_early_stop      = 1
    # mdd_stop_pct : arrêt si drawdown <= -mdd_stop_pct (ex. 0.15 = -15%)
    mdd_stop_pct           = 0.15
    # max_consecutive_losses : arrêt si N pertes d’affilée (ex. 5)
    max_consecutive_losses = 5
    # max_steps_no_trade : arrêt si aucun trade après X pas
    max_steps_no_trade     = 1200


    # === 10. Création finale : appliquer les contraintes puis instancier ===
    params_dict = {
        "ema_short_period": ema_short_period,
        "ema_long_period": ema_long_period,
        "rsi_period": rsi_period,
        "rsi_buy_zone": rsi_buy_zone,
        "rsi_sell_zone": rsi_sell_zone,
        "rsi_past_lookback": rsi_past_lookback,
        "atr_tp_multiplier": atr_tp_multiplier,
        "atr_sl_multiplier": atr_sl_multiplier,
        "atr_period": atr_period,
        "macd_signal_period": macd_signal_period,
        "rsi_thresholds_1m": rsi_thresholds_1m,
        "rsi_thresholds_5m": rsi_thresholds_5m,
        "rsi_thresholds_15m": rsi_thresholds_15m,
        "rsi_thresholds_1h": rsi_thresholds_1h,
        "ewma_period": ewma_period,
        "weight_atr_combined_vol": weight_atr_combined_vol,
        "threshold_volume": threshold_volume,
        "hist_volum_period": hist_volum_period,
        "detect_supp_resist_period": detect_supp_resist_period,
        "trend_period": trend_period,
        "threshold_factor": threshold_factor,
        "min_qty": min_qty,
        "step_size": step_size,
        "tick_size": tick_size,
        "min_profit_margin": min_profit_margin,
        "resistance_buffer_margin": resistance_buffer_margin,
        "risk_reward_ratio": risk_reward_ratio,
        "confidence_score_params": confidence_score_params,
        "signal_weight_bonus": signal_weight_bonus,
        "penalite_resistance_factor": penalite_resistance_factor,
        "penalite_multi_tf_step": penalite_multi_tf_step,
        "override_score_threshold": override_score_threshold,
        "rsi_extreme_threshold": rsi_extreme_threshold,
        "signal_pure_threshold": signal_pure_threshold,
        "signal_pure_weight": signal_pure_weight,
        "tp_mode": tp_mode,
        "enable_early_stop": enable_early_stop,
        "mdd_stop_pct": mdd_stop_pct,
        "max_consecutive_losses": max_consecutive_losses,
        "max_steps_no_trade": max_steps_no_trade
    }

    params_dict = _enforce_constraints_params(params_dict)

    params = StrategyParams(
        params_dict["ema_short_period"],
        params_dict["ema_long_period"],
        params_dict["rsi_period"],
        params_dict["rsi_buy_zone"],
        params_dict["rsi_sell_zone"],
        params_dict["rsi_past_lookback"],
        params_dict["atr_tp_multiplier"],
        params_dict["atr_sl_multiplier"],
        params_dict["atr_period"],
        params_dict["macd_signal_period"],
        params_dict["rsi_thresholds_1m"],
        params_dict["rsi_thresholds_5m"],
        params_dict["rsi_thresholds_15m"],
        params_dict["rsi_thresholds_1h"],
        params_dict["ewma_period"],
        params_dict["weight_atr_combined_vol"],
        params_dict["threshold_volume"],
        params_dict["hist_volum_period"],
        params_dict["detect_supp_resist_period"],
        params_dict["trend_period"],
        params_dict["threshold_factor"],
        params_dict["min_qty"],
        params_dict["step_size"],
        params_dict["tick_size"],
        params_dict["min_profit_margin"],
        params_dict["resistance_buffer_margin"],
        params_dict["risk_reward_ratio"],
        params_dict["confidence_score_params"],
        params_dict["signal_weight_bonus"],
        params_dict["penalite_resistance_factor"],
        params_dict["penalite_multi_tf_step"],
        params_dict["override_score_threshold"],
        params_dict["rsi_extreme_threshold"],
        params_dict["signal_pure_threshold"],
        params_dict["signal_pure_weight"],
        params_dict["tp_mode"],
        params_dict["enable_early_stop"],
        params_dict["mdd_stop_pct"],
        params_dict["max_consecutive_losses"],
        params_dict["max_steps_no_trade"]
    )

    return params

def params_to_dict(params):
    # 🔁 Inversion du mapping {str: int} → {int: str}
    tp_mode_map_reverse = {0: "rr_only", 1: "min_profit_only", 2: "avg_all"}

    return {
        "ema_short_period": params.ema_short_period,
        "ema_long_period": params.ema_long_period,
        "rsi_period": params.rsi_period,
        "rsi_buy_zone": params.rsi_buy_zone,
        "rsi_sell_zone": params.rsi_sell_zone,
        "rsi_past_lookback": params.rsi_past_lookback,
        "atr_tp_multiplier": params.atr_tp_multiplier,
        "atr_sl_multiplier": params.atr_sl_multiplier,
        "atr_period": params.atr_period,
        "macd_signal_period": params.macd_signal_period,
        "rsi_thresholds_1m": params.rsi_thresholds_1m,
        "rsi_thresholds_5m": params.rsi_thresholds_5m,
        "rsi_thresholds_15m": params.rsi_thresholds_15m,
        "rsi_thresholds_1h": params.rsi_thresholds_1h,
        "ewma_period": params.ewma_period,
        "weight_atr_combined_vol": params.weight_atr_combined_vol,
        "threshold_volume": params.threshold_volume,
        "hist_volum_period": params.hist_volum_period,
        "detect_supp_resist_period": params.detect_supp_resist_period,
        "trend_period": params.trend_period,
        "threshold_factor": params.threshold_factor,
        "min_qty": params.min_qty,
        "step_size": params.step_size,
        "tick_size": params.tick_size,
        "min_profit_margin": params.min_profit_margin,
        "resistance_buffer_margin": params.resistance_buffer_margin,
        "risk_reward_ratio": params.risk_reward_ratio,
        "confidence_score_params": params.confidence_score_params,
        "signal_weight_bonus": params.signal_weight_bonus,
        "penalite_resistance_factor": params.penalite_resistance_factor,
        "penalite_multi_tf_step": params.penalite_multi_tf_step,
        "override_score_threshold": params.override_score_threshold,
        "rsi_extreme_threshold": params.rsi_extreme_threshold,
        "signal_pure_threshold": params.signal_pure_threshold,
        "signal_pure_weight": params.signal_pure_weight,
        "tp_mode": tp_mode_map_reverse.get(params.tp_mode, "unknown"),  # decode int → str
        "enable_early_stop": params.enable_early_stop,
        "mdd_stop_pct": params.mdd_stop_pct,
        "max_consecutive_losses": params.max_consecutive_losses,
        "max_steps_no_trade": params.max_steps_no_trade,
    }


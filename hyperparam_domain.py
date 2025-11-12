# hyperparam_domain.py
"""
Bornes 'de design' de l'espace d'hyperparamètres.
Ces bornes sont utilisées pour:
- clampler les suggestions GP (post-traitement)
- tirer des valeurs par défaut cohérentes (optionnel: dans init_params)
NB: Les contraintes structurelles (ema_short < ema_long, buy < sell, etc.)
sont gérées séparément (helpers), ici on ne met que des fourchettes 'raisonnables'.
"""

DOMAIN_BOUNDS = {
    # Core tendances / oscillateurs
    "ema_short_period": (3, 50),
    "ema_long_period":  (10, 200),
    "rsi_period":       (5, 30),
    "rsi_buy_zone":     (10, 45),
    "rsi_sell_zone":    (55, 90),
    "rsi_past_lookback":(5, 50),

    # ATR / SL-TP / MACD / périodes
    "atr_tp_multiplier": (0.5, 5.0),
    "atr_sl_multiplier": (0.5, 5.0),
    "atr_period":        (5, 50),
    "macd_signal_period":(5, 18),

    # Multi-timeframes RSI (si utilisés comme scalaires 0..100)
    "rsi_thresholds_1m":  (0.0, 100.0),
    "rsi_thresholds_5m":  (0.0, 100.0),
    "rsi_thresholds_15m": (0.0, 100.0),
    "rsi_thresholds_1h":  (0.0, 100.0),

    # Volatilité combinée / volume / historiques
    "ewma_period":            (5, 200),
    "weight_atr_combined_vol":(0.0, 1.0),
    "threshold_volume":       (0, 1_000_000),  # adapte selon ton univers/pair
    "hist_volum_period":      (5, 200),

    # Support/Résistance & tendance
    "detect_supp_resist_period": (10, 200),
    "trend_period":              (20, 200),

    # Seuils génériques / marges / ratio
    "threshold_factor":        (0.5, 2.0),
    "min_profit_margin":       (0.0, 1.0),
    "resistance_buffer_margin":(0.0, 0.05),
    "risk_reward_ratio":       (0.5, 5.0),

    # Scores / pénalités / overrides
    "confidence_score_params":   (0.0, 3.0),
    "signal_weight_bonus":       (-5.0, 5.0),
    "penalite_resistance_factor":(0.0, 5.0),
    "penalite_multi_tf_step":    (0.0, 3.0),
    "override_score_threshold":  (0.0, 5.0),
    "rsi_extreme_threshold":     (50, 100),

    # "Signal pur"
    "signal_pure_threshold": (0.0, 3.0),
    "signal_pure_weight":    (0.0, 2.0),
}

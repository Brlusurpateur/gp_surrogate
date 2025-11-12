"""
===============================================================================================
Fichier : scalping_signal_engine.py
Objectif : Implémentation complète de la stratégie de scalping algorithmique (RL-compatible)
===============================================================================================

Résumé fonctionnel :
Ce module représente le **cœur décisionnel** de la stratégie de trading haute fréquence. 
Il traite les données de marché en temps réel et applique des règles hybrides (signal-based + context-based)
afin de produire des décisions d’entrée sur le marché. Il est **intégré dans un environnement de backtest RL** 
et fonctionne avec des structures de données compatibles `NumPy`.

Fonctions principales :
    • `scalping_strategy(...)` : moteur central de décision de prise de position
    • `multi_timeframe_confirmation(...)` : validation RSI à travers plusieurs timeframes
    • `detect_market_trend(...)` : identification de tendance via volatilité relative
    • `calculate_*` : ensemble de fonctions techniques pour produire les indicateurs :
        - `RSI`, `EMA`, `MACD`, `VWAP`, `ATR`, `EWMA`, `Support/Resistance`
    • `adjust_value(...)` : fonction de quantification conforme aux contraintes d’exécution (tick/step)

Logique stratégique :
    ✔️ Score de confiance pondéré selon :
        - Validité des signaux (RSI, EMA, MACD)
        - Contexte de marché (trend vs range)
        - Confirmation multi-timeframe
        - Proximité de zones de résistance
    ✔️ Override intelligent dans certains cas extrêmes (RSI extrême, signaux tous activés)
    ✔️ Journalisation explicite des raisons d’acceptation/rejet à chaque `tick` pour post-analyse

Indicateurs gérés :
    - RSI (standard + logique de repli)
    - EMA courte / longue
    - MACD + signal
    - ATR (volatilité absolue)
    - EWMA-based volatility (réactivité fine)
    - VWAP (flux moyen pondéré, utile pour capter le volume institutionnel)
    - Support / Résistance avec buffer dynamique

Intégration :
    - Appelé à chaque `tick` dans une boucle `for t in range(...)`
    - Interagit avec :
        - `new_order.py` : exécution des ordres OCO
        - `new_trade_loggers.py` : logging structuré
        - `data_handler.py` : gestion des séries temporelles brutes
    - Compatible avec `Numba` ou `Cython` (traitement vectoriel via `np.ndarray`)

Sécurité et robustesse :
    - Pénalités ajustées en cas de signaux ambigus (proximité résistance, incohérence multi-TF)
    - Logging précis pour chaque critère (score, signaux, indicateurs, contextes)
    - Paramétrage fin via objets `StrategyParams` pour tous les seuils critiques

Auteur : Moncoucut Brandon  
Version : Juin 2025
"""

# === Imports fondamentaux ===
import numpy as np
import pandas as pd
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from indicators_core import ema, rsi, macd, atr, vwap_daily, combined_volatility

# === Defaults marché pour BACKTEST (évite None dans Numba) ===
DEFAULT_MARKET_CONSTRAINTS = (1e-4, 1e-6, 1e-2)  # (min_qty, step_size, tick_size)

# Indices pour colonnes dans les matrices plates
TIMESTAMP = 0
OPEN = 1
HIGH = 2
LOW = 3
CLOSE = 4
VOLUME = 5
ATR = 6
EMA_SHORT = 7
EMA_LONG = 8
MACD = 9
MACD_SIGNAL = 10
RSI = 11
VOLUME_AVG = 12
SUPPORT = 13
RESISTANCE = 14
VOLATILITY_COMB = 15
VOLUME_SIGNAL = 16
VWAP = 17


# Fonction pour ajuster une valeur en fonction de la précision

def adjust_value(value, step):
    return round(value // step * step, 8)

# ======================= Dépendances injectables (I/O) ========================
@dataclass(frozen=True)
class SignalDeps:
    """
    Dépendances externes du moteur de signal.
    - get_pair_info(pair) -> (min_qty, step_size, tick_size) ou (None, None, None)
    - play_sound(name=None) -> None
    En backtest strict, ces fonctions sont des no-op (zéro I/O).
    """
    get_pair_info: Callable[[str], Tuple[Optional[float], Optional[float], Optional[float]]]
    play_sound: Callable[[Optional[str]], None]


# ---------- Implémentations BACKTEST (utilise des defaults sûrs) ----------
def _noop_get_pair_info(pair: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Évite les None qui cassent le jitclass Numba
    mq, ss, ts = DEFAULT_MARKET_CONSTRAINTS
    return mq, ss, ts

def _noop_play_sound(name: Optional[str] = None) -> None:
    return

BACKTEST_DEPS = SignalDeps(get_pair_info=_noop_get_pair_info, play_sound=_noop_play_sound)


# ---------- Implémentations LIVE (Binance + son système) ----------
def _live_get_pair_info(pair: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Interroge l'API Binance pour min_qty, step_size, tick_size.
    Lazy import pour ne pas charger en backtest.
    """
    try:
        from config import get_binance_client  # lazy import (live only)
        binance = get_binance_client()
        symbol_info = binance.get_symbol_info(pair)
        min_qty = step_size = tick_size = None
        for f in symbol_info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                min_qty = float(f["minQty"]); step_size = float(f["stepSize"])
            elif f.get("filterType") == "PRICE_FILTER":
                tick_size = float(f["tickSize"])
        return min_qty, step_size, tick_size
    except Exception:
        return None, None, None

def _live_play_sound(name: Optional[str] = None) -> None:
    try:
        sound_file = "/System/Library/Sounds/Glass.aiff"
        subprocess.run(["afplay", sound_file], check=False)
    except Exception:
        pass

LIVE_DEPS = SignalDeps(get_pair_info=_live_get_pair_info, play_sound=_live_play_sound)

# Dépendances par défaut : BACKTEST (zéro I/O)
DEFAULT_DEPS: SignalDeps = BACKTEST_DEPS


# ---------- Facades rétro-compatibles ----------
def get_pair_info(pair: str, deps: Optional[SignalDeps] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Utilise les dépendances injectées (ou par défaut).
    Backtest: no-op ; Live: appelle l’API Binance.
    """
    return (deps or DEFAULT_DEPS).get_pair_info(pair)

def play_sound(name: Optional[str] = None, deps: Optional[SignalDeps] = None) -> None:
    """
    Utilise les dépendances injectées (ou par défaut).
    Backtest: no-op ; Live: joue un son système.
    """
    return (deps or DEFAULT_DEPS).play_sound(name)
# ==============================================================================

def calculate_support_resistance_with_buffer(data, period, buffer_pct):
    """
    Calcule les niveaux de support et de résistance avec un buffer de tolérance.

    Args:
        data (dict): Données de marché avec les clés 'low' et 'high'.
        period (int): Fenêtre pour le calcul du support/résistance (rolling).
        buffer_pct (float): Pourcentage de tolérance autour du support/résistance (ex: 0.002 pour 0.2%).

    Returns:
        tuple: (adjusted_support, adjusted_resistance) — deux np.ndarray
    """
    lows = np.array(data["low"], dtype=float)
    highs = np.array(data["high"], dtype=float)

    # Initialisation avec NaN
    support = np.full_like(lows, np.nan)
    resistance = np.full_like(highs, np.nan)

    # Rolling min/max pour chaque point à partir de `period`
    for i in range(period, len(lows)):
        support[i] = np.min(lows[i - period:i])
        resistance[i] = np.max(highs[i - period:i])

    # Application du buffer
    adjusted_support = support * (1 + buffer_pct)         # on relâche légèrement le support
    adjusted_resistance = resistance * (1 - buffer_pct)   # on serre légèrement la résistance

    return adjusted_support, adjusted_resistance

def detect_support_resistance(data, window):
    """
    Identifie les niveaux de support et de résistance basés sur des points pivots.

    Args:
        data (DataFrame): Données de marché avec 'high' et 'low'.
        window (int): Fenêtre pour l'analyse.

    Returns:
        tuple: Niveaux de support et de résistance.
    """
    resistance = data['high'].rolling(window=window).max().iloc[-1]
    support = data['low'].rolling(window=window).min().iloc[-1]

    return support, resistance

def detect_market_trend(volatility_array, threshold_factor, lookback_period):
    """
    Détermine si le marché est en "Trend" ou en "Range" sur la dernière bougie disponible,
    en analysant la volatilité combinée dans un contexte de données temps réel.

    Args:
        volatility_array (np.ndarray): Tableau en temps réel représentant la volatilité combinée (%).
        threshold_factor (float): Facteur multiplicateur appliqué à la médiane historique de volatilité.
        lookback_period (int): Nombre de bougies à utiliser pour calculer la médiane de volatilité.

    Returns:
        int: 
            - 1 = Marché en tendance (Trend)
            - 0 = Marché en range (latéral)
            - -1 = Indisponible (pas assez de données)
    """
    
    # 🧮 Position de la dernière valeur disponible (temps réel)
    index = len(volatility_array) - 1

    # 🚫 Pas assez d'historique pour analyser la tendance
    if index < lookback_period:
        return -1  # Signal que l'analyse ne peut pas être faite

    # 🔁 Récupère la fenêtre des dernières volatilités
    rolling_window = volatility_array[index - lookback_period:index]

    # 🟰 Calcule la médiane sur la période
    rolling_median_vol = np.median(rolling_window)

    # 🎯 Seuil dynamique : la médiane multipliée par un facteur (ex: 1.2)
    threshold = rolling_median_vol * threshold_factor

    # 📈 Si la volatilité actuelle dépasse le seuil → Trend (1), sinon → Range (0)
    return int(volatility_array[index] >= threshold)

def _compute_signal_series_for_window(
    data: np.ndarray,
    params,
    end_index: int,
    window: int
) -> np.ndarray:
    """
    Construit une série 'signal_strength' binaire/entière sur une fenêtre glissante,
    combinant 4 heuristiques simples:
      - ema_sig   = 1 si EMA_SHORT > EMA_LONG
      - rsi_sig   = 1 si RSI_past < rsi_buy_zone (même logique que dans scalping_strategy)
      - macd_sig  = 1 si MACD > MACD_SIGNAL
      - vol_sig   = 1 si VOLUME_SIGNAL > 0
    Args:
        data: Matrice [T, cols] contenant EMA/RSI/MACD/… (voir indices).
        params: Hyperparamètres avec rsi_buy_zone et rsi_past_lookback.
        end_index: Index courant (inclus) — fin de fenêtre.
        window: Longueur de la fenêtre (nb de pas).
    Returns:
        np.ndarray: shape (W,), valeurs entières 0..4.
    """
    start = max(0, end_index - window + 1)
    if end_index < 0 or start >= data.shape[0]:
        return np.array([], dtype=float)

    # Fenêtres utiles
    ema_s = data[start:end_index + 1, EMA_SHORT]
    ema_l = data[start:end_index + 1, EMA_LONG]
    rsi_v = data[start:end_index + 1, RSI]
    macd_v = data[start:end_index + 1, MACD]
    macd_s = data[start:end_index + 1, MACD_SIGNAL]
    vol_sig = data[start:end_index + 1, VOLUME_SIGNAL]

    # Pour rsi_sig, on compare le RSI "passé" (lookback) au seuil buy_zone.
    # On bâtit une série alignée sur la fenêtre, avec NaN si l'offset est indisponible.
    look = int(max(1, getattr(params, "rsi_past_lookback", 1)))
    rsi_past_series = np.full_like(rsi_v, np.nan, dtype=float)
    if (end_index - start + 1) > look:
        rsi_past_series[look:] = rsi_v[:-look]

    ema_sig = (ema_s > ema_l).astype(int)
    rsi_sig = (rsi_past_series < float(params.rsi_buy_zone)).astype(int)
    macd_sig = (macd_v > macd_s).astype(int)
    vol_b = (vol_sig > 0).astype(int)

    # NaN dans rsi_past → met 0 (pas de signal)
    rsi_sig = np.where(np.isnan(rsi_past_series), 0, rsi_sig)

    return (ema_sig + rsi_sig + macd_sig + vol_b).astype(float)


def _stability_metrics_from_series(x: np.ndarray) -> tuple[float, float]:
    """
    Calcule (variance, drift) d'une série 1D.
    - variance: np.var(x, ddof=1) si len(x)>1 sinon NaN
    - drift: pente (slope) d'une régression linéaire sur index 0..n-1 (np.polyfit)
    Args:
        x: Série 1D.
    Returns:
        (var, slope): floats (NaN si insuffisant).
    """
    n = int(x.shape[0]) if x is not None else 0
    if n <= 1:
        return (np.nan, np.nan)
    var = float(np.var(x, ddof=1))
    try:
        t = np.arange(n, dtype=float)
        slope = float(np.polyfit(t, x.astype(float), 1)[0])
    except Exception:
        slope = np.nan
    return (var, slope)

def compute_signal_stability_score(var: float, drift: float, *, var_scale: float = 10.0, drift_scale: float = 1.0) -> float:
    """
    Calcule un score unique de stabilité du signal (0..100), où un score plus élevé signifie
    un signal plus "stable" (faible variance et faible dérive).

    Paramètres
    ----------
    var : float
        Variance de la série de force de signal (non négative).
    drift : float
        Pente (slope) de la série (dérive). Plus |drift| est faible, plus la stabilité est forte.
    var_scale : float, par défaut 10.0
        Échelle de normalisation de la variance (robuste au choix de fenêtre).
    drift_scale : float, par défaut 1.0
        Échelle de normalisation de la dérive.

    Retour
    ------
    float
        Score dans [0, 100]. 100 = très stable ; 0 = très instable.

    Notes
    -----
    - La formule combine deux facteurs multiplicatifs bornés dans (0,1]:
        S_var   = 1 / (1 + (var / var_scale))
        S_drift = 1 / (1 + (|drift| / drift_scale))
        score   = 100 * S_var * S_drift
    - Ajustez var_scale et drift_scale selon la dispersion typique observée.
    """
    if not (np.isfinite(var) and np.isfinite(drift)):
        return np.nan
    var = max(0.0, float(var))
    s_var = 1.0 / (1.0 + (var / float(var_scale)))
    s_drift = 1.0 / (1.0 + (abs(float(drift)) / float(drift_scale)))
    return float(100.0 * s_var * s_drift)

def _classify_regime_tag(
    vol_current: float,
    vol_window: np.ndarray,
    threshold_factor: float,
    market_condition: int
) -> str:
    """
    Classe le régime en 'vol' (volatilité extrême), 'trend' ou 'range'.
    Règle:
      - 'vol' si vol_current >= 1.5 * median(vol_window) * threshold_factor
      - sinon 'trend' si market_condition==1
      - sinon 'range' si market_condition==0
      - sinon 'unknown'
    """
    if vol_window is None or vol_window.size == 0 or not np.isfinite(vol_current):
        return "unknown"
    med = float(np.median(vol_window))
    thr = med * float(threshold_factor)
    if vol_current >= 1.5 * thr:
        return "vol"
    if market_condition == 1:
        return "trend"
    if market_condition == 0:
        return "range"
    return "unknown"

def multi_timeframe_confirmation(
    data_5m, data_1m, data_15m, data_1h,
    market_condition, log_array,
    rsi_thresholds_1m, rsi_thresholds_5m,
    rsi_thresholds_15m, rsi_thresholds_1h,
    i_log
):
    """
    Confirme une condition multi-timeframe en utilisant les dernières valeurs disponibles.
    Fonction adaptée au flux temps réel, avec enregistrement du log à l'index `i_log`.

    Args:
        data_* (np.ndarray): Données MTF contenant RSI et ATR
        market_condition (int): 1 = Trend, 0 = Range
        log_array (list[dict]): Tableau de logs à remplir à l’index i_log
        rsi_thresholds_* (float): Seuils RSI pour chaque timeframe
        i_log (int): Index dans log_array correspondant à cette itération

    Returns:
        bool: True si confirmation suffisante, False sinon
    """
    index = -1  # Utilise toujours la dernière bougie disponible

    # 🛡️ Vérification de la présence des données
    if any(arr.shape[0] == 0 for arr in [data_1m, data_5m, data_15m, data_1h]):
        print("[⛔ ERREUR] Données manquantes dans au moins un timeframe")
        log_array[i_log]['multi_tf_confirmation'] = -1
        return False

    confirmations = 0

    # --- RSI actuels ---
    rsi_1m = data_1m[index, RSI]
    rsi_5m = data_5m[index, RSI]
    rsi_15m = data_15m[index, RSI]
    rsi_1h = data_1h[index, RSI]

    # --- ATRs ---
    atr_1m = data_1m[index, ATR]
    atr_5m = data_5m[index, ATR]
    atr_15m = data_15m[index, ATR]
    atr_1h = data_1h[index, ATR]

    # --- Moyenne des ATRs ---
    if any(np.isnan(x) for x in [atr_1m, atr_5m, atr_15m, atr_1h]):
        avg_atr = 0.0
    else:
        avg_atr = (atr_1m + atr_5m + atr_15m + atr_1h) / 4.0

    # 🔧 Barème plus permissif :
    # - Si régime "trend" OU atr_15m > 1.1 * avg_atr → 1 confirmation suffit
    # - Sinon → 2 confirmations
    if (market_condition == 1) or (avg_atr > 0 and atr_15m > 1.1 * avg_atr):
        required_confirmations = 1
    else:
        required_confirmations = 2

    # --- Ajustement RSI selon la condition de marché ---
    delta = 0.0
    if market_condition == 1:
        delta = 5.0
    elif market_condition == 0:
        delta = -5.0

    adj_1m = rsi_thresholds_1m + delta
    adj_5m = rsi_thresholds_5m + delta
    adj_15m = rsi_thresholds_15m + delta
    adj_1h = rsi_thresholds_1h + delta

    # --- Log des valeurs ajustées ---
    log_array[i_log]['atr_15m'] = atr_15m
    log_array[i_log]['avg_atr'] = avg_atr
    log_array[i_log]['required_confirmations'] = required_confirmations
    log_array[i_log]['adjusted_rsi_1m'] = adj_1m
    log_array[i_log]['adjusted_rsi_5m'] = adj_5m
    log_array[i_log]['adjusted_rsi_15m'] = adj_15m
    log_array[i_log]['adjusted_rsi_1h'] = adj_1h

    # --- RSI et seuils enregistrés ---
    log_array[i_log]['rsi_1m'] = rsi_1m
    log_array[i_log]['rsi_1m_threshold'] = adj_1m
    log_array[i_log]['rsi_5m'] = rsi_5m
    log_array[i_log]['rsi_5m_threshold'] = adj_5m
    log_array[i_log]['rsi_15m'] = rsi_15m
    log_array[i_log]['rsi_15m_threshold'] = adj_15m
    log_array[i_log]['rsi_1h'] = rsi_1h
    log_array[i_log]['rsi_1h_threshold'] = adj_1h

    # --- Vérification des confirmations ---
    if rsi_1m > adj_1m:
        confirmations += 1
    if rsi_5m > adj_5m:
        confirmations += 1
    if rsi_15m > adj_15m:
        confirmations += 1
    if rsi_1h > adj_1h:
        confirmations += 1

    log_array[i_log]['total_confirmations'] = confirmations
    confirmation_result = confirmations >= required_confirmations
    log_array[i_log]['multi_tf_confirmation'] = int(confirmation_result)

    return confirmation_result












###########################################      Stratégie Fast     ###############################################







def scalping_strategy(data, params, data_slice_1m, data_slice_15m, data_slice_1h, log_array, index, i_log):
    """
    Moteur central de décision (scalping intraday).

    Inputs:
        data (np.ndarray): Matrice 5m (ou TF principal) avec colonnes indexées (OPEN/HIGH/LOW/CLOSE/…).
        params (StrategyParams): Hyperparamètres (seuils RSI, poids signaux, périodes, etc.).
        data_slice_1m / 15m / 1h (np.ndarray): Fenêtres synchrones pour vérifs MTF (RSI/ATR).
        log_array (np.ndarray structuré): Buffer de logs à remplir à l’index `i_log`.
        index (int): Index courant dans `data`.
        i_log (int): Ligne de log correspondante (synchronisée avec la boucle externe).

    Sortie:
        int: Code d’acceptation de trade (0 = rejet, 100/101/102 = acceptances selon logique).

    Notes:
        - Calcule la stabilité du signal (variance/drift) sur une fenêtre récente.
        - Détecte un tag de régime ('trend'/'range'/'vol') cohérent avec detect_market_trend().
        - Journalise toutes les features clés (signaux, scores, pénalités, raisons).
    """
    rejection_reasons = []  # Liste des raisons expliquant un éventuel rejet de trade
    buy_reasons = " "
    accept_raison = 0 # Trade refusé

    # 🔹 Récupération des dernières valeurs de marché
    price = data[index, CLOSE]
    ema_short = data[index, EMA_SHORT]
    ema_long = data[index, EMA_LONG]
    rsi = data[index, RSI]
    macd = data[index, MACD]
    macd_signal = data[index, MACD_SIGNAL]
    atr_val = data[index, ATR]
    vol_comb = data[index, VOLATILITY_COMB]
    volume_signal = data[index, VOLUME_SIGNAL]
    support = data[index, SUPPORT]
    resistance = data[index, RESISTANCE]
    vwap = data[index, VWAP]
    volume_avg = data[index, VOLUME_AVG]
    rsi_past = data[-params.rsi_past_lookback, RSI]  # Valeur RSI à un moment passé (lookback)

    # ===== Signal stability (variance / drift) sur une fenêtre récente =====
    # Fenêtre par défaut: max(20, params.trend_period) bornée par l'index courant
    stability_window = int(max(20, getattr(params, "trend_period", 20)))
    signal_series = _compute_signal_series_for_window(
        data=data,
        params=params,
        end_index=index,
        window=stability_window
    )
    sig_var, sig_drift = _stability_metrics_from_series(signal_series)
    # Score unique 0..100 (plus haut = plus stable)
    sig_stab_score = compute_signal_stability_score(sig_var, sig_drift, var_scale=10.0, drift_scale=1.0)

    # (régime tag calculé après la détection de market_condition — voir plus bas)

    # 🔹 Détection des signaux simples (EMA, RSI, MACD)
    ema_sig = int(ema_short > ema_long)
    rsi_sig = int(rsi_past < params.rsi_buy_zone)
    macd_sig = int(macd > macd_signal)
    valid_signals = ema_sig + rsi_sig + macd_sig  # Somme des signaux activés

    # 🔹 Détection de la condition de marché (range ou trend)
    market_condition = detect_market_trend(data[:, VOLATILITY_COMB], params.threshold_factor, params.trend_period)

    # ===== Tag de régime ('trend'/'range'/'vol') — DOIT utiliser market_condition =====
    vol_current = float(vol_comb)
    start_for_vol = max(0, index - stability_window + 1)
    vol_window = data[start_for_vol:index + 1, VOLATILITY_COMB]
    regime_tag = _classify_regime_tag(
        vol_current=vol_current,
        vol_window=vol_window,
        threshold_factor=float(params.threshold_factor),
        market_condition=int(market_condition)
    )

    # 🔹 Vérifie la cohérence RSI multi-timeframes
    confirmation_multi_tf = multi_timeframe_confirmation(
        data, data_slice_1m, data_slice_15m, data_slice_1h,
        market_condition, log_array,
        params.rsi_thresholds_1m, params.rsi_thresholds_5m,
        params.rsi_thresholds_15m, params.rsi_thresholds_1h, i_log
    )

    # 🔹 Calcul du score "pur" selon les signaux uniquement
    score_signal_pure = valid_signals * params.signal_pure_weight

    # 🔹 Score de confiance : dépend du type de marché
    if market_condition == 0:  # Marché en range
        confidence_score = 50.0 * (rsi_sig + volume_signal)
    else:  # Marché en tendance
        confidence_score = 25.0 * (ema_sig + rsi_sig + macd_sig + volume_signal)

    confidence_score += params.signal_weight_bonus * valid_signals

    # 🔹 Modificateurs de score en fonction de critères supplémentaires
    score_modifier = 1.0

    if rsi > params.rsi_sell_zone:
        score_modifier *= 0.9  # réduction douce du score
        rejection_reasons.append("RSI_élevé_réduction_confiance")

    # 🔹 Si trop proche d'une résistance, on applique une pénalité
    near_resistance = price >= resistance
    distance_resistance_pct = ((resistance - price) / price) * 100
    penalite_resistance = 0

    # 🔹 Vérifie si le prix est trop proche de la résistance
    if near_resistance:
        # Si c'est le cas, on considère que le potentiel de hausse est limité → risque accru
        distance_resistance_pct = 0  # 🔧 Pour le log : on note que la distance est nulle ou trop faible
        score_modifier *= params.penalite_resistance_factor  # ❗ Application d'une pénalité multiplicative (ex: 0.9)
        penalite_resistance = 1  # 📌 Flag activé pour journaliser la pénalité
        rejection_reasons.append(f"Pénalité_résistance")  # Pour le log

    # 🔹 Vérifie si la confirmation multi-timeframe est absente
    penalite_multi_tf = 0
    if confirmation_multi_tf == 0:
        penalite_multi_tf = 1  # ✅ On note qu'il y a un défaut de cohérence entre les timeframes

        # 🧠 Cas exceptionnel : les 3 signaux EMA, RSI et MACD sont valides malgré tout
        if valid_signals == 3:
            # Aucun impact sur le score, la stratégie accepte cette exception
            rejection_reasons.append("Aucune_penalite_multi_TF")

        # ⚖️ Cas intermédiaire : au moins 2 signaux forts + EMA ou MACD bien orienté
        elif valid_signals >= 2 and (ema_sig or macd_sig):
            score_modifier *= 0.96  # 🎯 Très légère pénalité : -4% sur le score
            rejection_reasons.append("Pénalité_multi_TF_allégée")

        # 🚨 Cas critique : peu de signaux valides + pas de confirmation → grosse pénalité
        else:
            # Coefficient de réduction du score selon le nombre de signaux valides
            coef_reduction = 1 - (params.penalite_multi_tf_step * valid_signals)
            # On plafonne la pénalité : on ne descend pas en-dessous de 80% du score initial
            score_modifier *= max(coef_reduction, 0.8)
            rejection_reasons.append(f"Pénalité_multi_TF")  # Log explicite

    confidence_score *= score_modifier  # Score final

    # 🔹 Override possible si tous les signaux + bon score
    override_critical_refus = valid_signals == 3 and confidence_score >= params.override_score_threshold

    # 🔹 Cas 1 : RSI extrême => achat immédiat
    if rsi_sig and rsi_past < params.rsi_extreme_threshold:
        trade_accept = True
        penalite_multi_tf = 0  # annule la pénalité
        buy_reasons = "RSI_extreme"
        accept_raison = 101

    # 🔹 Cas 2 : Score presque bon + bon score pur => accepter borderline
    elif confirmation_multi_tf == 0 and confidence_score >= (params.confidence_score_params - 5) and score_signal_pure >= params.signal_pure_threshold:
        trade_accept = True
        buy_reasons = "Buy_borderline"
        accept_raison = 102

    # 🔹 Cas 3 : Acceptation normale si score suffisant ou override critique
    else:
        trade_accept = confidence_score >= params.confidence_score_params or override_critical_refus
        if trade_accept:
            buy_reasons = "Strong_Buy"
            accept_raison = 100
        else:
            # 🔸 Trade rejeté : score trop bas + aucun autre motif
            if not rejection_reasons:
                rejection_reasons.append("Score_insuffisant")

    # --- 📋 Journalisation dans le log array ---
    log_array[i_log]['ema_short'] = ema_short
    log_array[i_log]['ema_long'] = ema_long
    log_array[i_log]['rsi'] = rsi
    log_array[i_log]['macd'] = macd
    log_array[i_log]['macd_signal'] = macd_signal
    log_array[i_log]['atr'] = atr_val
    log_array[i_log]['volatilite_combinee'] = vol_comb
    log_array[i_log]['volume_signal'] = volume_signal
    log_array[i_log]['support'] = support
    log_array[i_log]['resistance'] = resistance
    log_array[i_log]['vwap'] = vwap
    log_array[i_log]['prix_actuel'] = price
    log_array[i_log]['volume_avg'] = volume_avg
    log_array[i_log]['rsi_past'] = rsi_past
    log_array[i_log]['condition_marche'] = market_condition
    log_array[i_log]['confidence_score_real'] = confidence_score
    log_array[i_log]['score_signal_pure'] = score_signal_pure
    log_array[i_log]['signal_ema'] = ema_sig
    log_array[i_log]['signal_rsi'] = rsi_sig
    log_array[i_log]['signal_macd'] = macd_sig
    log_array[i_log]['penalite_resistance'] = penalite_resistance
    log_array[i_log]['penalite_multi_tf'] = penalite_multi_tf
    log_array[i_log]['distance_resistance_pct'] = distance_resistance_pct
    log_array[i_log]['raison_refus'] = ";".join(rejection_reasons)  # ✅ tronqué à 63 caractères
    log_array[i_log]['refus_critique_structurel'] = int(penalite_multi_tf and not trade_accept)
    log_array[i_log]['refus_critique_technique'] = int(not penalite_multi_tf and not trade_accept)
    log_array[i_log]['nb_signaux_valides'] = valid_signals
    log_array[i_log]['buy_reasons'] = buy_reasons

    # ===== Ecriture conditionnelle des nouveaux champs (si présents dans le dtype) =====
    _names = set(log_array.dtype.names or ())
    if 'signal_stability_var' in _names:
        log_array[i_log]['signal_stability_var'] = float(sig_var) if np.isfinite(sig_var) else np.nan
    if 'signal_stability_drift' in _names:
        log_array[i_log]['signal_stability_drift'] = float(sig_drift) if np.isfinite(sig_drift) else np.nan
    if 'signal_stability_score' in _names:
        log_array[i_log]['signal_stability_score'] = float(sig_stab_score) if np.isfinite(sig_stab_score) else np.nan
    if 'regime_tag' in _names:
        # On tronque proprement si le dtype est limité (ex: 'U16')
        try:
            log_array[i_log]['regime_tag'] = str(regime_tag)[:15]
        except Exception:
            pass

    return accept_raison  # code numérique de la raison d’acceptation
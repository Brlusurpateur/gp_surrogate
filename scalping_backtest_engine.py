"""
=========================================================================================
Fichier : scalping_backtest_engine.py
Objectif : Infrastructure de backtesting vectorisé pour stratégie de scalping multi-timeframe
=========================================================================================

Résumé fonctionnel :
Ce module implémente une boucle de backtest vectorisé haute fréquence, centrée sur la simulation réaliste 
d'une stratégie de scalping algorithmique exploitant plusieurs horizons temporels (`1m`, `5m`, `15m`, `1h`). 
Il assure à la fois l'intégration des indicateurs techniques pré-calculés, la simulation d'exécution d’ordres OCO 
(Take Profit / Stop Loss), et la journalisation des décisions dans des structures NumPy optimisées.

Fonctions principales :
    • `fetch_historical_data(...)` : récupération robuste des données Binance multi-TF
    • `precalculate_indicators(...)` : pré-calcul complet des indicateurs avec vectorisation NumPy
    • `backtest_loop(...)` : boucle principale simulant chaque tick et chaque décision du bot
    • `compute_trade_metrics(...)` : calcul des métriques de performance classiques
    • `backtest_strategy(...)` : pipeline orchestral du backtest avec logging et métriques finales

Logique d’exécution :
    1. Importation des données OHLC multi-timeframe
    2. Calcul des indicateurs : RSI, MACD, EMA, VWAP, ATR, EWMA, support/résistance, etc.
    3. Appel de la fonction `scalping_strategy` à chaque pas de temps (moteur décisionnel)
    4. Simulation de l’exécution OCO (via `handle_trade_execution`)
    5. Enregistrement systématique des logs techniques pour post-analyse
    6. Calcul des métriques : Sharpe Ratio, Drawdown, Profit Factor, Win Rate

Indicateurs utilisés :
    - RSI (avec EMA smoothing)
    - EMA (court/long)
    - MACD & signal
    - ATR
    - VWAP (réinitialisé quotidiennement)
    - Volatilité combinée (ATR% + EWMA)
    - Niveaux de support / résistance dynamiques (avec buffer)
    - Signal de volume conditionnel (breakout + volume relatif)

Métriques de performance :
    - Sharpe Ratio (non annualisé)
    - Maximum Drawdown (via cumul des pertes)
    - Profit Factor (somme gains / pertes)
    - Win Rate (% de trades gagnants)

Intégration :
    - Le moteur de stratégie (`scalping_strategy`) est appelé à chaque tick 5m.
    - Simulation de chaque trade sous forme vectorisée (NumPy) pour compatibilité `Numba`.
    - Résultats renvoyés sous forme de `np.recarray` enrichi avec logs et performance.

Particularités d’ingénierie :
    - Vectorisation stricte des indicateurs pour maximiser la performance CPU
    - `log_array` structuré avec `dtype` personnalisé pour audit rigoureux post-backtest
    - Fonction `precalculate_indicators` centralise le pipeline technique
    - Injection explicite des dépendances critiques (paramètres, taux de fees, risk%)

Hypothèses :
    - Exécution via modèle OCO (Take Profit / Stop Loss / Timeout)
    - Unité de granularité centrale = bougie 5 minutes (base du time loop)
    - L’ordre de priorité temporelle est géré manuellement entre TFs

Auteur : Moncoucut Brandon  
Version : Juin 2025
"""

# === Imports fondamentaux ===
import pandas as pd
import numpy as np
from api_mainnet import connect_wallet
from config import set_binance_client
import time 
from datetime import datetime, timedelta, timezone
from oco_trade_engine import handle_trade_execution
from scipy.ndimage import uniform_filter1d
from strategy_params_core import create_empty_array, log_dtype, trade_array_columns
from backtest_db_manager import array_to_clean_dataframe
from scalping_signal_engine import (
    calculate_support_resistance_with_buffer,
    scalping_strategy,
    TIMESTAMP,
)
from indicators_core import (
    ema,   # EMA sur close
    rsi,   # RSI de Wilder
    atr,   # ATR de Wilder
    vwap_daily,             # VWAP reset quotidien
    combined_volatility,    # ATR% + EMA(|log-returns|)
)



def fetch_historical_data(pair, intervals, lookback_days, timestamp_source: str = "open"):
    """
    Récupère jusqu'à X jours de données historiques de Binance pour plusieurs intervalles,
    en respectant la limite de 1000 bougies par appel.

    Args:
        pair (str): Paire de trading (ex: 'BTCUSDC').
        intervals (list): Liste des intervalles des bougies à récupérer.
        lookback_days (int): Nombre de jours à remonter (ex: 365 pour 1 an).

    Returns:
        dict: Un dictionnaire contenant les DataFrames pour chaque intervalle.
    """

    binance, api_key = connect_wallet()
    set_binance_client(binance)

    if binance is None:
        raise ValueError("❌ Erreur : Impossible d'initialiser le client Binance. Vérifiez vos clés API.")

    data_dict = {}

    for interval in intervals:
        all_data = []
        limit_per_call = 1000
        interval_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360,
            '12h': 720, '1d': 1440
        }.get(interval, 60)

        total_minutes = lookback_days * 1440
        start_time = datetime.now(timezone.utc) - timedelta(minutes=total_minutes)
        now = datetime.now(timezone.utc)

        while start_time < now:
            end_time = start_time + timedelta(minutes=interval_minutes * limit_per_call)
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)

            try:
                candles = binance.get_klines(
                    symbol=pair,
                    interval=interval,
                    startTime=start_ts,
                    endTime=end_ts
                )
            except Exception as e:
                print(f"⚠️ get_klines échoué pour {pair} {interval}: {e}")
                break

            # ✅ Ajoute les bougies récupérées
            if not candles:
                break
            all_data.extend(candles)

            # Avance le curseur temps jusqu'à la dernière bougie incluse
            last_candle_time = pd.to_datetime(candles[-1][0], unit='ms', utc=True).to_pydatetime()
            start_time = last_candle_time + timedelta(minutes=interval_minutes)

            time.sleep(0.2)

        if not all_data:
            continue

        data = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Choix du timestamp utilisé en base
        if timestamp_source == "close" and "close_time" in data.columns:
            data['timestamp'] = pd.to_datetime(data['close_time'], unit='ms', utc=True)
        else:
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

        typical_price = ((data['high'] + data['low'] + data['close']) / 3.0)
        data['vwap'] = vwap_daily(data['timestamp'].values, typical_price.values, data['volume'].values)

        data_dict[interval] = data

    return data_dict

def rolling_mean(array, window):
    return uniform_filter1d(array, size=window, mode='nearest')

def rolling_min(array, window):
    return np.array([np.min(array[max(0, i - window + 1):i + 1]) for i in range(len(array))])

def rolling_max(array, window):
    return np.array([np.max(array[max(0, i - window + 1):i + 1]) for i in range(len(array))])

def precalculate_indicators(data, params):
    """
    Pré-calcule les indicateurs techniques à partir d'un dict[str, np.ndarray].
    Retourne un tableau 2D NumPy : shape = (n_samples, n_features)
    """
    # Conversion sécurisée des timestamps vers le format UNIX en millisecondes
    timestamp = np.array([
        int(pd.to_datetime(ts, utc=True).timestamp() * 1000) if not isinstance(ts, (int, np.integer)) else int(ts)
        for ts in data["timestamp"]
    ], dtype=np.int64)

    # Séries numpy locales
    close  = np.asarray(data["close"],  dtype=np.float64)
    high   = np.asarray(data["high"],   dtype=np.float64)
    low    = np.asarray(data["low"],    dtype=np.float64)
    volume = np.asarray(data["volume"], dtype=np.float64)

    atr_vals   = atr(high, low, close, params.atr_period)
    ema_short  = ema(close, params.ema_short_period)
    ema_long   = ema(close, params.ema_long_period)

    # MACD = EMA_short - EMA_long, signal = EMA(macd_line, signal_period)
    macd_line     = ema_short - ema_long
    macd_signal   = ema(macd_line, params.macd_signal_period)

    rsi_vec    = rsi(close, params.rsi_period)

    volume_avg = rolling_mean(data["volume"], params.hist_volum_period)
    volume_avg = np.nan_to_num(volume_avg, nan=np.nanmean(data["volume"]))  # ou autre fallback

    # Support / Résistance (on conserve ta fonction existante)
    support, resistance = calculate_support_resistance_with_buffer(
        data, params.detect_supp_resist_period, buffer_pct=0.002
    )

    # Volatilité combinée (indicators_core) : close + atr_vals
    volatility_comb = combined_volatility(
        close,
        atr_vals,
        ewma_period=params.ewma_period,
        weight_atr=params.weight_atr_combined_vol
    )

    volume_signal = (
        ((data["close"] > resistance) | (data["close"] < support)) &
        (data["volume"] > volume_avg * params.threshold_volume)
    ).astype(int)

    # VWAP quotidien (indicators_core)
    vwap = vwap_daily(
        pd.to_datetime(data["timestamp"], utc=True).values,
        ((high + low + close) / 3.0),
        volume
    )

    # Stack tous les indicateurs dans une matrice 2D
    indicators_matrix = np.column_stack([
        timestamp,        # 0
        data["open"],     # 1
        data["high"],     # 2
        data["low"],      # 3
        data["close"],    # 4
        data["volume"],   # 5
        atr_vals,         # 6
        ema_short,        # 7
        ema_long,         # 8
        macd_line,        # 9
        macd_signal,      # 10
        rsi_vec,          # 11
        volume_avg,       # 12
        support,          # 13
        resistance,       # 14
        volatility_comb,  # 15
        volume_signal,    # 16
        vwap              # 17
    ])

    return indicators_matrix



def _date_from_epoch_ms(ts_ms: int) -> "datetime.date":
    """
    Convertit un timestamp epoch ms en date UTC (YYYY-MM-DD).
    """
    try:
        return datetime.utcfromtimestamp(int(ts_ms) / 1000.0).date()
    except Exception:
        return None


def _set_eod_snapshot(log_array, idx: int, pnl_day: float, equity_val: float, ts_ms: int) -> None:
    """
    Écrit un snapshot de fin de journée (si les colonnes existent dans log_array.dtype) :
    - is_eod: bool drapeau EOD,
    - equity_eod: equity/solde à la clôture de la journée,
    - pnl_day: PnL cumulé de la journée,
    - date_eod: date sous forme d'entier AAAAMMJJ (pratique pour groupby).
    """
    if idx is None or idx < 0:
        return
    names = set(log_array.dtype.names or ())

    if 'is_eod' in names:
        log_array[idx]['is_eod'] = 1  # bool-like
    if 'equity_eod' in names:
        log_array[idx]['equity_eod'] = float(equity_val)
    if 'pnl_day' in names:
        log_array[idx]['pnl_day'] = float(pnl_day)

    # Écrit la date EOD (AAAA MM JJ) si disponible
    d = _date_from_epoch_ms(ts_ms)
    if d is not None and 'date_eod' in names:
        try:
            ymd = int(f"{d.year:04d}{d.month:02d}{d.day:02d}")
            log_array[idx]['date_eod'] = ymd
        except Exception:
            pass

def backtest_loop(data_5m, data_1m, data_15m, data_1h,
                  params, log_array, trade_history,
                  balance, position,
                  transaction_fee_rate, risk_per_trade,
                  window_size, length, trade_index):
    """
    Boucle principale du backtest (vectorisée).

    Notes "metrics-ready"
    - Journalise, si les champs existent dans `log_array.dtype.names` :
        * 'balance' / 'equity'         : equity instantanée
        * 'pnl_step'                   : variation de balance au pas courant
        * 'drawdown'                   : drawdown instantané (= balance/peak - 1)
        * 'mkt_quote_vol'              : proxy notionnel (close * volume)
        * 'day_id'                     : identifiant journalier AAAAMMJJ (int)
        * 'high_watermark'             : plus-haut d'equity atteint (HWM courant)
        * 'loss_streak'                : compteur de pertes consécutives
        * 'is_eod', 'equity_eod'       : snapshot fin de journée
        * 'pnl_day', 'date_eod'        : PnL cumulé du jour & date EOD AAAAMMJJ
    - Tous ces champs sont optionnels : on teste la présence des colonnes pour rétrocompatibilité.

    Sécurité
    - Itère jusqu'à min(longueurs) pour éviter tout overflow d'indices.
    """

    index = 0

    # Trackers early-stop
    peak_equity = float(balance)
    loss_streak = 0
    steps_since_last_trade = 0
    early_stop_reason = None
    early_stop_step = -1

    # === Trackers journaliers (agrégations pour KPI) ===
    current_day = None
    pnl_day_acc = 0.0

    # ✅ On boucle jusqu'à `min(length, data_5m.shape[0]) - 1`
    #    pour éviter que `i == data_5m.shape[0]` → overflow
    max_len = min(
        length,
        data_5m.shape[0],
        data_1m.shape[0],
        data_15m.shape[0],
        data_1h.shape[0],
        len(log_array)
    )

    assert len(log_array) >= max_len, "log_array est trop court pour supporter toutes les itérations."

    for i in range(window_size, max_len):

        # ✅ Slice les données de 0 jusqu'à i+1 (inclus)
        data_5m_slice = data_5m[:i + 1]
        data_1m_slice = data_1m[:i + 1]
        data_15m_slice = data_15m[:i + 1]
        data_1h_slice = data_1h[:i + 1]

        prev_balance = float(balance)
        prev_index   = trade_index

        # ✅ Logique de stratégie
        last_raison_refus_code = scalping_strategy(
            data_5m_slice, params, data_1m_slice, data_15m_slice, data_1h_slice, log_array, -1, index
        )

        # ✅ Exécution du trade
        trade_history, position, balance, trade_index = handle_trade_execution(
            trade_history, data_5m_slice, balance, position, params,
            transaction_fee_rate, risk_per_trade, last_raison_refus_code, trade_index, index, log_array
        )

        # Détection d’un trade réalisé et mise à jour des compteurs
        if float(balance) != prev_balance:
            # Trade clos → PnL réalisé
            steps_since_last_trade = 0
            if float(balance) < prev_balance:
                loss_streak += 1
            else:
                loss_streak = 0
        else:
            steps_since_last_trade += 1

        # MDD courant (equity = balance)
        if float(balance) > peak_equity:
            peak_equity = float(balance)
        dd = (float(balance) / peak_equity) - 1.0  # dd <= 0

        # Tests early-stop (si activés)
        if getattr(params, "enable_early_stop", 1) != 0:
            if dd <= -float(getattr(params, "mdd_stop_pct", 0.15)):
                early_stop_reason = "mdd_stop"
                early_stop_step = index
                break
            if loss_streak >= int(getattr(params, "max_consecutive_losses", 5)):
                early_stop_reason = "loss_streak_stop"
                early_stop_step = index
                break
            if steps_since_last_trade >= int(getattr(params, "max_steps_no_trade", 300)):
                early_stop_reason = "no_trade_stop"
                early_stop_step = index
                break

        # ✅ Journalisation sécurisée
        last_ts = int(data_5m_slice[-1, TIMESTAMP])
        last_close = float(data_5m_slice[-1, 4])  # colonne 4 = 'close' (cf. precalculate_indicators)
        last_vol   = float(data_5m_slice[-1, 5])  # colonne 5 = 'volume' base
        mkt_quote_vol = last_close * last_vol     # proxy notionnel (quote)

        # Gestion du changement de jour (EOD) AVANT d'écrire la ligne du nouveau jour
        d_curr = _date_from_epoch_ms(last_ts)
        if current_day is None:
            current_day = d_curr
        elif d_curr is not None and current_day is not None and d_curr != current_day:
            # On clôt le jour précédent sur la dernière ligne écrite (index-1)
            if index > 0:
                prev_ts = int(data_5m_slice[-2, TIMESTAMP]) if (data_5m_slice.shape[0] >= 2) else last_ts
                _set_eod_snapshot(log_array, index - 1, pnl_day_acc, float(prev_balance), prev_ts)
            # Reset accumulateur & jour courant
            pnl_day_acc = 0.0
            current_day = d_curr

        # Timestamps & position
        log_array[index]['timestamp'] = last_ts
        log_array[index]['position'] = position

        # Champs optionnels (si présents dans le dtype)
        names = set(log_array.dtype.names or ())
        pnl_step = float(balance) - float(prev_balance)

        # Accumulation PnL jour
        pnl_day_acc += pnl_step

        # Écriture des champs standards
        if 'balance' in names:
            log_array[index]['balance'] = float(balance)
        if 'equity' in names:
            log_array[index]['equity'] = float(balance)
        if 'pnl_step' in names:
            log_array[index]['pnl_step'] = pnl_step
        if 'drawdown' in names:
            log_array[index]['drawdown'] = float(dd)  # dd <= 0
        if 'mkt_quote_vol' in names:
            log_array[index]['mkt_quote_vol'] = float(mkt_quote_vol)
        if 'price' in names:
            log_array[index]['price'] = last_close  # utile pour diagnostics

        # Champs additionnels "metrics-ready"
        if d_curr is not None and 'day_id' in names:
            try:
                log_array[index]['day_id'] = int(f"{d_curr.year:04d}{d_curr.month:02d}{d_curr.day:02d}")
            except Exception:
                pass
        if 'high_watermark' in names:
            log_array[index]['high_watermark'] = float(peak_equity)
        if 'loss_streak' in names:
            log_array[index]['loss_streak'] = int(loss_streak)

        index += 1

    # ✅ Snapshot EOD pour le dernier jour traité (si au moins une ligne écrite)
    if index > 0:
        last_idx = index - 1
        last_ts_logged = int(log_array[last_idx]['timestamp']) if 'timestamp' in (log_array.dtype.names or ()) else 0
        _set_eod_snapshot(log_array, last_idx, pnl_day_acc, float(balance), last_ts_logged)

    return trade_history, log_array, {"early_stop_reason": early_stop_reason, "early_stop_step": early_stop_step}



def backtest_strategy(data, initial_balance, params, transaction_fee_rate, risk_per_trade):
    """
    Pipeline principal de backtest (pré-calculs → boucle → sortie).

    Args:
        data (dict): OHLCV multi-timeframe (np.ndarray ou DataFrame converti) indexés en 1m/5m/15m/1h.
        initial_balance (float): Solde initial du portefeuille.
        params (StrategyParams): Hyperparamètres de la stratégie.
        transaction_fee_rate (float): Taux de frais (ex: 0.001 = 10 bps).
        risk_per_trade (float): Fraction du capital risquée par trade (ex: 0.02 = 2%).

    Returns:
        tuple:
            trade_history (np.ndarray): tableau structuré des trades.
            log_array (np.ndarray): logs pas-à-pas (incluant balance/pnl_step si dtype le prévoit).
            early_stop (dict): raison & step d'arrêt anticipé (si déclenché).
    """

    balance = initial_balance
    position = 0  # 0 signifie aucune position ouverte
    trade_index = 0  # Compteur d’index de trade

    # Détermination de la taille minimale de fenêtre pour commencer à trader
    window_size = max(
        100,
        params.atr_period,
        params.ema_short_period,
        params.ema_long_period,
        params.rsi_period,
        params.detect_supp_resist_period,
        params.hist_volum_period,
        params.trend_period
    )

    # Pré-calcul des indicateurs techniques pour chaque timeframe
    data_5m = precalculate_indicators(data["5m"], params)
    data_1m = precalculate_indicators(data["1m"], params)
    data_15m = precalculate_indicators(data["15m"], params)
    data_1h = precalculate_indicators(data["1h"], params)

    # Taille des données à parcourir
    length = len(data_5m) + 1

    # Préparation des tableaux de logs et d’historique de trades
    log_array = create_empty_array(length + 1, log_dtype)
    trade_history = create_empty_array(100000, trade_array_columns)  # Prévoyance large

    # Initialiser la première ligne de log si les champs existent (equity curve complète)
    names0 = set(log_array.dtype.names or ())
    if 'balance' in names0:
        log_array[0]['balance'] = float(initial_balance)
    if 'equity' in names0:
        log_array[0]['equity'] = float(initial_balance)
    if 'timestamp' in names0:
        # Si les données 5m existent et ont un timestamp, on pose un point initial
        try:
            log_array[0]['timestamp'] = int(precalculate_indicators(data["5m"], params)[0, TIMESTAMP])
        except Exception:
            pass

    # Boucle principale de backtest (logique du bot de trading)
    trade_history, log_array, early_stop = backtest_loop(
        data_5m, data_1m, data_15m, data_1h,
        params, log_array, trade_history,
        balance, position,
        transaction_fee_rate, risk_per_trade,
        window_size, length, trade_index
    )

    df_trades = array_to_clean_dataframe(trade_history)
    if isinstance(df_trades, pd.DataFrame) and not df_trades.empty:
        # On ne calcule plus les KPI ici (source unique = launcher)
        df_trades["early_stop_reason"] = early_stop.get("early_stop_reason")
        df_trades["early_stop_step"]   = early_stop.get("early_stop_step")
        # Reconversion en records pour que le launcher puisse persister directement
        trade_history = df_trades.to_records(index=False)

    return trade_history, log_array, early_stop
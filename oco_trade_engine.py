"""
====================================================================================
Fichier : oco_trade_engine.py
Objectif : Gestion complète du cycle de vie d’un ordre OCO dans un moteur de backtest
====================================================================================

Description :
Ce module regroupe les fonctions critiques permettant de :
    - Calculer dynamiquement la taille de position optimale (selon le risque et la volatilité)
    - Déterminer les niveaux de Take Profit / Stop Loss via des règles adaptatives
    - Gérer l’exécution simulée d’ordres OCO (One-Cancels-the-Other) en environnement backtest
    - Suivre l’évolution des trades ouverts (trigger TP/SL ou timeout)
    - Logger les raisons de rejet (fail soft) ou d’acceptation du trade

Fonctions clés :
    - `calculate_position_size(...)` : sizing dynamique avec contraintes marché (step size, min qty)
    - `adjust_tp_sl_based_on_market(...)` : ajustement adaptatif des niveaux TP/SL via logique conditionnelle
    - `simulate_oco_order(...)` : simulation d’un trade avec insertion dans un tableau structuré NumPy
    - `check_oco_triggered_status(...)` : logique de gestion du trade (TP atteint, SL touché, timeout)
    - `handle_trade_execution(...)` : orchestrateur global de décision d’entrée / sortie

Contexte d’utilisation :
Ce fichier est utilisé dans le moteur principal de simulation de stratégie RL-based
ou rule-based, typiquement lors de la boucle temporelle `for t in range(...)`.

Il est compatible avec :
    - Des tableaux structurés NumPy (typed logs via `trade_array_columns`)
    - Des classes paramétriques compilées Numba (`StrategyParams`)
    - Des architectures de backtest type `SimEngine`, `AutoRLTrader`, etc.

Hypothèses clés :
    - Les données de marché sont pré-formatées en `ndarray` avec colonnes spécifiques
    - Les prix sont arrondis au `tick_size` défini dans les paramètres de marché
    - Le système intègre un contrôle du risque explicite par trade (`risk_per_trade`)

Sécurité intégrée :
    - Rejet automatique si quantité non tradable
    - Refus si les TP/SL sortent des bornes (ex: SL sous support, TP irréaliste)
    - Logging détaillé des raisons de rejet dans le tableau des logs (colonne `raison_refus`)

Références associées :
    - Risk-based sizing (Van Tharp, López de Prado)
    - Adaptive TP/SL mechanics in algorithmic trading (Zura Kakushadze et al.)
    - RL+rule-based hybrid strategy architectures (INRIA, ENSAE, J.P. Morgan AI Labs)

Auteur : Moncoucut Brandon
Version : Juin 2025
"""

# === Imports fondamentaux ===
import pandas as pd
import math
import numpy as np
from scalping_signal_engine import adjust_value

# === Execution helpers =======================================================

def _bps(x: float) -> float:
    """Convertit un ratio en basis points (×10_000)."""
    return float(x) * 10_000.0

def _slippage_bps(side: str, ref_px: float, exec_px: float) -> float:
    """
    Slippage signé en bps, convention 'coût positif':
      - BUY:  (exec - ref)/ref  → coût si exec > ref
      - SELL: (ref - exec)/ref  → coût si exec < ref
    """
    if not (ref_px and exec_px):
        return float("nan")
    if str(side).upper() == "BUY":
        return _bps((exec_px - ref_px) / ref_px)
    else:  # SELL
        return _bps((ref_px - exec_px) / ref_px)

def _spread_bps_captured(side: str, bid: float, ask: float, exec_px: float) -> float:
    """
    'Spread capturé' (bps) vs quotes disponibles:
      - BUY  idéal = ask ; capture = (ask - exec)/mid
      - SELL idéal = bid ; capture = (exec - bid)/mid
    Mid = (bid+ask)/2 ; Négatif → coût (exec pire que quote idéale).
    """
    if not (bid and ask and exec_px):
        return float("nan")
    mid = 0.5 * (float(bid) + float(ask))
    if mid <= 0:
        return float("nan")
    side_u = str(side).upper()
    if side_u == "BUY":
        return _bps((ask - exec_px) / mid)
    else:  # SELL
        return _bps((exec_px - bid) / mid)

def _fee_amount(notional: float, fee_rate: float) -> float:
    """Frais totaux (monnaie quote) = notionnel × fee_rate."""
    if notional is None:
        return 0.0
    return float(notional) * float(fee_rate or 0.0)

def _estimate_half_spread_bps(price: float, tick_size: float, fallback_bps: float = 0.5) -> float:
    """
    Estime le half-spread en points de base (bps) en l'absence d'order book.

    Hypothèse:
        - Proxy deterministe: max( tick_size / price, floor ) exprimé en bps.
        - floor par défaut = 0.5 bps (BTC/USDC spot a souvent < 1 bps sur pairs liquides).

    Args:
        price: Prix transactionnel (entry/exit).
        tick_size: Incrément minimal de prix (params.tick_size).
        fallback_bps: Plancher en bps si tick_size/price trop faible.

    Returns:
        float: half-spread estimé en bps.
    """
    if not np.isfinite(price) or price <= 0.0 or not np.isfinite(tick_size) or tick_size <= 0.0:
        return float(fallback_bps)
    tick_bps = (tick_size / price) * 1e4  # bps
    return float(max(tick_bps, fallback_bps))

def adjust_tp_sl_based_on_market(entry_price, last_volatility, support, resistance, params, last_atr, balance, risk_per_trade, adjusted_quantity):
    """
    Ajuste dynamiquement le Take Profit et Stop Loss en fonction de la volatilité,
    des niveaux de support/résistance, et du capital réellement risqué.
    """

    tick_size = params.tick_size

    # Vérification de sécurité
    if adjusted_quantity <= 0:
        return None, None, "quantite_ajustee_faible"

    # 1. Calcul du stop-loss basé sur risque
    max_loss_allowed = balance * risk_per_trade
    sl_distance = max_loss_allowed / adjusted_quantity
    dynamic_sl = entry_price - sl_distance

    # 2. Sécurité si SL sous le support
    if dynamic_sl < support:
        buffer = 5.0
        scaled_vol = last_volatility * 100.0
        if scaled_vol < 15.0:
            buffer = scaled_vol if scaled_vol > 5.0 else 5.0
        else:
            buffer = 15.0
        dynamic_sl = support - (tick_size * buffer)

    # 3. Calcul des variantes TP
    tp_atr = entry_price + last_atr * params.atr_tp_multiplier
    tp_rr = entry_price + sl_distance * params.risk_reward_ratio
    tp_min_profit = entry_price * (1 + params.min_profit_margin)

    # 4. Détermination du TP selon le mode
    mode = params.tp_mode
    if mode == 0:
        dynamic_tp = tp_rr
    elif mode == 1:
        dynamic_tp = tp_min_profit
    elif mode == 2:
        dynamic_tp = (tp_atr + tp_rr + tp_min_profit) / 3.0
    else:
        raise ValueError(f"❌ Mode TP inconnu: {mode}")

    # 5. Arrondi
    take_profit_price = adjust_value(dynamic_tp, tick_size)
    stop_loss_price = adjust_value(dynamic_sl, tick_size)

    return take_profit_price, stop_loss_price, None


def calculate_position_size(
    portfolio_value: float,
    entry_price: float,
    risk_per_trade: float,
    volatility: float,
    min_qty: float,
    step_size: float
) -> float:
    """
    Calcule une taille de position compatible avec un contrôle de risque "par trade".

    Méthode:
        - Montant risqué = portfolio_value * risk_per_trade (risk_per_trade est une FRACTION, ex: 0.02 = 2%).
        - Distance SL proxy = max(entry_price * (2 * volatility), min_sl_distance).
        - Taille brute = montant_risqué / distance_SL (capée par la quantité "achetable").
        - Quantité ajustée = arrondi au pas de marché (step_size), puis >= min_qty.

    Hypothèses:
        - Ce sizing est "risk-based" et donc invariant d'échelle : multiplier /2 la distance SL double la taille.
        - Dans ce moteur backtest "bar close", le slippage est ignoré (géré ailleurs si nécessaire).

    Args:
        portfolio_value: Solde USDC/USDT disponible.
        entry_price: Prix d'entrée.
        risk_per_trade: Fraction du capital risquée (ex: 0.02 = 2%).
        volatility: Volatilité instantanée (ex: 0.003 = 0.3%).
        min_qty: Lot minimum négociable.
        step_size: Incrément minimal autorisé (quantité).

    Returns:
        float: Quantité ajustée; 0.0 si non tradable.
    """
    # 1) Validations d'entrée
    if not all(map(np.isfinite, [portfolio_value, entry_price, risk_per_trade, volatility, min_qty, step_size])):
        return 0.0
    if portfolio_value <= 0.0 or entry_price <= 0.0 or risk_per_trade <= 0.0:
        return 0.0
    if step_size <= 0.0:
        # par sécurité: si step_size invalide, on ne trade pas
        return 0.0

    # 2) Montant risqué autorisé
    max_risk_amount = portfolio_value * risk_per_trade  # ex: 1000 × 2% = 20

    # 3) Distance SL proxy (USD)
    min_sl_distance = 0.5  # plancher (paramétrable si besoin)
    stop_loss_distance = max(entry_price * (volatility * 2.0), min_sl_distance)

    # 4) Taille brute et contrainte de pouvoir d'achat
    position_size = max_risk_amount / stop_loss_distance
    max_affordable_quantity = portfolio_value / entry_price
    position_size = min(position_size, max_affordable_quantity)

    # 5) Ajustement "marché"
    adjusted_quantity = math.floor(position_size / step_size) * step_size
    if adjusted_quantity < min_qty:
        # non tradable au sens des règles du marché
        return 0.0

    return float(adjusted_quantity)


def check_oco_triggered_status(
    trade_history,
    data,
    transaction_fee_rate: float,
    balance: float,
    last_trade_index: int
):
    """
    Met à jour le dernier trade ouvert : TP, SL ou timeout, puis calcule PnL & fees.

    Politique d'exécution (backtest bar-based) :
        - Si SL touché: exécution à SL (slippage = 0 bps ici).
        - Si TP touché: exécution à TP (slippage = 0 bps ici).
        - Timeout: close de la barre (slippage = 0 bps ici).
      → Ce moteur reste déterministe; un modèle de slippage microstructure peut être
        greffé plus tard si on dispose du LOB.

    Champs optionnels mis à jour si présents dans le dtype `trade_history`:
        - 'fee' : total fees du trade (entrée + sortie).
        - 'slippage_bps' : ici 0.0 par design.
        - 'fills' / 'orders_sent' : incrémentés (1 fill de sortie + 2 ordres OCO initiaux).
        - 'notional' : conservé si défini à l'entrée.

    Args:
        trade_history: Tableau structuré des trades.
        data: Dernière fenêtre OHLCV (numpy).
        transaction_fee_rate: Taux de frais (ex: 0.001).
        balance: Solde courant.
        last_trade_index: Index du dernier trade ouvert.

    Returns:
        tuple: (trade_history, position, balance, next_trade_index)
    """
    # Données du trade
    trade = trade_history[last_trade_index]

    tp = trade['tp']
    sl = trade['sl']
    entry_price = trade['entry_price']
    quantity = trade['quantity']
    entry_time = trade['entry_time']

    # Dernières données de marché
    high = data[-1, 2]   # HIGH
    low = data[-1, 3]    # LOW
    close = data[-1, 4]  # CLOSE
    current_time = np.datetime64(int(data[-1, 0]), 'ms')

    # Logique de déclenchement
    triggered = None
    exit_price = 0.0

    if low <= sl:
        triggered = "loss"
        exit_price = float(sl)
    elif high >= tp:
        triggered = "win"
        exit_price = float(tp)
    elif (current_time - entry_time) > np.timedelta64(4, 'h'):
        triggered = "timeout"
        exit_price = float(close)
    else:
        return trade_history, 1, balance, last_trade_index  # rien à modifier

    # --- PnL & frais ---
    entry_value = float(quantity) * float(entry_price)
    exit_value  = float(quantity) * float(exit_price)
    fees_total  = (entry_value + exit_value) * float(transaction_fee_rate)
    pnl         = exit_value - entry_value - fees_total
    balance     = float(balance) + float(pnl)

    # --- Mise à jour du trade (obligatoire) ---
    trade_history[last_trade_index]['status']     = triggered   # "win" | "loss" | "timeout"
    trade_history[last_trade_index]['exit_price'] = float(exit_price)
    trade_history[last_trade_index]['exit_time']  = current_time
    trade_history[last_trade_index]['pnl_net']    = float(pnl)
    trade_history[last_trade_index]['balance']    = float(balance)

    # --- Champs optionnels si le dtype les expose ---
    names = set(trade_history.dtype.names or ())

    # Frais totaux du trade (entrée + sortie)
    if 'fee' in names:
        trade_history[last_trade_index]['fee'] = float(fees_total)

    # Slippage : 0 bps par design (moteur déterministe bar-based).
    # Si la colonne 'spread_bps_captured' n'existe pas, on ajoute le coût de spread
    # dans slippage_bps pour garder une mesure “coût d'exécution”.
    if 'slippage_bps' in names:
        if 'spread_bps_captured' in names:
            # ici on laisse slippage_bps = 0.0 (le spread est traqué séparément)
            trade_history[last_trade_index]['slippage_bps'] = float(trade_history[last_trade_index]['slippage_bps']) if np.isfinite(trade_history[last_trade_index]['slippage_bps']) else 0.0
        else:
            # sera ajusté ci-dessous avec le half-spread de sortie
            current = float(trade_history[last_trade_index]['slippage_bps']) if np.isfinite(trade_history[last_trade_index]['slippage_bps']) else 0.0
            trade_history[last_trade_index]['slippage_bps'] = current  # init

    # Fills : +1 à la sortie
    if 'fills' in names:
        trade_history[last_trade_index]['fills'] = float(trade_history[last_trade_index]['fills']) + 1.0

    # === Spread “capturé” (approx bar-based) à la SORTIE ===
    # Exécution marketable (TP/SL/timeout) → coût ≈ -half-spread (en bps) à la sortie.
    # Ici on ne dispose pas du tick_size; on applique le fallback (robuste) basé sur le prix.
    half_spread_out_bps = _estimate_half_spread_bps(exit_price, tick_size=np.nan)
    if 'spread_bps_captured' in names:
        current = float(trade_history[last_trade_index]['spread_bps_captured']) if np.isfinite(trade_history[last_trade_index]['spread_bps_captured']) else 0.0
        trade_history[last_trade_index]['spread_bps_captured'] = current - half_spread_out_bps
    elif 'slippage_bps' in names:
        current = float(trade_history[last_trade_index]['slippage_bps']) if np.isfinite(trade_history[last_trade_index]['slippage_bps']) else 0.0
        trade_history[last_trade_index]['slippage_bps'] = current - half_spread_out_bps

    # === Durée du trade (optionnelle) ===
    if 'hold_minutes' in names:
        try:
            hold_min = float((trade_history[last_trade_index]['exit_time'] - trade_history[last_trade_index]['entry_time']).astype('timedelta64[m]').astype(float))
        except Exception:
            hold_min = np.nan
        trade_history[last_trade_index]['hold_minutes'] = hold_min

    # === Raison d'early stop (optionnelle) ===
    if triggered == "timeout":
        if 'early_stop_reason' in names:
            trade_history[last_trade_index]['early_stop_reason'] = "timeout_4h"
        if 'early_stop_step' in names:
            trade_history[last_trade_index]['early_stop_step'] = 0.0  # pas de notion de step ici

    return trade_history, 0, balance, last_trade_index + 1



###########################################      Partie ordre      ###############################################




def handle_trade_execution(
    trade_history,
    data,
    balance: float,
    position: int,
    params,
    transaction_fee_rate: float,
    risk_per_trade: float,
    raison_accept_code: int,
    trade_index: int,
    index: int,
    log_array
):
    """
    Orchestration exécution OCO (entrée/gestion/sortie) au sein de la boucle backtest.

    Convention de codes d'acceptation (externe au module):
        - 100 / 101 / 102 : signaux "buy" de force croissante → autorisent l'ouverture.

    Effets:
        - Si une position est ouverte, vérifie le déclenchement TP/SL/timeout,
          met à jour PnL, fees, balance, et index de trade.
        - Si aucune position et raison OK, tente d'ouvrir via simulate_oco_order(...).
          En cas de rejet, loggue la raison dans log_array[index]['raison_refus'] si dispo.

    Returns:
        tuple: (trade_history, position, balance, trade_index)
    """

    if position == 1:   # Trade open
        # Attention : check_oco_triggered_status() doit aussi être numpy-compatible
        trade_history, position, balance, trade_index = check_oco_triggered_status(
            trade_history, data, transaction_fee_rate, balance, trade_index 
        )

    if position == 0 and raison_accept_code in (100, 101, 102):
        # Simulate_oco_order() aussi doit être numpy-compatible
        trade_history, trade_index, position, rejection_reasons = simulate_oco_order(
            trade_history, data, params, balance, risk_per_trade, -1, trade_index
        )
        if position == 0:
            rejection_reasons = [rejection_reasons]
            log_array[index]['raison_refus'] = ";".join(rejection_reasons)

    
    return trade_history, position, balance, trade_index



def simulate_oco_order(
    trade_array,
    data,
    params,
    balance: float,
    risk_per_trade: float,
    last_idx: int,
    trade_index: int
):
    """
    Ouvre un trade OCO (entrée + 2 ordres conditionnels) et inscrit une ligne "open".

    Champs écrits (obligatoires):
        - num_trade, status="open", entry_price/entry_time, tp/sl, quantity, pnl_net=0, balance (pré-entrée).

    Champs optionnels (si présents dans le dtype):
        - notional: valeur notionnelle à l'entrée (qty * entry_price).
        - orders_sent: +2 (TP et SL OCO) +1 (ordre d'entrée) = 3.
        - fills: 1 (l'entrée est réputée exécutée).
        - slippage_bps: 0.0 (moteur déterministe).
        - fee: non fixé ici (calculé à la sortie pour couvrir entrée+sortie).

    Args:
        trade_array: Tableau structuré des trades (dtype défini dans strategy_params_core).
        data: Données OHLCV (numpy).
        params: Paramètres de stratégie (ck: min_qty, step_size, atr_tp_multiplier, risk_reward_ratio, min_profit_margin, tp_mode, tick_size).
        balance: Solde courant.
        risk_per_trade: Fraction du capital risquée (0.02 = 2%).
        last_idx: Index marché (non utilisé ici, conservé pour compat API).
        trade_index: Position d'écriture dans trade_array.

    Returns:
        tuple: (trade_array, trade_index, position_apres (1=ouvert/0=non), rejection_reason|None)
    """
    # Lecture des dernières données (dernière ligne uniquement)
    entry_price = float(data[-1, 4])      # CLOSE
    last_atr = float(data[-1, 6])         # ATR
    last_volatility = float(data[-1, 15]) # VOLATILITY_COMB
    support = float(data[-1, 13])         # SUPPORT
    resistance = float(data[-1, 14])      # RESISTANCE

    # Conversion du timestamp en datetime64
    entry_time = np.datetime64(int(data[-1, 0]), 'ms')

    # Taille de position (risk-based)
    quantity = calculate_position_size(
        balance, entry_price, risk_per_trade,
        last_volatility, params.min_qty, params.step_size
    )
    if quantity < params.min_qty:
        return trade_array, trade_index, 0, "quantite_ajustee_faible"

    # Niveaux TP/SL
    take_profit, stop_loss, rejection_reason = adjust_tp_sl_based_on_market(
        entry_price, last_volatility, support, resistance,
        params, last_atr, balance, risk_per_trade, quantity
    )
    if rejection_reason is not None:
        return trade_array, trade_index, 0, rejection_reason

    # Écriture de la ligne "open"
    trade_array[trade_index]['num_trade'] = trade_index
    trade_array[trade_index]['balance'] = float(balance)
    trade_array[trade_index]['status'] = "open"
    trade_array[trade_index]['entry_price'] = float(entry_price)
    trade_array[trade_index]['entry_time'] = entry_time
    trade_array[trade_index]['exit_price'] = 0.0
    trade_array[trade_index]['exit_time'] = np.datetime64('NaT')
    trade_array[trade_index]['tp'] = float(take_profit)
    trade_array[trade_index]['sl'] = float(stop_loss)
    trade_array[trade_index]['quantity'] = float(quantity)
    trade_array[trade_index]['pnl_net'] = 0.0

    # Champs optionnels si présents
    names = set(trade_array.dtype.names or ())
    entry_notional = float(quantity) * float(entry_price)

    # Valeur notionnelle à l'entrée
    if 'notional' in names:
        trade_array[trade_index]['notional'] = entry_notional

    # Compteur d'ordres envoyés : 1 (entrée) + 2 (OCO TP/SL)
    if 'orders_sent' in names:
        current = float(trade_array[trade_index]['orders_sent'])
        trade_array[trade_index]['orders_sent'] = current + 3.0

    # Fills : on considère l'entrée exécutée
    if 'fills' in names:
        current = float(trade_array[trade_index]['fills'])
        trade_array[trade_index]['fills'] = current + 1.0

    # Initialisations “propres” à l'ouverture pour éviter des NaN ultérieurs
    if 'fee' in names:
        trade_array[trade_index]['fee'] = 0.0
    if 'slippage_bps' in names:
        # moteur bar-based déterministe → 0.0 par défaut (on ajoute le spread capturé ci-dessous)
        trade_array[trade_index]['slippage_bps'] = float(trade_array[trade_index]['slippage_bps']) if np.isfinite(trade_array[trade_index]['slippage_bps']) else 0.0
    if 'spread_bps_captured' in names:
        trade_array[trade_index]['spread_bps_captured'] = float(trade_array[trade_index]['spread_bps_captured']) if np.isfinite(trade_array[trade_index]['spread_bps_captured']) else 0.0
    if 'early_stop_reason' in names:
        trade_array[trade_index]['early_stop_reason'] = ""  # vide à l'ouverture
    if 'early_stop_step' in names:
        trade_array[trade_index]['early_stop_step'] = np.nan
    if 'hold_minutes' in names:
        trade_array[trade_index]['hold_minutes'] = np.nan  # sera fixé à la sortie

    # === Spread “capturé” (approx bar-based) à l'ENTRÉE ===
    # Exécution marketable → coût ≈ -half-spread (bps) à l'entrée.
    half_spread_in_bps = _estimate_half_spread_bps(entry_price, params.tick_size)
    if 'spread_bps_captured' in names:
        trade_array[trade_index]['spread_bps_captured'] = float(trade_array[trade_index]['spread_bps_captured']) - half_spread_in_bps
    elif 'slippage_bps' in names:
        trade_array[trade_index]['slippage_bps'] = float(trade_array[trade_index]['slippage_bps']) - half_spread_in_bps

    return trade_array, trade_index, 1, None

import time
import json
import hmac
import hashlib
import requests
import pandas as pd
from websocket import WebSocketApp
from datetime import datetime
from api_mainnet import *
from strategy import *
from order import *
from config import set_binance_client
from update_orderbook_features import *
from init_orderbook_features import *
from backtest import *

# Global variables
data = None
pair = None
period = 14
STOP_BOT = False  # Drapeau pour contrôler l'arrêt du bot
binance = None
api_key = None

# ✅ Récupération des données initiales avec gestion des bougies non clôturées
def fetch_data(pair, interval='1m', limit=1000, depth=10):
    """
    Récupère les données de marché à partir de l'API Binance.

    Args:
        pair (str): Le symbole de la paire de trading (ex: 'BTCUSDC').
        interval (str): L'intervalle de temps des bougies (par défaut '1m').
        limit (int): Le nombre de bougies à récupérer (par défaut 1000).

    Returns:
        DataFrame: Données de marché formatées.
    """
    binance = get_binance_client()

    # 📈 Récupération des données de marché OHLCV
    candles = binance.get_klines(symbol=pair, interval=interval, limit=limit)
    data = pd.DataFrame(candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # Conversion des types
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 🔍 Récupération des valeurs initiales du carnet d'ordres
    order_book_data = analyze_order_book(pair, depth, data)

    # ✅ Ajout des nouvelles variables du carnet d'ordres au DataFrame
    for key, value in order_book_data.items():
        data[key] = value

    # ✅ Suppression des lignes incomplètes (NaN) pour garantir la fiabilité des calculs
    missing_data = data.isnull().sum()
    if missing_data.any():
        print(f"⚠️ Données manquantes détectées :\n{missing_data}")
        data = data.dropna()  # Suppression des lignes contenant des NaN
        print(f"✅ Lignes incomplètes supprimées. Nouvelles dimensions : {data.shape}")

    return data

# ✅ Gestion des messages WebSocket
def on_message(ws, message):
    """
    Gère les messages WebSocket reçus et met à jour les données en temps réel.

    Args:
        ws (WebSocketApp): Instance de la connexion WebSocket.
        message (str): Message JSON contenant les données de la bougie.
    """
    global data, period, binance, pair
    msg = json.loads(message)

    if "k" in msg:
        kline = msg["k"]
        
        # ✅ Vérification si la bougie est clôturée (kline["x"] == True)
        if kline["x"]:
            # Utilisation du timestamp de clôture de la bougie
            close_time = pd.to_datetime(kline["T"], unit='ms', utc=True)
            
            # Préparation de la nouvelle bougie
            new_row = {
                "timestamp": close_time,
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"])
            }
            new_df = pd.DataFrame([new_row])
            
            # ✅ Ajout de la bougie clôturée au DataFrame
            if data is not None:
                data = pd.concat([data, new_df], ignore_index=True)
                
            else:
                data = new_df  # Initialisation si le DataFrame est vide
                

            # ✅ Mise à jour des nouvelles variables du carnet d'ordres
            data = update_order_book(pair, 10, data)
            
            # 🚀 Exécution de la stratégie de trading
            decision = scalping_strategy(data, pair)
            
            if decision == 1:
                print("📈 Achat détecté. Placer un ordre d'achat.")
                play_sound()
                # ✅ Détermination dynamique du type d'ordre
                handle_trade_execution(pair, data, portfolio_value)
            else:
                print("⏸️ Aucune action requise.")
            print("\n-----------------\n")

            #check_trade_exit(current_price, take_profit, stop_loss, position, balance, trade_history, entry_price, transaction_fee_rate)
            # ✅ Gestion des anciens ordres
            cancel_old_orders(pair)

# Gestion des erreurs WebSocket
def on_error(ws, error):
    """
    Gère les erreurs survenant lors de la connexion WebSocket.

    Args:
        ws (WebSocketApp): Instance de la connexion WebSocket.
        error (Exception): Détails de l'erreur survenue.
    """
    print(f"❌ Erreur WebSocket : {error}")

# Gestion de la fermeture WebSocket
def on_close(ws, close_status_code, close_msg):
    """
    Gère la fermeture de la connexion WebSocket.

    Args:
        ws (WebSocketApp): Instance de la connexion WebSocket.
        close_status_code (int): Code de fermeture.
        close_msg (str): Message de fermeture.
    """
    global STOP_BOT
    print("🔴 Connexion WebSocket fermée.")
    if not STOP_BOT:
        print("🔁 Tentative de reconnexion dans 5 secondes...")
        time.sleep(5)
        run_bot(fee_rate, portfolio_value, currency_unit, base_url, orders_df)
    else:
        print("🛑 Le bot a été arrêté proprement.")

# Connexion réussie
def on_open(ws):
    """
    Confirme que la connexion WebSocket a été établie avec succès.

    Args:
        ws (WebSocketApp): Instance de la connexion WebSocket.
    """
    print("✅ Connexion WebSocket établie.")

# Lancer le WebSocket
def run_bot(fee_rate, portfolio_value, currency_unit, base_url, orders_df):
    """
    Démarre le bot de trading en initialisant la connexion WebSocket.

    Args:
        fee_rate (float): Taux de frais de trading.
        portfolio_value (float): Valeur totale du portefeuille.
        currency_unit (str): Devise utilisée.
        base_url (str): URL de l'API Binance.
        orders_df (DataFrame): Historique des ordres.
    """
    global data, pair

    # Charger les données initiales pour le calcul des EMA
    data = fetch_data(pair)

    # URL de connexion WebSocket
    stream_url = f"wss://stream.binance.com:9443/ws/{pair.lower()}@kline_1m"
    ws = WebSocketApp(stream_url,
                      on_message=on_message,
                      on_error=on_error,
                      on_close=on_close,
                      on_open=on_open)

    try:
        ws.run_forever()
    except KeyboardInterrupt:
        global STOP_BOT
        STOP_BOT = True
        ws.close()
        print("🛑 Arrêt manuel du bot par l'utilisateur.")
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")
        time.sleep(5)
        if not STOP_BOT:
            run_bot(fee_rate, portfolio_value, currency_unit, base_url, orders_df)

# Initialisation
def init():
    """
    Initialise le bot en configurant les paramètres de trading.
    """
    global pair, fee_rate, portfolio_value, currency_unit, base_url, orders_df, binance, api_key

    binance, api_key = connect_wallet()

    # Définir le client Binance globalement dans strategy.py
    set_binance_client(binance)

    # Vérifier la synchronisation de l'horodatage
    server_time = get_server_time(api_key)
    print(f"🕰️ Heure du serveur Binance : {server_time}")

    pair = "BTCUSDC"
    fee_rate = get_fee_rate()
    portfolio_value, currency_unit = get_portfolio_value(pair)
    base_url = "https://api.binance.com"
    orders_df = pd.DataFrame(columns=[
        "OrderID", "Pair", "Side", "Quantity", "Price", "StopPrice",
        "StopLimitPrice", "Status", "Timestamp", "Gains", "BuyFee", "SellFee",
        "PortfolioValue", "CurrencyUnit"
    ])

    # ✅ Annulation des ordres ouverts avant de démarrer
    cancel_all_open_orders(pair)
    run_bot(fee_rate, portfolio_value, currency_unit, base_url, orders_df)

if __name__ == "__main__":
    init()

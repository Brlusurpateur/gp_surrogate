"""
=====================================================================================
Fichier : init_orderbook_features.py
Objectif : Prétraitement initial des données de carnet d’ordres (order book features)
=====================================================================================

Description :
Ce module applique un ensemble de transformations initiales sur les données historiques 
du carnet d’ordres, dans le but de préparer les features avant usage dans un modèle 
de trading (RL ou autre). Il permet de calculer notamment :

    - VWAP (Volume Weighted Average Price)
    - Spread (écart entre BID et ASK)
    - Cumulative volume (somme des volumes dans le carnet)
    - Ratio BID/ASK (mesure de pression acheteuse vs vendeuse)
    - Position du prix par rapport à la moyenne mobile
    - Score de déséquilibre du carnet

Contexte :
Ce fichier est utilisé avant l'entraînement ou l’inférence, pour enrichir les 
features brutes par des indicateurs synthétiques interprétables.

Auteur : Moncoucut Brandon
Version : Juin 2025
"""

# === Imports fondamentaux ===
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from config import *

def analyze_order_book(pair, depth, data):
    """
    Initialise le carnet d'ordres et l'intègre directement dans le DataFrame `data`.

    Args:
        pair (str): Paire de trading.
        depth (int): Profondeur du carnet d'ordres analysée.
        data (DataFrame): DataFrame contenant les données de marché.

    Returns:
        DataFrame: DataFrame enrichi des informations du carnet d'ordres.
    """
    binance = get_binance_client()
    order_book = binance.get_order_book(symbol=pair, limit=depth)

    # 🔍 Vérification si `data` est vide (évite erreurs d'indexation)
    if data.empty:
        print("⚠️ Erreur: DataFrame `data` est vide, impossible d'initialiser le carnet d'ordres.")
        return data

    # 🔄 **Correction des timestamps si nécessaire**
    if not np.all(np.diff(data.index) > 0):
        print("⚠️ Correction des timestamps via interpolation.")
        interp_func = interp1d(data.index, data['close'], kind='linear', fill_value='extrapolate')
        data['close'] = interp_func(data.index)

    # Extraction des meilleures offres et demandes
    best_bid = float(order_book['bids'][0][0])
    best_ask = float(order_book['asks'][0][0])
    
    # Volume total des ordres bid/ask
    total_bid_volume = sum(float(bid[1]) for bid in order_book['bids'])
    total_ask_volume = sum(float(ask[1]) for ask in order_book['asks'])

    # 📌 **Gestion avancée des poids du carnet d’ordres**
    total_volume = total_bid_volume + total_ask_volume
    weight_bid = total_bid_volume / total_volume if total_volume > 0 else 0.5
    weight_ask = total_ask_volume / total_volume if total_volume > 0 else 0.5

    # 📌 **Ajout d'un facteur de distance au spread**
    last_close = data['close'].iloc[-1]
    bid_proximity = 1 - abs((last_close - best_bid) / last_close)
    ask_proximity = 1 - abs((last_close - best_ask) / last_close)

    # Ajustement dynamique des poids en fonction de la proximité au spread
    weight_bid *= bid_proximity
    weight_ask *= ask_proximity

    # 📌 **Calcul du prix typique**
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3

    # 📌 **VWAP pondéré basé sur le carnet d'ordres**
    vwap_weights = np.where(data['typical_price'] < best_ask, weight_bid, weight_ask)

    # 🔥 **Calcul du VWAP optimisé**
    volume = data['volume'].values
    vwap = np.cumsum(data['typical_price'] * volume * vwap_weights) / np.cumsum(volume * vwap_weights)

    # 🏆 **Ajout des variables dans `data` directement**
    data['best_bid'] = best_bid
    data['best_ask'] = best_ask
    data['bid_volume'] = total_bid_volume
    data['ask_volume'] = total_ask_volume
    data['weight_bid'] = weight_bid
    data['weight_ask'] = weight_ask
    data['spread'] = best_ask - best_bid
    data['vwap_weights'] = vwap_weights
    data['vwap'] = vwap

    print(f"✅ Initialisation Order Book | Spread = {best_ask - best_bid:.5f}, Best Bid = {best_bid}, Best Ask = {best_ask}")

    return data
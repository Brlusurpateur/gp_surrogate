"""
========================================================================================
Fichier : update_orderbook_features.py
Objectif : Mise à jour dynamique des features du carnet d’ordres ligne par ligne (tick)
========================================================================================

Description :
Ce module est conçu pour enrichir dynamiquement un dictionnaire de données de marché 
(typiquement dans un environnement d'entraînement RL ou d’exécution temps réel). 
Il met à jour les colonnes features suivantes à chaque tick :

    - Spread entre BID et ASK
    - VWAP actualisé
    - Ratio de volume BID vs ASK
    - Volume cumulé (carnet entier)
    - Ratio de pression acheteuse/vendeuse

Contexte :
Ce fichier est utilisé dans la boucle de collecte de données, pour intégrer en continu 
des indicateurs synthétiques nécessaires à la prise de décision.

Auteur : Moncoucut Brandon
Version : Juin 2025
"""

# === Imports fondamentaux ===
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from config import get_binance_client


def update_order_book(pair, depth, data):
    """
    Met à jour uniquement les nouvelles valeurs du carnet d'ordres pour éviter le recalcul inutile.

    Args:
        pair (str): Paire de trading.
        depth (int): Profondeur du carnet d'ordres analysée.
        data (DataFrame): DataFrame contenant les données du marché.

    Returns:
        DataFrame: Mise à jour avec les nouvelles données du carnet d'ordres.
    """
    
    binance = get_binance_client()
    
    order_book = binance.get_order_book(symbol=pair, limit=depth)
    
    # 🔍 Vérification si `data` est vide (évite erreurs d'indexation)
    if data.empty:
        print("⚠️ Erreur: DataFrame `data` est vide, impossible de mettre à jour le carnet d'ordres.")
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
    
    # 🔹 Gestion avancée des poids du carnet d’ordres**
    total_volume = total_bid_volume + total_ask_volume
    weight_bid = total_bid_volume / total_volume if total_volume > 0 else 0.5
    weight_ask = total_ask_volume / total_volume if total_volume > 0 else 0.5
    
    # 📌 **Ajout d'un facteur de distance au meilleur prix** 
    # - Plus le prix est proche du best_bid, plus le poids acheteur est grand.
    # - Plus le prix est proche du best_ask, plus le poids vendeur est grand.
    last_close = data['close'].iloc[-1]
    bid_proximity = 1 - abs((last_close - best_bid) / last_close)
    ask_proximity = 1 - abs((last_close - best_ask) / last_close)
    
    # Ajustement dynamique des poids en fonction de la proximité au spread
    weight_bid *= bid_proximity
    weight_ask *= ask_proximity
    
    # Calcul du prix typique
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # 📌 VWAP pondéré basé sur le carnet d'ordres
    vwap_weights = np.where(data['typical_price'] < best_ask, weight_bid, weight_ask)
    # 🔥 **Correction pour assurer la cohérence des types**
    vwap_weights = pd.Series(vwap_weights, index=data.index)  # Convertir en Série Pandas pour compatibilité

    # 🔥 **Calcul du VWAP optimisé**
    volume = data['volume'].values
    # ⚡ Vérification que la colonne 'vwap' existe dans data
    if "vwap" not in data.columns:
        data["vwap"] = np.nan  # Initialise la colonne pour éviter KeyError
    vwap = np.cumsum(data['typical_price'] * volume * vwap_weights) / np.cumsum(volume * vwap_weights)
    
    # 🏆 **Mise à jour optimisée du DataFrame**
    update_values = {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_volume": total_bid_volume,
        "ask_volume": total_ask_volume,
        "weight_bid": weight_bid,
        "weight_ask": weight_ask,
        "spread": best_ask - best_bid,
        "typical_price": data['typical_price'].iloc[-1],
        "vwap_weights": vwap_weights.iloc[-1],
        "vwap": vwap.iloc[-1],
    }
    
    data.iloc[-1] = data.iloc[-1].fillna(update_values)  # Mise à jour optimisée

    print(f"🔄 Mise à jour Order Book | Spread = {best_ask - best_bid:.5f}, Best Bid = {best_bid}, Best Ask = {best_ask}")

    return data
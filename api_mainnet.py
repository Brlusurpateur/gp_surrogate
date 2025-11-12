"""
Binance Spot API helper (mainnet/testnet switch) with robust I/O.

- Loads credentials from API.env first, then .env (override-friendly).
- Exposes a connection helper that syncs server time and optionally points to Spot Testnet.
- Provides small HTTP wrappers (price/time) with retry/backoff and clear typing.

This module is environment-agnostic: set USE_TESTNET via argument or env var.
"""

import os
import time
import json
import requests
from datetime import datetime
from typing import Optional, Tuple

from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Clés API Binance + switch testnet
api_key = None
secret_key = None
USE_TESTNET = False
BASE_URL = "https://api.binance.com"  # mainnet par défaut

# === Configuration & Helpers ===

# Endpoints REST Spot
MAINNET_API_BASE = "https://api.binance.com"
TESTNET_API_BASE = "https://testnet.binance.vision"

def _env_bool(name: str, default: bool = False) -> bool:
    """
    Lit un booléen depuis l'environnement.
    Accepte: "1", "true", "yes", "on" (case-insensitive) pour True.
    """
    v = os.getenv(name, "").strip().lower()
    if not v:
        return bool(default)
    return v in {"1", "true", "yes", "on", "y", "t"}

def _load_env_chain() -> None:
    """
    Charge d'abord API.env (local aux secrets), puis .env (racine, optionnel).
    Idempotent: peut être rappelé plusieurs fois sans effet de bord.
    """
    # charge explicitement API.env si présent
    if os.path.exists("API.env"):
        load_dotenv("API.env")
    # charge .env si présent
    load_dotenv()

def _get_credentials() -> Tuple[str, str]:
    """
    Retourne (api_key, api_secret) depuis l'environnement.
    Cherche plusieurs alias pour compatibilité.
    """
    ak = (
        os.getenv("BINANCE_API_KEY")
        or os.getenv("API_KEY_BINANCE")
        or os.getenv("BINANCE_KEY")
        or ""
    ).strip()
    sk = (
        os.getenv("BINANCE_API_SECRET")
        or os.getenv("API_SECRET_BINANCE")
        or os.getenv("BINANCE_SECRET")
        or ""
    ).strip()
    if not ak or not sk:
        raise RuntimeError("Clés API Binance introuvables dans l'environnement (BINANCE_API_KEY / BINANCE_API_SECRET).")
    return ak, sk

def _get_api_base(testnet: Optional[bool] = None) -> str:
    """
    Retourne la base URL REST selon testnet/mainnet.
    - Si testnet is None, lit TESTNET dans l'environnement.
    """
    if testnet is None:
        testnet = _env_bool("TESTNET", False)
    return TESTNET_API_BASE if testnet else MAINNET_API_BASE

def get_binance_client(testnet: Optional[bool] = None) -> Client:
    """
    Construit un Client python-binance correctement configuré (Spot).

    Notes
    -----
    - Tolère l'absence de clés API pour les endpoints publics (ping, time, ticker/price).
    - Force client.API_URL pour supporter proprement le Spot Testnet.
    """
    _load_env_chain()
    base_url = _get_api_base(testnet)

    # ✅ Mode tolérant : on essaie de charger les clés, sinon on passe en "public-only"
    try:
        api_key, api_secret = _get_credentials()
    except RuntimeError:
        api_key, api_secret = None, None  # public-only (endpoints non signés)

    # python-binance accepte None/"" pour les clefs sur les endpoints publics
    client = Client(api_key or "", api_secret or "")
    client.API_URL = f"{base_url}/api"   # aligne la base REST mainnet/testnet
    return client

def _http_get(url: str, headers: dict, params: Optional[dict] = None, retries: int = 3, backoff: float = 0.5) -> requests.Response:
    """
    GET HTTP robuste avec retry/backoff simple.

    Args
    ----
    url : str
        URL complète.
    headers : dict
        En-têtes HTTP (incluant X-MBX-APIKEY si nécessaire).
    params : Optional[dict]
        Paramètres de requête.
    retries : int
        Nombre d'essais (>=1).
    backoff : float
        Pause (secondes) entre essais, croissance linéaire.

    Returns
    -------
    requests.Response
        Réponse brute (caller valide le status_code et parse).
    """
    attempt = 0
    last_exc = None
    while attempt < max(1, retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            return resp
        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff * (attempt + 1))
            attempt += 1
    # Si on sort de la boucle, on re-lève la dernière exception
    raise last_exc if last_exc else RuntimeError("HTTP GET failed without exception")

def connect_wallet(testnet: Optional[bool] = None) -> Client:
    """
    Initialise un client Binance Spot (mainnet ou testnet) à partir des secrets
    chargés via dotenv, et vérifie la connectivité avec /api/v3/time.

    Args
    ----
    testnet : Optional[bool]
        True pour Testnet, False pour Mainnet, None => lit l'env TESTNET.

    Returns
    -------
    Client
        Instance python-binance configurée (client.API_URL fixé).

    Raises
    ------
    RuntimeError
        Si la connectivité échoue (les clés ne sont pas nécessaires pour /time).
    """
    client = get_binance_client(testnet=testnet)

    # Vérifie la connectivité (server time)
    server_dt = get_server_time(client)
    if not server_dt:
        raise RuntimeError("Impossible de récupérer l'heure serveur Binance — vérifie les clés/réseau.")
    return client

def get_current_price(pair: str, api_key: Optional[str]) -> Optional[float]:
    """
    Récupère le dernier prix Spot pour `pair` (ex. "BTCUSDC"), côté mainnet/testnet
    selon la configuration courante.

    Notes
    -----
    - Endpoint public: l’entête X-MBX-APIKEY n’est pas requis, mais toléré.
    - Utilise un retry/backoff léger pour résilience réseau.

    Returns
    -------
    Optional[float]
        Prix flottant si succès, None sinon.
    """
    global BASE_URL
    api_call = "/api/v3/ticker/price"
    headers = {'content-type': 'application/json'}
    if api_key:
        headers['X-MBX-APIKEY'] = api_key
    params = {"symbol": pair}

    try:
        response = _http_get(BASE_URL + api_call, headers=headers, params=params)
    except Exception as e:
        print(f"❌ Erreur réseau lors de la récupération du prix: {type(e).__name__}: {e}")
        return None

    if response.status_code == 200:
        try:
            price_data = response.json()
            return float(price_data['price'])
        except Exception as e:
            print(f"❌ Parsing JSON prix échoué: {e}")
            return None
    else:
        print(f"❌ Erreur HTTP {response.status_code} prix: {response.text}")
        return None

def get_server_time(client: Client) -> Optional[datetime]:
    """
    Récupère l'heure serveur depuis l'endpoint /api/v3/time en respectant le
    switch testnet/mainnet via client.API_URL.

    Returns
    -------
    datetime | None
        Heure serveur UTC si succès, sinon None.
    """
    api_base = getattr(client, "API_URL", f"{_get_api_base()}/api")
    url = f"{api_base}/v3/time"

    headers = {'content-type': 'application/json'}
    # X-MBX-APIKEY n'est pas nécessaire pour /time, mais on l'ajoute si dispo
    try:
        if client.API_KEY:
            headers['X-MBX-APIKEY'] = client.API_KEY
    except Exception:
        pass

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        return datetime.fromtimestamp(payload['serverTime'] / 1000.0)
    except Exception as e:
        print(f"❌ Erreur get_server_time: {e}")
        return None

def ping(client: Client) -> bool:
    """
    Teste /api/v3/ping sur la base URL du client.
    """
    api_base = getattr(client, "API_URL", f"{_get_api_base()}/api")
    url = f"{api_base}/v3/ping"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return True
    except Exception:
        return False

def get_symbol_price_ticker(client: Client, symbol: str) -> Optional[float]:
    """
    Retourne le dernier prix (ticker) pour `symbol` via /api/v3/ticker/price.
    Compatible testnet/mainnet selon client.API_URL.
    """
    api_base = getattr(client, "API_URL", f"{_get_api_base()}/api")
    url = f"{api_base}/v3/ticker/price"
    try:
        r = requests.get(url, params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return float(data["price"])
    except Exception as e:
        print(f"❌ get_symbol_price_ticker({symbol}) échoué: {e}")
        return None

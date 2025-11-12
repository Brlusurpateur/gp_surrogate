# alerts_telegram.py
import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def _get_telegram_env() -> Tuple[Optional[str], Optional[str]]:
    """
    Récupère les identifiants Telegram depuis les variables d'environnement.

    Returns:
        Tuple[Optional[str], Optional[str]]: (bot_token, chat_id)
            - bot_token: jeton du bot Telegram (TELEGRAM_BOT_TOKEN)
            - chat_id: identifiant du chat cible (TELEGRAM_CHAT_ID)
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return (token or None, chat_id or None)


def _http_post(url: str, data: dict) -> bool:
    """
    Envoie une requête HTTP POST (sans dépendance forte).
    Essaye `requests` si dispo, sinon bascule en `urllib`.

    Args:
        url (str): URL complète de l'endpoint Telegram.
        data (dict): Corps de la requête (form-data).

    Returns:
        bool: True si la requête semble réussie, sinon False.
    """
    # 1) requests si disponible
    try:
        import requests  # type: ignore
        resp = requests.post(url, data=data, timeout=10)
        if 200 <= resp.status_code < 300:
            return True
        logger.warning("POST %s failed: HTTP %s - %s", url, resp.status_code, resp.text[:200])
        return False
    except Exception as e:
        logger.debug("requests indisponible ou erreur (%s), fallback urllib", e)

    # 2) urllib (fallback)
    try:
        import urllib.request
        import urllib.parse
        encoded = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(url, data=encoded)
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec (URL contrôlée Telegram)
            code = getattr(resp, "status", 200)
            return 200 <= code < 300
    except Exception as e:
        logger.warning("POST urllib vers %s échoué: %s", url, e)
        return False


def send_alert(message: str, level: str = "ERROR", dry_run: bool = False) -> bool:
    """
    Envoie une alerte via Telegram (message texte simple).
    - N'échoue pas bruyamment si les identifiants sont absents: log + retour False.
    - Mode 'dry_run=True' : n'envoie pas, mais loggue l'intention (utile en dev).

    Variables d'environnement requises:
        TELEGRAM_BOT_TOKEN : str  (ex: '123456789:ABC-xyz...')
        TELEGRAM_CHAT_ID   : str  (ex: '123456789' pour un DM; ou l'id du groupe)

    Args:
        message (str): Contenu du message (≤ ~4096 chars).
        level (str, optional): Niveau de log local ('INFO', 'WARNING', 'ERROR', ...).
        dry_run (bool, optional): Si True, ne fait aucun appel réseau.

    Returns:
        bool: True si l'envoi semble réussi, False sinon (ou si désactivé).
    """
    logger.log(getattr(logging, level.upper(), logging.ERROR), "ALERTE: %s", message)

    token, chat_id = _get_telegram_env()
    if not token or not chat_id:
        logger.debug("Alertes Telegram désactivées (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID manquants).")
        return False
    if dry_run:
        logger.info("Dry-run Telegram: message non envoyé.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    ok = _http_post(url, data)
    if ok:
        logger.info("Alerte Telegram envoyée (chat_id=%s)", chat_id)
    return ok

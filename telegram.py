# telegram.py
from dotenv import load_dotenv

# Charge d'abord API.env, puis .env (si tu en as un)
load_dotenv("API.env")
load_dotenv()  # optionnel : .env à la racine

from logging_setup import setup_logging
from alerts_telegram import send_alert
import os, requests

def check_env() -> None:
    """
    Affiche les infos essentielles pour diagnostiquer l'envoi Telegram.
    Ne loggue jamais le token en clair (on masque).
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat  = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    token_masked = (token[:9] + "..." + token[-4:]) if token else "(absent)"
    print("[ENV] TELEGRAM_BOT_TOKEN:", token_masked)
    print("[ENV] TELEGRAM_CHAT_ID  :", chat or "(absent)")

def get_last_chat_id() -> int | None:
    """
    Tente de récupérer le dernier chat_id vu par le bot via getUpdates.
    Prérequis: tu as envoyé /start au bot dans le chat cible (DM/groupe).
    Returns:
        int|None: chat_id si trouvé, sinon None.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return None
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        for u in reversed(r.json().get("result", [])):
            m = u.get("message") or u.get("channel_post") or u.get("edited_message")
            if m and "chat" in m:
                return m["chat"]["id"]
    except Exception as e:
        print("[WARN] getUpdates error:", e)
    return None

def main() -> None:
    """
    Initialise le logger, affiche l'état de l'environnement, tente un envoi et
    imprime clairement le résultat (True/False).
    """
    setup_logging()  # logs vers console + logs/app.log
    check_env()

    # (Optionnel) Aide à vérifier le chat_id sans toucher au code du module
    detected = get_last_chat_id()
    if detected is not None:
        print("[INFO] chat_id détecté par getUpdates:", detected)

    ok = send_alert("🔔 Test alert (Telegram) — bot en ligne ✅", level="INFO")
    print("send_alert() →", ok)

if __name__ == "__main__":
    main()

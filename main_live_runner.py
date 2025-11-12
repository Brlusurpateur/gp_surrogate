#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Live Runner — Bot multi-stratégies (profil prudent 2% par trade)

Ce squelette met en place :
- Chargement env (API.env puis .env), switch Testnet/Mainnet via api_mainnet.py
- Logger rotatif + alertes Telegram (crash/événements critiques)
- Garde-fou de risque : 2% du capital global PAR position, et limite d’exposition
- Boucle de décision intraday (polling simple) + hooks de fin de journée (snapshot)
- Persistance d’état minimale (JSON) + arrêt propre (SIGINT/SIGTERM)

À câbler :
- Récupération des signaux depuis tes stratégies (module scalping_signal_engine / autres)
- Exécution ordres (live via python-binance, ou paper via moteur OCO simulé)
- Dashboard/monitoring (optionnel)
"""

from __future__ import annotations

import os
import sys
import json
import time
import signal
import logging
from logging.handlers import TimedRotatingFileHandler
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv

# === Projet : DB & KPI & API & Alertes ===
from backtest_db_manager import init_sqlite_database, migrate_sqlite_schema
from alerts_telegram import send_alert  # suppose déjà configuré (TOKEN, CHAT_ID dans API.env)
from api_mainnet import connect_wallet, get_symbol_price_ticker

# Facultatif selon ton wiring de signaux/exécution
# from scalping_signal_engine import compute_signals_for_all_strats   # TODO: adapter
# from oco_trade_engine import simulate_oco_order                     # TODO: si paper trading local


# ------------------------- Configuration par défaut -------------------------

STATE_PATH = os.environ.get("RUNNER_STATE_PATH", "runner_state.json")
LOG_DIR = os.environ.get("RUNNER_LOG_DIR", "logs")
DB_PATH = os.environ.get("FINAL_DB_PATH", "data/final_results.sqlite")

PAIR = os.environ.get("TRADING_PAIR", "BTCUSDC")
BASE_CCY = "USDC"

INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "1000"))
RISK_PCT_PER_TRADE = float(os.environ.get("RISK_PCT", "0.02"))  # 2%
MAX_CONCURRENT_POSITIONS = int(os.environ.get("MAX_CONCURRENT_POS", "1"))  # prudent: 1
MAX_TOTAL_EXPOSURE_PCT = float(os.environ.get("MAX_TOTAL_EXPOSURE_PCT", "0.02"))  # = 2% agrégé

POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "5"))
USE_TESTNET_ENV = os.environ.get("USE_TESTNET", "0")

# Alertes PnL jour
DAILY_LOSS_ALERT_PCT = float(os.environ.get("DAILY_LOSS_ALERT_PCT", "0.05"))  # -5% par défaut
DAILY_GAIN_ALERT_PCT = float(os.environ.get("DAILY_GAIN_ALERT_PCT", "0.05"))  # +5% par défaut

# ------------------------- Utilitaires généraux -------------------------

def setup_logging() -> logging.Logger:
    """
    Configure un logger rotatif quotidien (UTC) + console.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("runner")
    logger.setLevel(logging.INFO)

    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # fichier (rotation minuit, conserver 14 jours)
    fh = TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, "runner.log"),
        when="midnight",
        backupCount=14,
        utc=True,
        encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


@dataclass
class Position:
    """
    Représente une position unique sur l'instrument (profil prudent = 1 max par défaut).
    """
    side: str                  # "LONG" ou "SHORT" (ici BTC/USDC en spot → plutôt LONG uniquement)
    qty: float
    entry_price: float
    entry_time: float          # epoch seconds (UTC)
    notional_usd: float


@dataclass
class RunnerState:
    """
    État persistant minimal pour reprise après redémarrage.
    """
    capital_usd: float = INITIAL_CAPITAL
    open_positions: List[Position] = None
    last_snapshot_day: str = ""  # "YYYY-MM-DD"
    day_start_capital: float = INITIAL_CAPITAL  # base pour PnL jour & alertes

    def to_json(self) -> str:
        def _enc(o):
            if isinstance(o, Position):
                return asdict(o)
            raise TypeError()
        return json.dumps(asdict(self), default=_enc, ensure_ascii=False, indent=2)

    @staticmethod
    def from_file(path: str) -> "RunnerState":
        if not os.path.exists(path):
            return RunnerState(capital_usd=INITIAL_CAPITAL, open_positions=[])
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        pos = [Position(**p) for p in raw.get("open_positions", [])]
        
        return RunnerState(
            capital_usd=float(raw.get("capital_usd", INITIAL_CAPITAL)),
            open_positions=pos,
            last_snapshot_day=raw.get("last_snapshot_day", ""),
            day_start_capital=float(raw.get("day_start_capital", raw.get("capital_usd", INITIAL_CAPITAL))),
        )

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        os.replace(tmp, path)


class GlobalRiskManager:
    """
    Garde-fou de risque global (2% par trade, exposition agrégée plafonnée).
    """
    def __init__(self, capital_ref_usd: float, risk_pct: float, max_concurrent: int, max_exposure_pct: float):
        self.capital_ref = float(capital_ref_usd)
        self.risk_pct = float(risk_pct)
        self.max_concurrent = int(max_concurrent)
        self.max_exposure_pct = float(max_exposure_pct)

    def target_notional(self, capital_now: float) -> float:
        """
        Notionnel cible par trade = risk_pct * capital_actuel.
        """
        return float(capital_now) * self.risk_pct

    def can_open(self, state: RunnerState) -> bool:
        """
        Autorise l'ouverture d'une nouvelle position si :
        - nombre de positions < max_concurrent
        - somme des notionnels ouverts / capital_now ≤ max_exposure_pct
        """
        n_open = len(state.open_positions or [])
        if n_open >= self.max_concurrent:
            return False
        exposure = sum(p.notional_usd for p in state.open_positions or [])
        if state.capital_usd <= 0:
            return False
        if (exposure / state.capital_usd) >= self.max_exposure_pct:
            return False
        return True


# ------------------------- Stratégies & signaux (placeholder) -------------------------

def load_strategies() -> List[Any]:
    """
    Charge les stratégies actives (placeholder).
    Retour attendu : liste d'objets/fonctions ayant une méthode/func .compute_signal(price, ts) -> (signal, score)
      - signal ∈ {"BUY", "SELL", "HOLD"}
      - score  ∈ float (confiance)
    """
    # TODO: instancier tes stratégies réelles
    return []


def aggregate_signals(strats: List[Any], price: float, ts: float) -> Tuple[str, float, dict]:
    """
    Agrège les signaux des stratégies (ex: voter BUY si la somme des scores BUY > SELL).
    Retourne (signal_global, score_agrégé, diagnostics_par_strat)
    """
    diag = {}
    buy_score = sell_score = 0.0

    for s in strats:
        try:
            sig, score = s.compute_signal(price, ts)  # TODO: adapter prototype réel
            diag[getattr(s, "name", str(s))] = {"signal": sig, "score": score}
            if sig == "BUY":
                buy_score += float(score or 0.0)
            elif sig == "SELL":
                sell_score += float(score or 0.0)
        except Exception as e:
            diag[getattr(s, "name", str(s))] = {"signal": "ERR", "score": 0.0, "error": repr(e)}

    if buy_score > sell_score:
        return "BUY", buy_score, diag
    if sell_score > buy_score:
        return "SELL", sell_score, diag
    return "HOLD", 0.0, diag


# ------------------------- Exécution d'ordres (placeholder) -------------------------

def place_market_order(pair: str, side: str, qty: float, use_testnet: bool) -> Dict[str, Any]:
    """
    Envoi d'un ordre marché (placeholder). À remplacer par l’appel python-binance live.
    Doit retourner un dict avec au minimum: {"status": "FILLED", "price": exec_price, "qty": qty}
    """
    # TODO: implémentation réelle via Client (python-binance) ou moteur paper
    return {"status": "FILLED", "price": None, "qty": qty}


def close_position_market(pos: Position, use_testnet: bool) -> Dict[str, Any]:
    """
    Clôture marché (placeholder). À remplacer par l’appel python-binance.
    """
    # TODO: implémentation réelle
    return {"status": "FILLED", "price": None, "qty": pos.qty}


# ------------------------- Runner principal -------------------------

_shutdown = False
def _signal_handler(sig, frame):
    global _shutdown
    _shutdown = True

def main():
    # 0) ENV + logging
    load_dotenv("API.env")
    load_dotenv()
    logger = setup_logging()

    try:
        # 0bis) DB init/migration
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        init_sqlite_database(DB_PATH)
        migrate_sqlite_schema(DB_PATH)

        # 1) API client (testnet/mainnet)
        use_testnet = str(USE_TESTNET_ENV).lower() in {"1", "true", "yes", "on"}
        client = connect_wallet(testnet=use_testnet)
        logger.info(f"API ready (testnet={use_testnet}) for pair {PAIR}")

        # 2) État & risk manager
        state = RunnerState.from_file(STATE_PATH)
        if state.open_positions is None:
            state.open_positions = []

        # Init day_start_capital si changement de jour
        day_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state.last_snapshot_day != day_utc:
            state.last_snapshot_day = day_utc
            state.day_start_capital = state.capital_usd
            state.save(STATE_PATH)

        risk = GlobalRiskManager(
            capital_ref_usd=state.capital_usd,
            risk_pct=RISK_PCT_PER_TRADE,
            max_concurrent=MAX_CONCURRENT_POSITIONS,
            max_exposure_pct=MAX_TOTAL_EXPOSURE_PCT
        )

        # 3) Stratégies
        strategies = load_strategies()
        logger.info(f"{len(strategies)} stratégie(s) chargée(s).")

        # 4) Gestion arrêt propre
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        send_alert("✅ Runner live démarré")
        logger.info("Runner live started.")

    except Exception as e:
        logger.exception("Global init failed")
        send_alert(f"🚨 Global init failed: {type(e).__name__}: {e}")
        sys.exit(1)

    # 5) Boucle principale
    while not _shutdown:
        loop_ts = time.time()

        try:
            # a) Prix courant
            price = get_symbol_price_ticker(client, PAIR)
            if price is None or not (price > 0):
                logger.warning("Prix indisponible; skip tick.")
                time.sleep(POLL_SECONDS)
                continue

            # b) Hook fin de journée (UTC ~ 23:59) → snapshot + reset base PnL jour
            day_utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            hour_utc = datetime.now(timezone.utc).hour
            if state.last_snapshot_day != day_utc_now and hour_utc >= 23:
                # TODO: enregistrer un résumé PnL / KPI journalier si souhaité
                state.last_snapshot_day = day_utc_now
                state.day_start_capital = state.capital_usd  # reset base jour
                state.save(STATE_PATH)
                logger.info("📄 Daily snapshot saved & day_start_capital reset.")

            # c) Décision: agrégation multi-stratégies
            signal_global, score, diag = aggregate_signals(strategies, price, loop_ts)

            # d) Garde-fou 2% global + trading logique
            if signal_global == "BUY":
                if risk.can_open(state):
                    target = risk.target_notional(state.capital_usd)
                    qty = target / price
                    # Envoi ordre marché (live/paper)
                    resp = place_market_order(PAIR, "BUY", qty, use_testnet)
                    if resp.get("status") == "FILLED":
                        exec_price = float(resp.get("price") or price)
                        pos = Position(
                            side="LONG",
                            qty=float(resp.get("qty", qty)),
                            entry_price=exec_price,
                            entry_time=loop_ts,
                            notional_usd=exec_price * float(resp.get("qty", qty)),
                        )
                        state.open_positions.append(pos)
                        state.save(STATE_PATH)
                        logger.info(f"✅ BUY filled @ {exec_price:.2f} (qty={pos.qty:.6f})")
                    else:
                        logger.warning(f"BUY not filled: {resp}")
                else:
                    logger.info("BUY signal ignoré (garde-fou risque).")

            elif signal_global == "SELL":
                # Politique prudente: si LONG ouvert → on clôture
                if state.open_positions:
                    # (Dans ce squelette, une seule position max)
                    pos = state.open_positions[0]
                    resp = close_position_market(pos, use_testnet)
                    if resp.get("status") == "FILLED":
                        exec_px = float(resp.get("price") or price)
                        pnl = (exec_px - pos.entry_price) * pos.qty
                        state.capital_usd += pnl
                        state.open_positions.clear()
                        state.save(STATE_PATH)
                        logger.info(f"✅ SELL filled @ {exec_px:.2f} | PnL={pnl:.2f} {BASE_CCY} | Capital={state.capital_usd:.2f}")

                        # 🔔 Alertes PnL jour (± seuils)
                        try:
                            pnl_day = state.capital_usd - float(state.day_start_capital)
                            if DAILY_LOSS_ALERT_PCT and pnl_day <= -abs(DAILY_LOSS_ALERT_PCT) * float(state.day_start_capital):
                                send_alert(f"🚨 PnL jour {pnl_day:.2f} {BASE_CCY} (≤ -{DAILY_LOSS_ALERT_PCT*100:.1f}%)")
                            if DAILY_GAIN_ALERT_PCT and pnl_day >=  abs(DAILY_GAIN_ALERT_PCT) * float(state.day_start_capital):
                                send_alert(f"✅ PnL jour {pnl_day:.2f} {BASE_CCY} (≥ +{DAILY_GAIN_ALERT_PCT*100:.1f}%)")
                        except Exception:
                            pass

                    else:
                        logger.warning(f"SELL not filled: {resp}")

            # e) Tempo de polling
            elapsed = time.time() - loop_ts
            sleep_left = max(0.0, POLL_SECONDS - elapsed)
            time.sleep(sleep_left)

        except Exception as e:
            logger.exception("Unhandled exception in main loop")
            send_alert(f"🚨 Crash loop: {type(e).__name__}: {e}")
            # selon préférence: break ou continuer après courte pause
            time.sleep(2)

    # 6) Arrêt propre
    try:
        state.save(STATE_PATH)
    except Exception:
        pass
    logger.info("Runner live stopped.")
    send_alert("🛑 Runner live arrêté proprement")


if __name__ == "__main__":
    main()

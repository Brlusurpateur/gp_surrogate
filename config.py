# -*- coding: utf-8 -*-
"""
Configuration centrale du projet.

- Conserve la compat historique (binance client getter/setter).
- Ajoute un bloc "seuils" centralisé pour le filtrage strict ("export_good_iteration.py").
- Charge des valeurs par défaut raisonnables, surchargées par variables d'environnement.

Variables d'environnement (facultatives)
----------------------------------------
MIN_TRADES                : int     (déf: 30)
MIN_PCT_GREEN             : float   (déf: 0.58 ; accepte 0.58 ou 58)
MIN_SHARPE_D              : float   (déf: 1.20)   # s'applique au Sharpe annualisé avec ANN_FACTOR_DAILY
MIN_PROFIT_FACTOR         : float   (déf: 1.40)
MAX_MDD_ABS               : float   (déf: 0.12 ; accepte 0.12 ou 12)
MAX_TOP5_SHARE            : float   (déf: 0.25)
MAX_ULCER                 : float   (déf: 0.05)
MIN_MEDIAN_DAILY_PNL      : float   (déf: 0.0)
MIN_SKEW_DAILY_PNL        : float   (déf: 0.0)
ANN_FACTOR_DAILY          : int     (déf: 365)     # 252 pour marchés ouvrés actions, 365 pour crypto 24/7 (recommandé)

# Garde-fous anti-outliers (appliqués au calcul/filtrage KPI)
SHARPE_CAP                : float   (déf: 5.0)     # cap/winsorize |Sharpe| pour ignorer les valeurs aberrantes
MIN_ACTIVE_DAYS           : int     (déf: 30)      # nb. de jours distincts de PnL requis pour valider un Sharpe
MIN_STD_DAILY             : float   (déf: 1e-6)    # écart-type min des PnL journaliers (évite division quasi nulle)

"""

from __future__ import annotations
import os
from dataclasses import dataclass
import time
import json
import sqlite3
from contextlib import closing

# Résolution adaptative des seuils (SOFT/HARD/STRICT)
try:
    from adaptive.thresholds import resolve_thresholds as _resolve_thresholds_adaptive
except Exception:
    _resolve_thresholds_adaptive = None  # fallback si module absent

# Petit cache de module pour éviter de relire la DB plusieurs fois par import
_THR_CACHE: dict | None = None
_THR_META: dict | None = None

def _resolve_thresholds_cached() -> tuple[dict, dict]:
    """
    Appelle le résolveur adaptatif s'il est disponible, sinon lève pour fallback.
    Met en cache la dernière décision (seulement au chargement de config.py).
    """
    global _THR_CACHE, _THR_META
    if _THR_CACHE is not None and _THR_META is not None:
        return _THR_CACHE, _THR_META
    if _resolve_thresholds_adaptive is None:
        raise RuntimeError("adaptive.thresholds non disponible")
    thr, meta = _resolve_thresholds_adaptive()
    _THR_CACHE, _THR_META = thr, meta
    # Log léger
    try:
        print(f"[CONFIG] THR_MODE={meta.get('mode')} n_soft={meta.get('n_soft')} n_hard={meta.get('n_hard')} bands={meta.get('band_soft')}/{meta.get('band_hard')}")
    except Exception:
        pass
    return thr, meta

# ──────────────────────────────────────────────────────────────────────────────
# Compat historique : binance client
binance = None

def set_binance_client(client):
    """Enregistre un client Binance au niveau global (compat existante)."""
    global binance
    binance = client

def get_binance_client():
    """Retourne le client Binance global (compat existante)."""
    return binance


# ──────────────────────────────────────────────────────────────────────────────
# Seuils centralisés (filtrage "hard cut" pour l'export)
@dataclass(frozen=True)
class Thresholds:
    """Conteneur immuable des seuils de filtrage “consistency-first”."""
    min_trades: int = 10
    min_pct_green: float = 0.58          # accepte fraction (0.58) ou pourcentage (58)
    min_sharpe_d: float = 0.50
    min_profit_factor: float = 1.10
    max_mdd_abs: float = 0.25            # accepte fraction (0.12) ou pourcentage (12)
    max_top5_share: float = 0.25
    max_ulcer: float = 0.15
    min_median_daily_pnl: float = 0.0
    min_skew_daily_pnl: float = 0.0


def _env_float(name: str, default: float) -> float:
    """Lecture float tolérante depuis l'ENV (retourne default si manquant/illégal)."""
    try:
        val = float(os.environ.get(name, "").strip())
        # Harmonise % → fraction si l'utilisateur écrit 58 plutôt que 0.58
        if name in {"MIN_PCT_GREEN", "MAX_MDD_ABS"} and val > 1.5:
            return val / 100.0
        return val
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    """Lecture int tolérante depuis l'ENV (retourne default si manquant/illégal)."""
    try:
        return int(os.environ.get(name, "").strip())
    except Exception:
        return default
    
def _env_frac(name: str, default: float) -> float:
    """
    Lecture float destinée à des *fractions* (0.0..1.0) mais tolère l'entrée en pourcentage.
    - Si l'utilisateur met 55 ou 8, on considère 55% → 0.55, 8% → 0.08.
    - Sinon on prend la valeur telle quelle (ex: 0.55).
    """
    try:
        val = float(os.environ.get(name, "").strip())
        return val / 100.0 if val > 1.5 else val
    except Exception:
        return default

# ──────────────────────────────────────────────────────────────────────────────
# Paramètres globaux de métriques (Sharpe & cible surrogate/GP)

# - En crypto 24/7, on annualise le Sharpe à partir des PnL/jour avec sqrt(365).
#   On laisse la valeur configurable par ENV pour d'éventuels backtests "marchés ouvrés".
ANN_FACTOR_DAILY: int = _env_int("ANN_FACTOR_DAILY", 365)

# Métrique cible de performance pour le surrogate/GP.
# Par défaut, on vise le Sharpe daily annualisé avec ANN_FACTOR_DAILY (=365) et
# on enregistre/consomme la colonne "sharpe_d_365" dans la base/exports.
# Si tu conserves une colonne unique "sharpe_d" côté DB, garde TARGET_METRIC cohérent.
TARGET_METRIC: str = "sharpe_d_365"

# --- SHARPE_CAP: auto/fixed via résolveur adaptatif ---------------------------
# ENV supportés:
#   SHARPE_CAP_MODE = "auto" | "fixed"   (defaut: "auto")
#   SHARPE_CAP      = valeur de base (auto) ou fixe (fixed), defaut: 5.0
#   BACKTEST_DB_PATH = chemin DB où se trouve 'adaptive_decisions' (si mode=auto)
try:
    from adaptive.sharpe_cap import resolve_sharpe_cap  # fourni dans ton module 'adaptive/'
    _cap, _cap_meta = resolve_sharpe_cap()
    SHARPE_CAP: float = float(_cap)
    # petit log console explicite
    print(
        f"[CONFIG] SHARPE_CAP={SHARPE_CAP} "
        f"(mode={_cap_meta.get('mode')}, N_eff={_cap_meta.get('eligible_count')}, "
        f"fallback={_cap_meta.get('fallback_used')})"
    )
except Exception as _e:
    # Fallback ultra-sûr si le module adaptatif est absent/indisponible
    SHARPE_CAP: float = _env_float("SHARPE_CAP", 5.0)
    print(f"[CONFIG] SHARPE_CAP={SHARPE_CAP} (fallback: {type(_e).__name__})")
# ------------------------------------------------------------------------------
MIN_ACTIVE_DAYS: int = _env_int("MIN_ACTIVE_DAYS", 5)
MIN_STD_DAILY: float = _env_float("MIN_STD_DAILY", 1e-6)


def load_thresholds() -> Thresholds:
    """
    Retourne les seuils STRICTs (vitrine/portfolio) en s’appuyant sur la
    résolution adaptative quand elle est disponible.

    Logique:
      - STRICT (min_sharpe_d, min_profit_factor, min_pct_green) ← adaptatif.STRICT
      - Rails/garde-fous (min_trades, max_mdd_abs, max_ulcer, max_top5_share) ← adaptatif.HARD
      - Garde journalière (min_median_daily_pnl, min_skew_daily_pnl) ← ENV inchangé
    Fallback: si l’adaptatif est indisponible, on lit les ENV comme avant.
    """
    try:
        thr, _meta = _resolve_thresholds_cached()
        H = thr["HARD"]
        T = thr["STRICT"]
        return Thresholds(
            min_trades=int(H["MIN_TRADES"]),
            min_pct_green=float(T["MIN_PCT_GREEN"]),
            min_sharpe_d=float(T["MIN_SHARPE_D"]),
            min_profit_factor=float(T["MIN_PROFIT_FACTOR"]),
            max_mdd_abs=float(H["MAX_MDD_ABS"]),
            max_top5_share=float(H["MAX_TOP5_SHARE"]),
            max_ulcer=float(H["MAX_ULCER"]),
            min_median_daily_pnl=_env_float("MIN_MEDIAN_DAILY_PNL", 0.0),
            min_skew_daily_pnl=_env_float("MIN_SKEW_DAILY_PNL", 0.0),
        )
    except Exception:
        # Fallback ENV (comportement historique)
        return Thresholds(
            min_trades=_env_int("MIN_TRADES", 5),
            min_pct_green=_env_float("MIN_PCT_GREEN", 0.2),
            min_sharpe_d=_env_float("MIN_SHARPE_D", 0.50),
            min_profit_factor=_env_float("MIN_PROFIT_FACTOR", 0.80),
            max_mdd_abs=_env_float("MAX_MDD_ABS", 0.25),
            max_top5_share=_env_float("MAX_TOP5_SHARE", 0.40),
            max_ulcer=_env_float("MAX_ULCER", 0.15),
            min_median_daily_pnl=_env_float("MIN_MEDIAN_DAILY_PNL", 0.0),
            min_skew_daily_pnl=_env_float("MIN_SKEW_DAILY_PNL", 0.0),
        )

# ──────────────────────────────────────────────────────────────────────────────
# Hard / Soft thresholds pour la sélection multi-objectif
# - HARD : coupe stricte (élimination immédiate)
# - SOFT : minimas “souples” utilisés pour le scoring
#
# ENV acceptés (exemples):
#   HARD_MIN_TRADES=40
#   HARD_MAX_MDD=0.15  (ou 15)
#   HARD_MAX_ULCER=0.08 (ou 8)
#   HARD_MAX_TOP5=0.35 (ou 35)
#   SOFT_MIN_SHARPE=1.0
#   SOFT_MIN_PF=1.3
#   SOFT_MIN_GREEN=0.55 (ou 55)
#
# Poids du score (overrides possibles):
#   SCORE_W_SHARPE=0.4 SCORE_W_PF=0.2 SCORE_W_GREEN=0.1 SCORE_W_MDD=0.2 SCORE_W_ULCER=0.05 SCORE_W_TOP5=0.05
# Les poids seront normalisés pour sommer à 1.0.

def hard_soft_thresholds() -> tuple[dict, dict]:
    """
    Retourne (HARD, SOFT) avec priorité à la résolution adaptative.
    - HARD : garde-fous communs (MIN_TRADES, MAX_MDD, MAX_ULCER, MAX_TOP5)
    - SOFT : minima souples pour élargir le pool GP/BO (MIN_SHARPE, MIN_PF, MIN_GREEN)
    Fallback: si l’adaptatif est indisponible, on conserve le comportement ENV existant.
    """
    try:
        thr, _meta = _resolve_thresholds_cached()
        H = thr["HARD"]
        S = thr["SOFT"]
        HARD = dict(
            MIN_TRADES=int(H["MIN_TRADES"]),
            MAX_MDD=float(H["MAX_MDD_ABS"]),
            MAX_ULCER=float(H["MAX_ULCER"]),
            MAX_TOP5=float(H["MAX_TOP5_SHARE"]),
        )
        SOFT = dict(
            MIN_SHARPE=float(S["MIN_SHARPE"]),
            MIN_PF=float(S["MIN_PF"]),
            MIN_GREEN=float(S["MIN_GREEN"]),
        )
        return HARD, SOFT
    except Exception:
        # --- Fallback ENV (comportement historique) ---
        HARD = dict(
            MIN_TRADES=_env_int("HARD_MIN_TRADES", 15),
            MAX_MDD=_env_frac("HARD_MAX_MDD", 0.20),
            MAX_ULCER=_env_frac("HARD_MAX_ULCER", 0.12),
            MAX_TOP5=_env_frac("HARD_MAX_TOP5", 0.50),
        )
        RELAX_SOFT = int(os.getenv("RELAX_SOFT", "1"))
        _SOFT_BASE_STRICT = dict(
            MIN_SHARPE=float(os.getenv("SOFT_MIN_SHARPE", 1.0)),
            MIN_PF    =float(os.getenv("SOFT_MIN_PF", 1.3)),
            MIN_GREEN =float(os.getenv("SOFT_MIN_GREEN", 0.55)),
        )
        _SOFT_BASE_RELAXED = dict(
            MIN_SHARPE=float(os.getenv("SOFT_MIN_SHARPE", 0.1)),
            MIN_PF    =float(os.getenv("SOFT_MIN_PF", 0.70)),
            MIN_GREEN =float(os.getenv("SOFT_MIN_GREEN", 0.1)),
        )
        SOFT = _SOFT_BASE_RELAXED if RELAX_SOFT else _SOFT_BASE_STRICT
        return HARD, SOFT


def _env_weight(name: str, default: float) -> float:
    """Lecture d'un poids non-négatif ; fallback sur default si invalide."""
    try:
        v = float(os.environ.get(name, "").strip())
        is_finite = (_np is not None and _np.isfinite(v)) if 'v' in locals() else False
        return v if (v >= 0.0 and is_finite) else default
    except Exception:
        return default

# NOTE: numpy est optionnel ici ; on évite une dépendance dure
try:
    import numpy as _np
except Exception:
    _np = None


def score_weights(normalize: bool = True) -> dict:
    """
    Retourne un dict de poids (normalisés) pour le score composite :
        score = w_S*Sharpe + w_PF*PF + w_G*Green - w_M*|MDD| - w_U*Ulcer - w_T*Top5
    Par défaut :
        SHARPE=0.35, PF=0.20, GREEN=0.10, MDD=0.20, ULCER=0.10, TOP5=0.05
    Overrides ENV :
        SCORE_W_SHARPE, SCORE_W_PF, SCORE_W_GREEN, SCORE_W_MDD, SCORE_W_ULCER, SCORE_W_TOP5
    """
    base = dict(SHARPE=0.35, PF=0.20, GREEN=0.10, MDD=0.20, ULCER=0.10, TOP5=0.05)

    w = {
        "SHARPE": _env_weight("SCORE_W_SHARPE", base["SHARPE"]),
        "PF":     _env_weight("SCORE_W_PF",     base["PF"]),
        "GREEN":  _env_weight("SCORE_W_GREEN",  base["GREEN"]),
        "MDD":    _env_weight("SCORE_W_MDD",    base["MDD"]),
        "ULCER":  _env_weight("SCORE_W_ULCER",  base["ULCER"]),
        "TOP5":   _env_weight("SCORE_W_TOP5",   base["TOP5"]),
    }

    if normalize:
        s = sum(w.values())
        if s > 0.0:
            if _np is not None:
                # petite normalisation numérique stable
                arr = _np.array(list(w.values()), dtype=float)
                arr = arr / float(s)
                for k, v in zip(w.keys(), arr.tolist()):
                    w[k] = v
            else:
                for k in w:
                    w[k] = w[k] / s
    return w

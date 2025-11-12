# indicators_core.py
"""
Source de vérité unique pour les indicateurs.
API numpy-first, déterministe, sans dépendance temps-réel.

Fonctions exposées
- ema(price, period)
- rsi(price, period=14)
- macd(price, fast=12, slow=26, signal=9)
- atr(high, low, close, period=14)            # Wilder
- vwap_daily(ts, price, volume)               # reset quotidien
- combined_volatility(close, atr_vals, ewma_period=20, weight_atr=0.5)
- bollinger_bandwidth(close, period=20, mult=2.0)  # proxy "range"

Utilitaires (stabilité & régimes)
- rolling_var(x, window)                      # variance glissante
- rolling_downside_std(rets, window)          # std des pertes (downside) glissante
- downside_std(rets)                          # std des pertes (série entière)
- rolling_slope(x, window)                    # pente (drift) glissante via régression linéaire
- zscore_ewma(x, span)                        # z-score lissé EWMA
- trend_strength_proxy(close, window=50)      # proxy "trend" (pente normée × R² local)
- market_condition_proxy(close, window=50, bb_period=20, bb_mult=2.0, thr_trend=0.5, thr_range=0.03)
                                              # 1 = trend, 0 = range, -1 = inconnu
- market_regime_tag(vol_series, idx, threshold_factor, market_condition)
                                              # 'vol' / 'trend' / 'range' / 'unknown'
"""

from __future__ import annotations
import numpy as np

# ---------- helpers ----------
def _as_float1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("Input must be 1-D")
    return a

def _sma_init(x: np.ndarray, period: int) -> float:
    # Moyenne simple sur la première fenêtre non-NaN suffisante
    valid = x[~np.isnan(x)]
    if len(valid) >= period:
        return float(np.nanmean(x[:period]))
    # sinon, premier non-NaN
    for v in x:
        if not np.isnan(v):
            return float(v)
    return float("nan")

# ---------- EMA ----------
def ema(price, period: int) -> np.ndarray:
    """EMA classique (alpha=2/(period+1)), init par SMA(period)."""
    p = _as_float1d(price)
    n = len(p)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or period <= 0:
        return out
    alpha = 2.0 / (period + 1.0)
    # init
    start = min(period, n)
    seed = _sma_init(p, period)
    if np.isnan(seed):
        return out
    out[start-1] = seed
    prev = seed
    for i in range(start, n):
        x = p[i]
        if np.isnan(x):
            out[i] = prev
            continue
        prev = alpha * x + (1.0 - alpha) * prev
        out[i] = prev
    # si start-2 etc. sont NaN, on laisse NaN (warmup)
    return out

# ---------- RSI (Wilder) ----------
def rsi(price, period: int = 14) -> np.ndarray:
    """RSI de Wilder (moyennes lissées des gains/pertes)."""
    p = _as_float1d(price)
    n = len(p)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2 or period <= 0:
        return out

    delta = np.diff(p)
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)

    # moyennes initiales (SMA sur la première fenêtre)
    if n - 1 < period:
        return out
    avg_gain = np.nanmean(gains[:period])
    avg_loss = np.nanmean(losses[:period])

    rs = np.inf if avg_loss == 0 else (avg_gain / avg_loss)
    out[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder smoothing alpha = 1/period
    for i in range(period + 1, n):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        rs = np.inf if avg_loss == 0 else (avg_gain / avg_loss)
        out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out

# ---------- MACD ----------
def macd(price, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    p = _as_float1d(price)
    ema_fast = ema(p, fast)
    ema_slow = ema(p, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ---------- ATR (Wilder) ----------
def atr(high, low, close, period: int = 14) -> np.ndarray:
    """ATR de Wilder (smoothing alpha=1/period)."""
    h = _as_float1d(high)
    l = _as_float1d(low)
    c = _as_float1d(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if not (len(h) == len(l) == len(c)) or n == 0 or period <= 0:
        return out

    tr = np.full(n, np.nan, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        tr[i] = max(hl, hc, lc)

    # init ATR = SMA(TR, period)
    if n < period:
        return out
    atr_prev = np.nanmean(tr[:period])
    out[period - 1] = atr_prev
    # Wilder smoothing
    for i in range(period, n):
        atr_prev = (atr_prev * (period - 1) + tr[i]) / period
        out[i] = atr_prev
    return out

# ---------- VWAP (reset quotidien) ----------
def vwap_daily(ts, price, volume) -> np.ndarray:
    """
    VWAP avec reset journalier (ts en datetime64, price = typical price ou close).
    """
    t = np.asarray(ts)
    p = _as_float1d(price)
    v = _as_float1d(volume)
    n = len(p)
    out = np.full(n, np.nan, dtype=np.float64)
    if not (len(t) == n == len(v)) or n == 0:
        return out

    # clé jour
    days = t.astype('datetime64[D]')
    start = 0
    while start < n:
        day = days[start]
        end = start + 1
        while end < n and days[end] == day:
            end += 1
        # segment [start:end)
        seg_p = p[start:end]
        seg_v = v[start:end]
        tpv = seg_p * seg_v
        ctpv = np.cumsum(tpv)
        cv = np.cumsum(seg_v)
        seg_vwap = np.divide(ctpv, cv, out=np.full_like(ctpv, np.nan), where=(cv != 0))
        out[start:end] = seg_vwap
        start = end
    return out

# ---------- Combined Volatility ----------
def combined_volatility(close, atr_vals, ewma_period: int = 20, weight_atr: float = 0.5) -> np.ndarray:
    """
    Volatilité composite : ATR% (atr/close) + EMA des |log-returns| (pondérée).
    weight_atr in [0,1] : poids de la composante ATR%.
    """
    c = _as_float1d(close)
    a = _as_float1d(atr_vals)
    if len(c) != len(a):
        raise ValueError("close and atr_vals must have same length")
    # ATR% (évite div0)
    atr_pct = np.divide(a, c, out=np.zeros_like(a), where=(c != 0))
    # abs log-returns
    lr = np.zeros_like(c)
    lr[1:] = np.abs(np.diff(np.log(np.clip(c, 1e-12, None))))
    lr_ema = ema(lr, ewma_period)
    comb = weight_atr * atr_pct + (1.0 - weight_atr) * lr_ema
    return comb

def bollinger_bandwidth(close, period: int = 20, mult: float = 2.0) -> np.ndarray:
    """
    Bollinger Bandwidth = (Upper - Lower) / Middle
    - Middle: SMA(period)
    - Upper/Lower: Middle ± mult * std(period)

    Args:
        close (array-like): Série de clôtures.
        period (int): Période de calcul de la bande.
        mult (float): Multiplicateur d'écart-type (typiquement 2.0).

    Returns:
        np.ndarray: Bandwidth (même longueur que close), NaN pendant le warmup.
    """
    c = _as_float1d(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or period <= 1:
        return out

    # SMA et STD glissantes (naïf, suffisant pour backtests)
    for i in range(period - 1, n):
        seg = c[i - period + 1:i + 1]
        m = float(np.nanmean(seg))
        s = float(np.nanstd(seg, ddof=1)) if period > 1 else 0.0
        upper = m + mult * s
        lower = m - mult * s
        out[i] = (upper - lower) / m if m != 0.0 else np.nan
    return out

# ---------- Stability & Regime Utilities ----------
def rolling_var(x: np.ndarray, window: int) -> np.ndarray:
    """
    Variance glissante (ddof=1) sur fenêtre fixe.

    Args:
        x (np.ndarray): Série 1D.
        window (int): Longueur de fenêtre (>=2).

    Returns:
        np.ndarray: Même longueur que x, NaN pour les warmups < window.
    """
    a = _as_float1d(x)
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float64)
    if window is None or window < 2 or n == 0:
        return out
    # calcul naïf O(n·w) — suffisant pour backtests; vectorisation possible ultérieurement
    for i in range(window - 1, n):
        seg = a[i - window + 1:i + 1]
        if np.all(np.isnan(seg)):
            continue
        out[i] = np.nanvar(seg, ddof=1)
    return out


def rolling_downside_std(rets: np.ndarray, window: int) -> np.ndarray:
    """
    Downside standard deviation glissante (pertes seulement).

    Args:
        rets (np.ndarray): Rendements (e.g., quotidiens).
        window (int): Longueur de fenêtre.

    Returns:
        np.ndarray: Même longueur que rets; NaN si insuffisant.
    """
    r = _as_float1d(rets)
    n = len(r)
    out = np.full(n, np.nan, dtype=np.float64)
    if window is None or window < 2 or n == 0:
        return out
    for i in range(window - 1, n):
        seg = r[i - window + 1:i + 1]
        neg = seg[seg < 0.0]
        if neg.size < 2:
            continue
        out[i] = np.nanstd(neg, ddof=1)
    return out

def downside_std(rets: np.ndarray) -> float:
    """
    Écart-type des pertes (downside) sur toute la série.

    Args:
        rets (np.ndarray): Rendements (1D).

    Returns:
        float: Downside standard deviation (ddof=1 sur valeurs négatives), NaN si insuffisant.

    Notes:
        - Utile pour des calculs de Sortino "côté indicateurs" (hors KPI core).
    """
    r = _as_float1d(rets)
    neg = r[r < 0.0]
    if neg.size < 2:
        return float("nan")
    return float(np.nanstd(neg, ddof=1))

def rolling_slope(x: np.ndarray, window: int) -> np.ndarray:
    """
    Pente (drift) glissante d'une série par régression linéaire y ~ a*t + b.

    Args:
        x (np.ndarray): Série 1D.
        window (int): Longueur de fenêtre (>=2).

    Returns:
        np.ndarray: slope par point (NaN pour warmups < window).
    """
    a = _as_float1d(x)
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float64)
    if window is None or window < 2 or n == 0:
        return out
    t = np.arange(window, dtype=np.float64)
    # polyfit par fenêtre (simple et robuste; optimisations possibles si besoin)
    for i in range(window - 1, n):
        seg = a[i - window + 1:i + 1]
        if np.all(np.isnan(seg)):
            continue
        try:
            coef = np.polyfit(t, seg.astype(np.float64), 1)
            out[i] = float(coef[0])
        except Exception:
            out[i] = np.nan
    return out


def zscore_ewma(x: np.ndarray, span: int = 20) -> np.ndarray:
    """
    Z-score lissé par EWMA : (x - ewma(x)) / ewma(|x - ewma(x)|).

    Args:
        x (np.ndarray): Série 1D.
        span (int): Pseudo-période EWMA (alpha = 2/(span+1)).

    Returns:
        np.ndarray: Z-score EWMA, NaN aux warmups.
    """
    a = _as_float1d(x)
    n = len(a)
    if n == 0 or span <= 0:
        return np.full(n, np.nan, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)

    def _ewma(v: np.ndarray) -> np.ndarray:
        out = np.full_like(v, np.nan, dtype=np.float64)
        # init au premier non-NaN
        idx = np.where(~np.isnan(v))[0]
        if idx.size == 0:
            return out
        i0 = int(idx[0])
        out[i0] = float(v[i0])
        prev = out[i0]
        for i in range(i0 + 1, len(v)):
            x = v[i]
            if np.isnan(x):
                out[i] = prev
            else:
                prev = alpha * x + (1.0 - alpha) * prev
                out[i] = prev
        return out

    mu = _ewma(a)
    dev = _ewma(np.abs(a - mu))
    z = np.divide(a - mu, dev, out=np.full(n, np.nan, dtype=np.float64), where=(dev != 0))
    return z


def market_regime_tag(
    vol_series: np.ndarray,
    idx: int,
    threshold_factor: float,
    market_condition: int
) -> str:
    """
    Classe un régime 'vol' / 'trend' / 'range' à l'instant `idx`.

    Règle :
      - 'vol'   si vol[idx] >= 1.5 * median(vol[window]) * threshold_factor
      - sinon 'trend' si market_condition == 1
      - sinon 'range' si market_condition == 0
      - sinon 'unknown'

    Args:
        vol_series (np.ndarray): Série de volatilité (ex: combined_volatility).
        idx (int): Index courant.
        threshold_factor (float): Facteur multiplicatif (cf. params.threshold_factor).
        market_condition (int): 1 = trend, 0 = range, autre = inconnu.

    Returns:
        str: 'vol' | 'trend' | 'range' | 'unknown'
    """
    v = _as_float1d(vol_series)
    n = len(v)
    if n == 0 or idx < 0 or idx >= n or not np.isfinite(v[idx]):
        return "unknown"

    # fenêtre locale pour la médiane : min(100, n) centrée sur idx si possible
    w = int(min(100, n))
    start = max(0, idx - w + 1)
    seg = v[start:idx + 1]
    if seg.size == 0 or np.all(np.isnan(seg)):
        return "unknown"

    med = float(np.nanmedian(seg))
    thr = med * float(threshold_factor)
    if v[idx] >= 1.5 * thr:
        return "vol"
    if market_condition == 1:
        return "trend"
    if market_condition == 0:
        return "range"
    return "unknown"

def trend_strength_proxy(close: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Proxy "trend" local combinant pente normalisée et qualité d'ajustement (R²) d'une régression
    linéaire sur fenêtre glissante.

    Forme:
        score[i] = |slope_i| / std(seg) * R2_i
    → score ∈ [0, +∞), plus grand = tendance plus marquée et plus "propre".

    Args:
        close (np.ndarray): Série 1D de prix de clôture.
        window (int): Fenêtre locale pour l'ajustement.

    Returns:
        np.ndarray: Score de trend (NaN sur warmup).

    Notes:
        - Normalisation par l'écart-type local pour rendre le slope "scale-free".
        - R² borne la contribution si l'ajustement est faible (choppy/range).
    """
    c = _as_float1d(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or window < 3:
        return out

    t = np.arange(window, dtype=np.float64)
    for i in range(window - 1, n):
        seg = c[i - window + 1:i + 1]
        if np.all(np.isnan(seg)):
            continue
        try:
            coef = np.polyfit(t, seg.astype(np.float64), 1)
            slope = float(coef[0])
            # R²
            yhat = np.polyval(coef, t)
            ss_res = float(np.nansum((seg - yhat) ** 2))
            ss_tot = float(np.nansum((seg - np.nanmean(seg)) ** 2))
            r2 = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)
            std_loc = float(np.nanstd(seg, ddof=1))
            norm = 0.0 if std_loc == 0.0 or not np.isfinite(std_loc) else abs(slope) / std_loc
            out[i] = norm * r2
        except Exception:
            out[i] = np.nan
    return out


def market_condition_proxy(
    close: np.ndarray,
    window: int = 50,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    thr_trend: float = 0.5,
    thr_range: float = 0.03
) -> np.ndarray:
    """
    Proxy de condition de marché ∈ {1 (trend), 0 (range), -1 (unknown)} par point.

    Règle :
      - calcule trend_strength_proxy(close, window)
      - calcule bollinger_bandwidth(close, bb_period, bb_mult)
      - si trend_score >= thr_trend → 1 (trend)
      - elif bb_bandwidth <= thr_range → 0 (range)
      - else → -1 (unknown)

    Args:
        close (np.ndarray): Série de clôture.
        window (int): Fenêtre pour trend_strength_proxy.
        bb_period (int): Période des bandes de Bollinger.
        bb_mult (float): Multiplicateur des bandes.
        thr_trend (float): Seuil de trend (ajuster selon instrument/timeframe).
        thr_range (float): Seuil de bandwidth pour "range".

    Returns:
        np.ndarray: Série 1D de {1, 0, -1}.
    """
    c = _as_float1d(close)
    n = len(c)
    out = np.full(n, -1.0, dtype=np.float64)  # unknown par défaut

    ts = trend_strength_proxy(c, window=window)
    bb = bollinger_bandwidth(c, period=bb_period, mult=bb_mult)

    for i in range(n):
        ts_i = ts[i] if i < len(ts) else np.nan
        bb_i = bb[i] if i < len(bb) else np.nan
        if np.isfinite(ts_i) and ts_i >= thr_trend:
            out[i] = 1.0
        elif np.isfinite(bb_i) and bb_i <= thr_range:
            out[i] = 0.0
        else:
            out[i] = -1.0
    return out

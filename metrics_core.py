# metrics_core.py
"""
Source de vérité unique pour les KPI de backtest (intra-day, par stratégie).

Principes généraux
- Les métriques "level" journalières s'appuient sur des rendements quotidiens
  dérivés de l'équité quotidienne (balance → resample 1D → pct_change).
- Si `balance` n'est pas disponible, certains KPI chutent en mode "fallback" via
  l'agrégation PnL/jour (moins précis, mais robuste).

Métriques couvertes ici (par backtest, calculables avec `trades_df` seul)
- Performance (annualisée/journalière) : sharpe_d_252, sortino_d_252,
  calmar_ratio, mean_daily_return, vol_daily_return, return_cagr (approx),
  profit_factor.
- Risque & extrêmes : max_drawdown, ulcer_index, dd_duration_max,
  worst_day_return, var_95_d, cvar_95_d, max_consecutive_losses, tail_ratio.
- Cohérence intraday (consistency-first) : nb_trades, trades_per_day (médiane),
  pct_green_days, median_daily_pnl, skew_daily_pnl, win_rate, avg_win/loss,
  expectancy_per_trade, time_to_recover_median, intra_day_hold_time_median.
- Qualité d’exécution (si colonnes disponibles) : fees_as_pct_gross,
  slippage_bps, fill_ratio, avg_spread_bps_captured, participation_rate_max,
  capacity_flag.

Remarques
- Les métriques nécessitant un contexte multi-stratégies (ex. pairwise_corr_to_others)
  ou un benchmark externe (alpha_vs_bh, information_ratio) sont gérées via
  paramètres optionnels (et renvoient NaN si non fournis).
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd

def _to_float_series(series):
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce")

def compute_kpis(trades_df, periods_per_year: float = 252.0, annualize: bool = True) -> Tuple[float, float, float, float]:
    """
    Calcule (sharpe, max_drawdown, profit_factor, win_rate) à partir d'un DataFrame `trades`
    contenant au moins `pnl_net`, idéalement aussi `balance`.
    """
    if trades_df is None or len(trades_df) == 0 or "pnl_net" not in trades_df.columns:
        return (np.nan, np.nan, np.nan, np.nan)

    pnl = _to_float_series(trades_df["pnl_net"]).dropna()

    # --- Equity-based metrics ---
    sharpe = np.nan
    max_dd = np.nan
    if "balance" in trades_df.columns:
        balance = _to_float_series(trades_df["balance"]).dropna()
        if len(balance) >= 2:
            rets = balance.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            if len(rets) >= 2 and rets.std(ddof=1) > 0:
                sharpe = rets.mean() / rets.std(ddof=1)
                if annualize:
                    sharpe *= np.sqrt(float(periods_per_year))
            cummax = balance.cummax()
            dd = (balance / cummax) - 1.0
            if not dd.empty:
                max_dd = float(dd.min())
    else:
        # Fallback si pas de balance
        if len(pnl) >= 2 and pnl.std(ddof=1) > 0:
            sharpe = pnl.mean() / pnl.std(ddof=1)
            if annualize:
                sharpe *= np.sqrt(float(periods_per_year))
        cum = pnl.cumsum()
        dd = cum - cum.cummax()
        max_dd = float(dd.min()) if not dd.empty else np.nan

    gains = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
    losses_sum = float(pnl[pnl < 0].sum()) if len(pnl) else 0.0
    if losses_sum == 0.0:
        profit_factor = np.inf if gains > 0.0 else 0.0
    else:
        profit_factor = float(gains / abs(losses_sum))
    win_rate = float((pnl > 0).mean()) if len(pnl) else np.nan

    return (
        float(sharpe) if np.isfinite(sharpe) else np.nan,
        float(max_dd) if np.isfinite(max_dd) else np.nan,
        float(profit_factor) if np.isfinite(profit_factor) else np.nan,
        float(win_rate) if np.isfinite(win_rate) else np.nan,
    )

# ===========================
# KPI QUOTIDIENS (INTRA-DAY)
# ===========================

def _get_ts_col(df):
    # préfère 'timestamp', sinon 'exit_time', sinon None
    for c in ("timestamp", "exit_time"):
        if c in df.columns:
            return c
    return None

def daily_returns_from_trades(trades_df) -> "pd.Series":
    """
    Construit une série de rendements journaliers:
      - si 'balance' existe: ret_d = balance_last_by_day.pct_change()
      - sinon: fallback sur PnL normalisé (somme/jour, / std abs(pnl))
    Retourne une pd.Series indexée par date (UTC, normalisée à 00:00).
    """
    if trades_df is None or len(trades_df) == 0:
        return pd.Series(dtype=float)

    ts_col = _get_ts_col(trades_df)
    if ts_col is None:
        return pd.Series(dtype=float)

    df = trades_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col])
    if df.empty:
        return pd.Series(dtype=float)

    if "balance" in df.columns:
        # equity → resample en daily (dernier point du jour), puis pct_change
        daily_equity = (
            df.set_index(ts_col)["balance"]
              .sort_index()
              .resample("1D")
              .last()
              .dropna()
        )
        rets = daily_equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        return rets
    else:
        # fallback: agrège PnL/jour et normalise par l'écart-type absolu (variance stable)
        pnl = _to_float_series(df.get("pnl_net")).dropna()
        if pnl.empty:
            return pd.Series(dtype=float)
        daily_pnl = (
            df.assign(_pnl=pnl)
              .set_index(ts_col)["_pnl"]
              .sort_index()
              .resample("1D")
              .sum()
              .dropna()
        )
        scale = float(np.std(np.abs(pnl), ddof=1)) if len(pnl) > 1 else 1.0
        scale = scale if scale > 0 else 1.0
        rets = (daily_pnl / scale).replace([np.inf, -np.inf], np.nan).dropna()
        return rets

def compute_percent_green_days(daily_rets: "pd.Series") -> float:
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    return float((daily_rets > 0).mean())

def compute_sortino_annualized(daily_rets: "pd.Series", periods_per_year: float = 252.0) -> float:
    neg = daily_rets[daily_rets < 0]
    dd = np.std(neg, ddof=1) if len(neg) > 1 else np.nan
    if not np.isfinite(dd) or dd == 0:
        return np.nan
    ratio = daily_rets.mean() / dd
    return float(ratio * np.sqrt(periods_per_year))

def compute_sharpe_daily_annualized(daily_rets: "pd.Series", periods_per_year: float = 252.0) -> float:
    if daily_rets is None or len(daily_rets) < 2:
        return np.nan
    sd = float(daily_rets.std(ddof=1))
    if sd <= 0:
        return np.nan
    return float(daily_rets.mean() / sd * np.sqrt(periods_per_year))

def compute_ulcer_index(daily_rets: "pd.Series") -> float:
    """
    Ulcer Index sur l'équity quotidienne: sqrt(moyenne(drawdown^2)).
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    equity = (1.0 + daily_rets.fillna(0)).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(np.sqrt(np.mean(np.square(dd))))

def compute_cvar(daily_rets: "pd.Series", alpha: float = 0.95) -> float:
    """
    CVaR (Expected Shortfall) côté perte (quantile 1-alpha).
    Retourne une valeur négative (perte attendue conditionnelle).
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    q = np.quantile(daily_rets, 1.0 - alpha)
    tail = daily_rets[daily_rets <= q]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())

def compute_median_daily_pnl(daily_rets: "pd.Series") -> float:
    """
    Médiane des rendements journaliers (robuste aux outliers).
    Renvoie NaN si la série est vide.
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    x = np.asarray(daily_rets, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float(np.nanmedian(x))


def compute_skew_daily_pnl(daily_rets: "pd.Series") -> float:
    """
    Skewness (asymétrie) corrigée du biais de Fisher-Pearson sur rendements journaliers.
    Renvoie NaN si longueur < 3 ou écart-type nul.
    """
    if daily_rets is None or len(daily_rets) < 3:
        return np.nan
    x = np.asarray(daily_rets, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        return np.nan
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return np.nan
    z = (x - mu) / sd
    # Fisher-Pearson unbiased estimator
    g1 = (n / ((n - 1) * (n - 2))) * float(np.sum(z ** 3))
    return float(g1)

def compute_var(daily_rets: "pd.Series", alpha: float = 0.95) -> float:
    """
    Compute Value-at-Risk (VaR) quotidienne au niveau `alpha` (ex: 0.95).
    Convention: VaR renvoyée comme une **valeur négative** (perte).
    Args:
        daily_rets: Série de rendements journaliers.
        alpha: Niveau de confiance (0.95 -> quantile 5% à gauche).
    Returns:
        float: VaR (<= 0) ou NaN si insuffisant.
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    q = float(np.quantile(daily_rets, 1.0 - alpha))
    return q


def compute_tail_ratio(daily_rets: "pd.Series", upper_q: float = 0.95, lower_q: float = 0.05) -> float:
    """
    Tail Ratio = moyenne des gains au-delà du quantile supérieur /
                 |moyenne des pertes en-deçà du quantile inférieur|.
    Args:
        daily_rets: Série de rendements journaliers.
        upper_q: Quantile haut (e.g. 0.95).
        lower_q: Quantile bas (e.g. 0.05).
    Returns:
        float: Ratio (>1 préférable) ou NaN.
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    up_thr = np.quantile(daily_rets, upper_q)
    lo_thr = np.quantile(daily_rets, lower_q)
    upper = daily_rets[daily_rets >= up_thr]
    lower = daily_rets[daily_rets <= lo_thr]
    if upper.size == 0 or lower.size == 0:
        return np.nan
    num = float(upper.mean())
    den = abs(float(lower.mean()))
    return np.nan if den == 0 else float(num / den)


def _equity_curve_from_daily_returns(daily_rets: "pd.Series") -> "pd.Series":
    """
    Construit une equity curve normalisée (base 1.0) depuis les rendements journaliers.
    Returns:
        pd.Series: Equity curve (index = dates).
    """
    if daily_rets is None or len(daily_rets) == 0:
        return pd.Series(dtype=float)
    return (1.0 + daily_rets.fillna(0)).cumprod()


def compute_calmar_ratio(daily_rets: "pd.Series", periods_per_year: float = 252.0) -> float:
    """
    Calmar = rendement annualisé (approx) / |max_drawdown|.
    Returns:
        float: Calmar ratio ou NaN si DD nul/inconnu.
    """
    if daily_rets is None or len(daily_rets) < 2:
        return np.nan
    equity = _equity_curve_from_daily_returns(daily_rets)
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min()) if not dd.empty else np.nan
    if not np.isfinite(max_dd) or max_dd == 0:
        return np.nan
    mu_d = float(daily_rets.mean())
    # Annualisation approchée via `periods_per_year`
    ann_return = (1.0 + mu_d) ** periods_per_year - 1.0
    return float(ann_return / abs(max_dd)) if max_dd != 0 else np.nan


def compute_cagr_from_daily(daily_rets: "pd.Series", periods_per_year: float = 252.0) -> float:
    """
    Approximates CAGR depuis des retours journaliers.
    Si moins d'un an de données, l'annualisation est extrapolée.
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    equity = _equity_curve_from_daily_returns(daily_rets)
    if equity.empty:
        return np.nan
    total_days = float(len(daily_rets))
    years = total_days / periods_per_year
    final = float(equity.iloc[-1])
    if final <= 0:
        return np.nan
    return float(final ** (1.0 / max(years, 1e-9)) - 1.0)


def compute_dd_duration_max(daily_rets: "pd.Series") -> float:
    """
    Durée maximale (en jours) d'un drawdown (temps entre un pic et la
    pleine récupération suivante). Retourne NaN si insuffisant.
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    equity = _equity_curve_from_daily_returns(daily_rets)
    cummax = equity.cummax()
    in_dd = equity < cummax
    if in_dd.sum() == 0:
        return 0.0
    # Compter la longueur des runs consécutifs de True
    max_len = 0
    curr = 0
    for flag in in_dd.astype(int).values:
        if flag:
            curr += 1
            if curr > max_len:
                max_len = curr
        else:
            curr = 0
    return float(max_len)


def compute_time_to_recover_median(daily_rets: "pd.Series") -> float:
    """
    Médiane des durées (en jours) nécessaires pour revenir au high watermark
    après être entré en drawdown. Si aucune récup complète, NaN.
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    equity = _equity_curve_from_daily_returns(daily_rets)
    hwm = equity.cummax()
    ttr = []
    in_dd = False
    start = None
    for i, (eq, hm) in enumerate(zip(equity.values, hwm.values)):
        if eq < hm and not in_dd:
            in_dd, start = True, i
        elif eq >= hm and in_dd:
            in_dd = False
            ttr.append(i - start)
            start = None
    if len(ttr) == 0:
        return np.nan
    return float(np.median(ttr))


def compute_trades_per_day_median(trades_df) -> float:
    """
    Médiane du nombre de trades par jour (approx).
    Nécessite une colonne temporelle ('timestamp' ou 'exit_time').
    """
    ts_col = _get_ts_col(trades_df)
    if ts_col is None or trades_df is None or len(trades_df) == 0:
        return np.nan
    df = trades_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    s = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    if s.empty:
        return np.nan
    counts = s["pnl_net"].resample("1D").count() if "pnl_net" in s.columns else s.resample("1D").size()
    counts = counts[counts > 0]
    return float(np.median(counts.values)) if counts.size else np.nan


def compute_avg_win_loss_expectancy(trades_df) -> Tuple[float, float, float, float]:
    """
    Calcule avg_win, avg_loss, win_rate et expectancy_par_trade à partir de `pnl_net`.
    Returns:
        (avg_win, avg_loss, win_rate, expectancy_per_trade)
    """
    if trades_df is None or "pnl_net" not in trades_df.columns or len(trades_df) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    pnl = _to_float_series(trades_df["pnl_net"]).dropna()
    if pnl.empty:
        return (np.nan, np.nan, np.nan, np.nan)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    avg_win = float(wins.mean()) if wins.size else np.nan
    avg_loss = float(losses.mean()) if losses.size else np.nan  # négatif
    win_rate = float((pnl > 0.0).mean())
    expectancy = float(pnl.mean())
    return (avg_win, avg_loss, win_rate, expectancy)


def compute_intra_day_hold_time_median(trades_df) -> float:
    """
    Durée médiane (en minutes) de maintien des positions intraday.
    Nécessite colonnes 'entry_time' et 'exit_time'. Sinon, NaN.
    """
    if trades_df is None or not {"entry_time", "exit_time"}.issubset(trades_df.columns):
        return np.nan
    dt_entry = pd.to_datetime(trades_df["entry_time"], errors="coerce", utc=True)
    dt_exit  = pd.to_datetime(trades_df["exit_time"],  errors="coerce", utc=True)
    dur = (dt_exit - dt_entry).dt.total_seconds() / 60.0
    dur = dur.replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.median(dur.values)) if dur.size else np.nan


def compute_max_consecutive_losses(trades_df) -> float:
    """
    Nombre maximum de pertes consécutives basé sur `pnl_net`.
    """
    if trades_df is None or "pnl_net" not in trades_df.columns or len(trades_df) == 0:
        return np.nan
    pnl = _to_float_series(trades_df["pnl_net"]).fillna(0.0).values
    streak = max_streak = 0
    for x in pnl:
        if x < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return float(max_streak)


def compute_execution_quality_metrics(trades_df) -> Dict[str, float]:
    """
    Calcule des métriques d'exécution si les colonnes existent.
    Colonnes attendues (optionnelles) :
      - 'fee' (frais par trade, même unité que pnl_net)
      - 'slippage_bps' (par trade)
      - 'fills' et 'orders_sent' (pour fill_ratio)
      - 'spread_bps_captured' (par trade)
      - 'notional' et 'market_quote_volume' (pour participation_rate)
    Returns:
      dict avec clés : fees_as_pct_gross, slippage_bps, fill_ratio,
                       avg_spread_bps_captured, participation_rate_max, capacity_flag
    """
    out = {
        "fees_as_pct_gross": np.nan,
        "slippage_bps": np.nan,
        "fill_ratio": np.nan,
        "avg_spread_bps_captured": np.nan,
        "participation_rate_max": np.nan,
        "capacity_flag": np.nan,
    }
    if trades_df is None or len(trades_df) == 0:
        return out

    df = trades_df.copy()

    # Frais / PnL brut
    if {"fee", "pnl_net"}.issubset(df.columns):
        fee = _to_float_series(df["fee"]).fillna(0.0)
        pnl = _to_float_series(df["pnl_net"]).fillna(0.0)
        gross_profit = pnl[pnl > 0.0].sum()
        gross_loss = abs(pnl[pnl < 0.0].sum())
        gross = gross_profit + gross_loss
        out["fees_as_pct_gross"] = float(fee.sum() / gross) if gross > 0 else np.nan

    # Slippage médian (bps)
    if "slippage_bps" in df.columns:
        sl = _to_float_series(df["slippage_bps"]).dropna()
        out["slippage_bps"] = float(np.median(sl)) if sl.size else np.nan

    # Fill ratio
    if {"fills", "orders_sent"}.issubset(df.columns):
        fills = _to_float_series(df["fills"]).sum()
        sent = _to_float_series(df["orders_sent"]).sum()
        out["fill_ratio"] = float(fills / sent) if sent > 0 else np.nan

    # Spread capturé (bps)
    if "spread_bps_captured" in df.columns:
        sp = _to_float_series(df["spread_bps_captured"]).dropna()
        out["avg_spread_bps_captured"] = float(np.mean(sp)) if sp.size else np.nan

    # Participation au volume (max)
    if {"notional", "market_quote_volume"}.issubset(df.columns):
        pr = _to_float_series(df["notional"]) / _to_float_series(df["market_quote_volume"]).replace(0, np.nan)
        pr = pr.replace([np.inf, -np.inf], np.nan).dropna()
        out["participation_rate_max"] = float(np.nanmax(pr)) if pr.size else np.nan
        # Capacity flag simple: true si > 50bp (0.5%) du volume d'une barre
        out["capacity_flag"] = bool(out["participation_rate_max"] is not np.nan and out["participation_rate_max"] > 0.005)

    return out


def compute_alpha_information_metrics(
    daily_rets: "pd.Series",
    benchmark_daily_rets: "pd.Series" | None = None
) -> Dict[str, float]:
    """
    Calcule alpha_vs_bh et information_ratio si un benchmark quotidien est fourni.
    Args:
        daily_rets: Rendements journaliers de la stratégie.
        benchmark_daily_rets: Rendements journaliers du benchmark (ex: buy&hold BTC).
    Returns:
        dict: {'alpha_vs_bh': float, 'information_ratio': float}
    """
    out = {"alpha_vs_bh": np.nan, "information_ratio": np.nan}
    if daily_rets is None or len(daily_rets) == 0 or benchmark_daily_rets is None:
        return out
    # Alignement des index
    df = pd.DataFrame({"s": daily_rets, "b": benchmark_daily_rets}).dropna()
    if df.shape[0] < 2:
        return out
    active = df["s"] - df["b"]
    te = float(active.std(ddof=1))
    if te <= 0:
        return out
    out["alpha_vs_bh"] = float(df["s"].mean() - df["b"].mean())
    out["information_ratio"] = float(active.mean() / te * np.sqrt(252.0))
    return out

def concentration_top_k_share(daily_rets: "pd.Series", k: int = 5) -> float:
    """
    Part du PnL total expliquée par les k meilleurs jours.
    Si PnL total <= 0, renvoie 1.0 (concentration maximale).
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    vals = np.array(daily_rets, dtype=float)
    total = float(np.sum(vals))
    if total <= 0:
        return 1.0
    topk = float(np.sum(np.sort(vals)[-k:]))
    return float(topk / total)

def _equity_from_daily_returns(daily_rets: "pd.Series", start: float = 1.0) -> "pd.Series":
    """
    Construit une courbe d'equity à partir de rendements journaliers.
    start=1.0 → equity normalisée; retourne une série alignée sur daily_rets.index.
    """
    if daily_rets is None or len(daily_rets) == 0:
        return pd.Series(dtype=float)
    eq = (1.0 + daily_rets.fillna(0.0)).cumprod() * float(start)
    return eq

def compute_cagr(daily_rets: "pd.Series", periods_per_year: float = 252.0) -> float:
    """
    CAGR approximé depuis des retours journaliers.
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    n = float(len(daily_rets))
    gross = float((1.0 + daily_rets.fillna(0.0)).prod())
    if gross <= 0:
        return np.nan
    return float(gross ** (periods_per_year / n) - 1.0)

def compute_calmar_ratio(daily_rets: "pd.Series", periods_per_year: float = 252.0) -> float:
    """
    Calmar = CAGR / |MaxDrawdown| (sur equity reconstruite à partir de daily_rets).
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    eq = _equity_from_daily_returns(daily_rets, start=1.0)
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    max_dd = float(dd.min()) if len(dd) else np.nan
    cagr = compute_cagr(daily_rets, periods_per_year=periods_per_year)
    if not np.isfinite(max_dd) or max_dd >= 0:
        return np.nan
    return float(cagr / abs(max_dd)) if np.isfinite(cagr) else np.nan

def compute_var(daily_rets: "pd.Series", alpha: float = 0.95) -> float:
    """
    VaR (quantile côté pertes) à 95% par défaut; signe négatif attendu si pertes.
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    q = float(np.quantile(daily_rets.dropna().values, 1.0 - alpha))
    return q

def compute_dd_duration_max(daily_rets: "pd.Series") -> float:
    """
    Durée maximale (en jours) passée sous un plus-haut (high watermark).
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    eq = _equity_from_daily_returns(daily_rets)
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    under = (eq < peak).astype(int)
    # longeur max de runs de 1
    max_run = 0
    curr = 0
    for v in under.values:
        if v == 1:
            curr += 1
            max_run = max(max_run, curr)
        else:
            curr = 0
    return float(max_run)

def compute_time_to_recover_median(daily_rets: "pd.Series") -> float:
    """
    Temps médian (en jours) pour revenir à un nouveau plus-haut d'equity.
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    eq = _equity_from_daily_returns(daily_rets)
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    # dates où l'on atteint un nouveau sommet
    new_high = (eq == peak)
    days = []
    last_peak_idx = None
    for i, is_high in enumerate(new_high.values):
        if is_high:
            if last_peak_idx is not None:
                days.append(i - last_peak_idx)
            last_peak_idx = i
    return float(np.median(days)) if len(days) else np.nan

def compute_worst_day(daily_rets: "pd.Series") -> float:
    """
    Pire rendement journalier observé.
    """
    if daily_rets is None or len(daily_rets) == 0:
        return np.nan
    return float(np.min(daily_rets.values))

def compute_trades_per_day_median(trades_df: "pd.DataFrame") -> float:
    """
    Médiane du nombre de trades par jour (via timestamps d'entrée ou génériques).
    """
    if trades_df is None or len(trades_df) == 0:
        return np.nan
    ts_col = _get_ts_col(trades_df)
    if ts_col is None:
        return np.nan
    df = trades_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    counts = df.dropna(subset=[ts_col]).set_index(ts_col).resample("1D").size()
    return float(np.median(counts.values)) if len(counts) else np.nan

def compute_avg_win_loss_and_expectancy(trades_df: "pd.DataFrame") -> Tuple[float, float, float]:
    """
    Calcule avg_win, avg_loss (valeurs positives/négatives en absolu) et expectancy par trade:
        expectancy = win_rate*avg_win - (1-win_rate)*avg_loss
    """
    if trades_df is None or len(trades_df) == 0 or "pnl_net" not in trades_df.columns:
        return np.nan, np.nan, np.nan
    pnl = pd.to_numeric(trades_df["pnl_net"], errors="coerce").dropna()
    if pnl.empty:
        return np.nan, np.nan, np.nan
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) else np.nan
    avg_loss = float(abs(losses.mean())) if len(losses) else np.nan
    wr = float((pnl > 0).mean()) if len(pnl) else np.nan
    if not (np.isfinite(avg_win) and np.isfinite(avg_loss) and np.isfinite(wr)):
        return avg_win, avg_loss, np.nan
    expectancy = wr * avg_win - (1.0 - wr) * avg_loss
    return avg_win, avg_loss, float(expectancy)

def compute_median_hold_time_minutes(trades_df: "pd.DataFrame") -> float:
    """
    Durée médiane (minutes) de rétention d'un trade (exit_time - entry_time).
    """
    if trades_df is None or len(trades_df) == 0:
        return np.nan
    if ("entry_time" not in trades_df.columns) or ("exit_time" not in trades_df.columns):
        return np.nan
    et = pd.to_datetime(trades_df["entry_time"], errors="coerce", utc=True)
    xt = pd.to_datetime(trades_df["exit_time"], errors="coerce", utc=True)
    dt = (xt - et).dt.total_seconds() / 60.0
    dt = pd.to_numeric(dt, errors="coerce").dropna()
    return float(np.median(dt.values)) if len(dt) else np.nan

def compute_fees_as_pct_gross(trades_df: "pd.DataFrame") -> float:
    """
    Frais / PnL brut (|gains| + |pertes|). Renvoie NaN si indéterminé.
    """
    if trades_df is None or len(trades_df) == 0:
        return np.nan
    fees = pd.to_numeric(trades_df.get("fee", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
    pnl = pd.to_numeric(trades_df.get("pnl_net", pd.Series(dtype=float)), errors="coerce").dropna()
    if pnl.empty:
        return np.nan
    gross = float(pnl[pnl > 0].sum() + abs(pnl[pnl < 0].sum()))
    if gross <= 0:
        return np.nan
    return float(fees / gross)

def compute_median_slippage_bps(trades_df: "pd.DataFrame") -> float:
    """
    Slippage médian (bps) si la colonne slippage_bps est présente.
    """
    if trades_df is None or len(trades_df) == 0 or "slippage_bps" not in trades_df.columns:
        return np.nan
    s = pd.to_numeric(trades_df["slippage_bps"], errors="coerce").dropna()
    return float(np.median(s.values)) if len(s) else np.nan

def compute_fill_ratio(trades_df: "pd.DataFrame") -> float:
    """
    Ratio d'exécution = fills / orders_sent (borné à [0,1]).
    """
    if trades_df is None or len(trades_df) == 0:
        return np.nan
    fills = pd.to_numeric(trades_df.get("fills", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    sent = pd.to_numeric(trades_df.get("orders_sent", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    num = float(fills.sum())
    den = float(sent.sum())
    if den <= 0:
        return np.nan
    return float(max(0.0, min(1.0, num / den)))

def compute_avg_spread_bps_captured(trades_df: "pd.DataFrame") -> float:
    """
    Moyenne du spread 'capturé' (bps) (négatif = coût si crosses).
    """
    if trades_df is None or len(trades_df) == 0 or "spread_bps_captured" not in trades_df.columns:
        return np.nan
    s = pd.to_numeric(trades_df["spread_bps_captured"], errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan

def daily_returns_from_logs(df_logs: Optional["pd.DataFrame"], price_col: str = "price", timestamp_col: str = "timestamp") -> "pd.Series":
    """
    Construit des retours journaliers benchmark (ex: BTC/USDC buy&hold) depuis des logs/quotes:
      - price_col: prix last/close,
      - timestamp_col: horodatage.
    """
    if df_logs is None or len(df_logs) == 0 or price_col not in df_logs.columns:
        return pd.Series(dtype=float)
    ts = pd.to_datetime(df_logs[timestamp_col], errors="coerce", utc=True)
    px = pd.to_numeric(df_logs[price_col], errors="coerce")
    df = pd.DataFrame({"ts": ts, "px": px}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    daily_close = df.set_index("ts")["px"].sort_index().resample("1D").last().dropna()
    return daily_close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

def alpha_info_beta_vs_bench(daily_rets: "pd.Series", bench_rets: "pd.Series", periods_per_year: float = 252.0) -> Tuple[float, float, float]:
    """
    Calcule (alpha annualisé, information_ratio, beta) vs un benchmark (ex: BTC buy&hold).
    - alpha = mean(excess) * periods_per_year
    - IR    = alpha / (std(excess)*sqrt(periods_per_year))
    - beta  = cov(strategy, bench) / var(bench)
    """
    if daily_rets is None or len(daily_rets) == 0 or bench_rets is None or len(bench_rets) == 0:
        return np.nan, np.nan, np.nan
    aligned = pd.concat([daily_rets, bench_rets], axis=1, join="inner").dropna()
    if aligned.empty:
        return np.nan, np.nan, np.nan
    s = aligned.iloc[:, 0].values.astype(float)
    b = aligned.iloc[:, 1].values.astype(float)
    excess = s - b
    mu = float(np.mean(excess))
    te = float(np.std(excess, ddof=1)) if len(excess) > 1 else np.nan
    var_b = float(np.var(b, ddof=1)) if len(b) > 1 else np.nan
    cov_sb = float(np.cov(np.vstack([s, b]))[0, 1]) if len(b) > 1 else np.nan
    alpha = mu * periods_per_year if np.isfinite(mu) else np.nan
    ir = (alpha / (te * np.sqrt(periods_per_year))) if (np.isfinite(alpha) and np.isfinite(te) and te > 0) else np.nan
    beta = (cov_sb / var_b) if (np.isfinite(cov_sb) and np.isfinite(var_b) and var_b > 0) else np.nan
    return float(alpha), float(ir), float(beta)

def compute_intraday_consistency_kpis(
    df_trades: "pd.DataFrame",
    df_logs: Optional["pd.DataFrame"] = None,
    *,
    price_col: str = "price",
    timestamp_col: str = "timestamp",
    periods_per_year: float = 365.0,   # <- devient optionnel
    fee_rate: float = 0.0,
    ann_factor: Optional[float] = None,      # <- NOUVEAU
) -> Dict[str, float]:
    """
    API unique de calcul des KPI "consistency-first" ...
    (docstring inchangée)
    """

    # --- Résolution du facteur d'annualisation ---
    if ann_factor is not None:
        _ann = float(ann_factor)
    elif periods_per_year is not None:
        _ann = float(periods_per_year)
    else:
        _ann = 365.0  # défaut intraday

    # Garde-fous d'entrée (bloc existant, inchangé)
    if df_trades is None or len(df_trades) == 0:
        return {
            "sharpe_d_252": np.nan, "sharpe_d_365": np.nan,  # <- ajout sharpe_d_365
            "sortino_d_252": np.nan, "calmar_ratio": np.nan,
            "sortino_d_365": np.nan,
            "return_cagr": np.nan, "information_ratio": np.nan, "alpha_vs_bh": np.nan,
            "beta_to_btc_intraday": np.nan, "mean_daily_return": np.nan, "vol_daily_return": np.nan,
            "max_drawdown": np.nan, "ulcer_index": np.nan, "dd_duration_max": np.nan,
            "worst_day_return": np.nan, "var_95_d": np.nan, "cvar_95_d": np.nan,
            "nb_trades": 0, "trades_per_day": np.nan, "pct_green_days": np.nan,
            "median_daily_pnl": np.nan, "skew_daily_pnl": np.nan,
            "win_rate": np.nan, "avg_win": np.nan, "avg_loss": np.nan, "expectancy_per_trade": np.nan,
            "time_to_recover_median": np.nan, "intra_day_hold_time_median": np.nan,
            "fees_as_pct_gross": np.nan, "slippage_bps": np.nan, "fill_ratio": np.nan,
            "avg_spread_bps_captured": np.nan,
            "profit_factor": np.nan, "top5_share": np.nan,
            "wfcv_corr": np.nan, "oos_perf_ratio": np.nan, "param_distance_from_bounds": np.nan,
            "sensitivity_penalty": np.nan, "regime_consistency_score": np.nan,
            "pairwise_corr_to_others": np.nan, "cluster_label": np.nan,
            "overlap_trade_ratio": np.nan, "capacity_flag": np.nan,
        }

    # 1) KPIs equity/PnL basiques
    sharpe_e, max_dd_e, profit_factor, win_rate = compute_kpis(
        df_trades, periods_per_year=_ann, annualize=True
    )

    # 2) Rendements journaliers issus des trades
    dr = daily_returns_from_trades(df_trades).astype(float)
    mean_d = float(dr.mean()) if len(dr) else np.nan
    vol_d  = float(dr.std(ddof=1)) if len(dr) > 1 else np.nan

    # Aides pour drawdowns / temps de récup / extrêmes
    calmar = compute_calmar_ratio(dr, periods_per_year=_ann)
    cagr   = compute_cagr(dr, periods_per_year=_ann)
    ulcer  = compute_ulcer_index(dr)
    dd_dur = compute_dd_duration_max(dr)
    worst_d = compute_worst_day(dr)
    var95  = compute_var(dr, alpha=0.95)
    cvar95 = compute_cvar(dr, alpha=0.95)

    # Cohérence intraday
    nb_trades = int(pd.to_numeric(df_trades.get("pnl_net", pd.Series(dtype=float)), errors="coerce").dropna().shape[0])
    trades_per_day = compute_trades_per_day_median(df_trades)
    pct_green = compute_percent_green_days(dr)
    med_d     = compute_median_daily_pnl(dr)
    skew_d    = compute_skew_daily_pnl(dr)
    avg_win, avg_loss, expectancy = compute_avg_win_loss_and_expectancy(df_trades)
    ttr_med = compute_time_to_recover_median(dr)
    hold_med = compute_median_hold_time_minutes(df_trades)

    # Qualité d'exécution
    fees_pct_gross = compute_fees_as_pct_gross(df_trades)
    slip_med = compute_median_slippage_bps(df_trades)
    fill_rt  = compute_fill_ratio(df_trades)
    spread_avg = compute_avg_spread_bps_captured(df_trades)

    # Concentration top-K
    top5_share = concentration_top_k_share(dr, k=5)

    # 3) Bench (optionnel)
    bench = daily_returns_from_logs(df_logs, price_col=price_col, timestamp_col=timestamp_col)
    alpha_bh, ir_bh, beta_bh = alpha_info_beta_vs_bench(dr, bench, periods_per_year=_ann)

    # 4) Assemblage — calcule Sharpe aux 2 échelles pour compat’ + cible 365
    sharpe_252 = compute_sharpe_daily_annualized(dr, periods_per_year=252.0)
    sharpe_365 = compute_sharpe_daily_annualized(dr, periods_per_year=_ann)

    sortino_252 = compute_sortino_annualized(dr, periods_per_year=252.0)
    sortino_365 = compute_sortino_annualized(dr, periods_per_year=_ann)

    out = {
        # Performance
        "sharpe_d_252": sharpe_252,
        "sharpe_d_365": sharpe_365,
        "sortino_d_252": sortino_252,   
        "sortino_d_365": sortino_365,
        "calmar_ratio": calmar,
        "return_cagr": cagr,
        "information_ratio": ir_bh,
        "alpha_vs_bh": alpha_bh,
        "beta_to_btc_intraday": beta_bh,
        "mean_daily_return": mean_d,
        "vol_daily_return": vol_d,

        # Risque & pertes extrêmes
        "max_drawdown": max_dd_e,
        "ulcer_index": ulcer,
        "dd_duration_max": dd_dur,
        "worst_day_return": worst_d,
        "var_95_d": var95,
        "cvar_95_d": cvar95,

        # Cohérence intraday
        "nb_trades": nb_trades,
        "trades_per_day": trades_per_day,
        "pct_green_days": pct_green,
        "median_daily_pnl": med_d,
        "skew_daily_pnl": skew_d,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy_per_trade": expectancy,
        "time_to_recover_median": ttr_med,
        "intra_day_hold_time_median": hold_med,

        # Qualité d'exécution
        "fees_as_pct_gross": fees_pct_gross,
        "slippage_bps": slip_med,
        "fill_ratio": fill_rt,
        "avg_spread_bps_captured": spread_avg,

        # Héritage
        "profit_factor": profit_factor,
        "top5_share": top5_share,

        # Placeholders
        "wfcv_corr": np.nan, "oos_perf_ratio": np.nan, "param_distance_from_bounds": np.nan,
        "sensitivity_penalty": np.nan, "regime_consistency_score": np.nan,
        "pairwise_corr_to_others": np.nan, "cluster_label": np.nan,
        "overlap_trade_ratio": np.nan, "capacity_flag": np.nan,
    }
    return out
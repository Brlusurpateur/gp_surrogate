# AutoML Trading Strategy Research Platform

Personal research project for **systematic trading strategy discovery** on crypto markets.

The goal of this platform is not “just a bot”, but a full **research and AutoML pipeline**:
from data ingestion and backtesting to Bayesian optimization, scoring, portfolio construction, and (experimental) live trading.

> Focus: intraday / mid-frequency strategies on crypto (e.g. BTCUSDT), with realistic execution, risk management, and interpretable model-based search.

---

## High-Level Overview

This project implements an end-to-end workflow:

1. **Data layer**
   - Fetch OHLCV and order book data from an exchange API.
   - Persist market data and backtest results into a **SQLite** database (logs, trades, KPIs, adaptive decisions).

2. **Strategy & backtesting engine**
   - Feature engineering on price & microstructure (RSI, EMA, ATR, VWAP, regimes…).
   - Vectorized backtesting engine with **realistic OCO execution** (TP/SL, slippage, spread, fees, early-stop).

3. **Metrics & evaluation**
   - Rich library of **risk & performance KPIs** (Sharpe/Sortino/Calmar, MDD, Ulcer, VaR/CVaR, execution quality, capacity…).
   - Daily PnL reconstruction, consistency metrics (% green days, median/skew), portfolio-level analytics.

4. **AutoML & model-based search**
   - Structured **hyperparameter search space** for strategies.
   - Surrogate modeling (tree-based + SHAP) for global understanding.
   - **Bayesian Optimization** with a **Gaussian Process + trust-region**, for sequential, data-efficient exploration.

5. **Selection & portfolio construction**
   - Multi-criteria filtering with SOFT/HARD thresholds and Pareto non-dominated front.
   - Ranking and selection of **diversified strategy portfolios** under correlation & risk constraints.

6. **Dashboard & live**
   - Streamlit dashboard for KPI exploration and strategy inspection.
   - Experimental **live runner** with global risk manager and Telegram alerts.

---

## Key Features

- **End-to-end research pipeline** (Python + SQLite) from historical data to live-like execution.
- **Realistic backtesting**:
  - OCO (TP/SL), fees, slippage, spread-captured, timeouts, early-stop rules.
  - Risk-based position sizing (% of capital at risk per trade).
- **Rich KPI engine**:
  - Performance: Sharpe, Sortino, Calmar, CAGR, Profit Factor…
  - Risk: max drawdown (size & duration), Ulcer Index, VaR/CVaR, tail ratio, concentration of PnL on top-K days.
  - Execution: fees as % of gross PnL, slippage in bps, spread captured, participation in market volume (“capacity”).
  - Consistency: % green days, median & skew of daily PnL, median holding time, max consecutive losses.
- **AutoML bayesian loop**:
  - Hyperparameter domain explicitly defined for all strategy knobs.
  - Surrogate model (tree-based) + **SHAP** for interpretability and feature importance.
  - **Gaussian-Process-based Bayesian Optimization** with trust-region and anti-clustering to propose new strategies.
  - Walk-forward cross-validation to evaluate the GP on a time-ordered setting.
- **Multi-criteria selection & portfolio**:
  - SOFT/HARD thresholds on KPIs (consistency-first criteria).
  - Ranking via percentile-based scoring and weighted scores.
  - Pareto non-dominated set and equal-weight portfolio under max-correlation constraints.
- **Tooling & ops**:
  - Centralized logging (console + rotating files).
  - Config/thresholds via `.env` + config module, with **adaptive thresholds** based on historical data.
  - Streamlit dashboard for exploration.
  - Live runner with global risk manager and Telegram alerts.

---

## Architecture

At a high level, the project is split into the following components:

- **Data & infra**
  - `api_mainnet.py` — exchange API client (testnet/mainnet), OHLCV fetching.
  - `backtest_db_manager.py` — SQLite schema init/migrations, inserts (logs, trades, KPIs), market data storage.
  - `logging_setup.py` — global logging configuration.
  - `config.py` — global thresholds, hard/soft KPIs, Sharpe cap, scoring weights, adaptive threshold resolver.
  - `hyperparam_domain.py` — search space definition for strategy hyperparameters.

- **Features & indicators**
  - `indicators_core.py` — EMA/RSI/MACD/ATR, VWAP, volatility proxies, regime tagging (trend/range/vol).
  - `init_orderbook_features.py` — order book snapshot → features (spread, bid/ask imbalance, VWAP weights).

- **Strategy & backtesting**
  - `strategy_params_core.py` — structured strategy parameter class (Numba jitclass + log dtype).
  - `scalping_signal_engine.py` — multi-timeframe signal engine (RSI/EMA/MACD/ATR, regimes, volume, confidence scores).
  - `oco_trade_engine.py` — OCO simulation engine (TP/SL, slippage, spread-bps captured, risk-based sizing).
  - `scalping_backtest_engine.py` — vectorized backtest loop + early-stop rules, logging of trade and step-level info.

- **Metrics & analytics**
  - `metrics_core.py` — centralized risk/perf KPI computation (time series, daily PnL, execution quality, alpha vs benchmark).
  - `strategy_portfolio.py` — portfolio of strategies (correlation matrix, beta to benchmark, portfolio Sharpe).

- **Orchestration & AutoML**
  - `backtest_interface_code.py` — single backtest runner (parameters → backtest → KPIs → DB).
  - `parallel_backtest_launcher.py` — multiprocess orchestration of backtest batches, DB inserts, KPI upserts.
  - `export_good_iteration.py` — multi-criteria filtering, soft/hard thresholds, Pareto front, pools GOOD/BEST.
  - `surrogate_modeling.py` — surrogate model (tree-based), SHAP computation, UMAP embeddings for diagnostics.
  - `gp_driver_utils.py` — GP/BO utilities: trust-region adaptation, candidate sampling, ranking, telemetry.
  - `gp_driver.py` — Gaussian Process training, walk-forward CV, candidate suggestion via acquisition + trust-region.
  - `automl_backtest_driver.py` — top-level AutoML loop orchestrating GP → backtests → export → surrogate updates.

- **Dashboard & live**
  - `streamlit_dashboard.py` — Streamlit app for browsing KPIs, filtering strategies, inspecting backtests.
  - `telegram.py` — Telegram alerts + configuration checks.
  - `main_live_runner.py` — experimental live runner (global risk manager, position tracking, PnL day, alerts).

---

## Data Model

All persistent data is stored in **SQLite** databases:

- **Backtest DB**
  - `trades` — one row per trade (entry/exit timestamps, side, size, prices, fees, slippage, PnL, holding time…).
  - `logs` — time-series of step-level logs (indicators, signals, equity, reasons for entering / skipping trades…).
  - `kpi_by_backtest` — aggregated KPIs per backtest (Sharpe, Sortino, MDD, Ulcer, capacity flags, validity flags…).
  - `adaptive_decisions` — telemetry for adaptive thresholds and GP quality (WFCV metrics, N_soft/N_hard counts).
  - `market_data` — historical OHLCV (and optionally order book features) per symbol/timeframe.

- **Selection DBs**
  - `good_iterations.db` — pool of “SOFT-pass” strategies (consistency-first constraints).
  - `filtered_full_copy.db` — “HARD-pass” / showroom subset, used for portfolio and reporting.

The schema is **migration-friendly**: new columns or tables can be added without breaking existing data.

---

## AutoML Pipeline

The AutoML loop works roughly as follows:

1. **Collect backtest results**
   - Run many backtests with random or GP-proposed hyperparameters.
   - Store logs, trades, and KPIs in SQLite.

2. **Filter and score**
   - Apply quality filters:
     - Minimal number of active days and daily volatility.
     - SOFT/HARD thresholds on key KPIs (Sharpe, MDD, % green days, Ulcer, concentration, capacity, etc.).
     - Outlier detection with a **Sharpe cap** and flags (`is_valid`, `flag_sharpe_outlier`).
   - Compute multi-criteria scores (percentile ranks + weighted score).

3. **Surrogate modeling & interpretability**
   - Train a surrogate model (e.g. gradient boosting) on `(hyperparameters → Sharpe capped)`.
   - Use **SHAP** to compute feature importances and understand which knobs matter.
   - Optionally embed SHAP vectors via UMAP for visual exploration (clusters, regimes).

4. **Gaussian Process & Bayesian Optimization**
   - Prepare a dataset for the GP (filtered strategies with valid KPIs).
   - Fit a **Gaussian Process** model and evaluate it with **walk-forward cross-validation** (time-ordered splits).
   - Use an acquisition function (Expected Improvement, etc.) plus a **trust-region**:
     - Explore around the best known strategies.
     - Adapt TR radius depending on recent improvement (shrink/expand).
     - Enforce diversity via distance constraints between candidates.

5. **Suggest new strategies**
   - Sample new hyperparameter sets from the GP+TR.
   - Feed them to the parallel backtest launcher.
   - Update databases, surrogate models, and thresholds.

6. **Stop condition**
   - Continue until a target number of GOOD strategies is reached (e.g. 100 SOFT-valid entries) or a max number of trials is hit.

---

## Risk & Execution Modeling

Risk management is implemented at multiple layers:

- **Per-trade**
  - Position size based on a **fixed % of capital at risk**.
  - TP/SL levels determined by volatility (ATR), support/resistance, and minimum profit margin.
  - Early-stop rules (max drawdown, max consecutive losses, max idle steps without trades).

- **Backtest level**
  - Rich KPI evaluation with Sharpe, drawdown metrics, Ulcer, VaR/CVaR, tail ratio, top-K day concentration, etc.
  - Flags for invalid or outlier strategies.

- **Portfolio level**
  - Correlation matrix between strategies.
  - Equal-weight portfolio optimized under max average correlation constraints and targeting consistent Sharpe.

- **Live runner**
  - Global risk manager (max % capital per position, max number of concurrent positions, daily gain/loss thresholds).
  - Telegram alerts on daily PnL thresholds and critical events.

---

## Dashboard

The **Streamlit dashboard** connects to the KPI SQLite database and provides:

- Filters on key metrics (Sharpe, Sortino, MDD, Win rate, % green days, Profit Factor, etc.).
- Sorted tables of candidate strategies.
- Detailed view for a given backtest ID (KPI JSON, potential links to logs/trades).
- Distribution plots and quick exploratory analysis.

This is the main tool to inspect and present the results to humans (including non-technical stakeholders).

---

## Quickstart (high-level)

> The exact commands depend on how you structure the repository; below is an indicative workflow.

1. **Set up environment**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill API keys, DB paths, etc.

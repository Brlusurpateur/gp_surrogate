#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit dashboard pour consulter les KPI standardisés et filtrer les stratégies.

- Lit la table 'kpi_by_backtest' (standard) et, si disponible, la table 'trades' pour nb_trades.
- Filtres côté barre latérale : Sharpe, Sortino, Max DD, Win Rate, Pct Green Days, Profit Factor.
- Classement par KPI (par défaut: sharpe_d_252) et export CSV.
- Détails d'une stratégie (clic sur une ligne) + histogrammes de distributions globales.
- Connexion SQLite idempotente; DB définie par env `DB_PATH` (default: "data/results.db").

Exécution:
    streamlit run streamlit_dashboard.py

Dépendances:
    pip install streamlit pandas altair python-dotenv
"""

from __future__ import annotations
import os
import sqlite3
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from typing import Optional, Tuple, Dict
from dotenv import load_dotenv


# ==============================
# Config / I/O de base
# ==============================

DEFAULT_DB_PATH = os.environ.get("DB_PATH", "data/results.db")

st.set_page_config(
    page_title="Intraday Strategies — KPI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charge d'abord API.env (secrets/paths locaux) puis .env (racine)
if os.path.exists("API.env"):
    load_dotenv("API.env")
load_dotenv()


# ==============================
# Helpers DB
# ==============================

@st.cache_data(show_spinner=False)
def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    """Retourne True si la table SQLite `name` existe dans la base."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
        return cur.fetchone() is not None
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load_kpis(db_path: str) -> pd.DataFrame:
    """
    Charge la table 'kpi_by_backtest' depuis `db_path`.

    Returns
    -------
    pd.DataFrame
        Colonnes standardisées (si calculées) : 'backtest_id', 'iteration',
        'sharpe_d_252', 'sortino_d_252', 'max_drawdown', 'profit_factor',
        'win_rate', 'pct_green_days', 'median_daily_pnl', 'ulcer_index', etc.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        if not _has_table(conn, "kpi_by_backtest"):
            return pd.DataFrame()
        df = pd.read_sql_query("SELECT * FROM kpi_by_backtest", conn)
    # Nettoyage basique (±Inf -> NaN)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


@st.cache_data(show_spinner=False)
def load_trades_counts(db_path: str) -> pd.DataFrame:
    """
    Retourne un DataFrame (backtest_id -> nb_trades) depuis la table 'trades' si dispo.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["backtest_id", "nb_trades"])
    with sqlite3.connect(db_path) as conn:
        if not _has_table(conn, "trades"):
            return pd.DataFrame(columns=["backtest_id", "nb_trades"])
        df = pd.read_sql_query(
            """
            SELECT backtest_id, COUNT(*) AS nb_trades
            FROM trades
            GROUP BY backtest_id
            """,
            conn,
        )
    return df


def _num(x) -> float | None:
    """Cast numérique tolérant (NaN si non convertible)."""
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _apply_filters(df: pd.DataFrame,
                   min_sharpe: float,
                   min_sortino: float,
                   max_dd: float,
                   min_win: float,
                   min_green: float,
                   min_pf: float) -> pd.DataFrame:
    """
    Applique les filtres communs sur le DataFrame KPI.

    Notes
    -----
    - Les colonnes absentes sont ignorées pour le filtre correspondant.
    """
    out = df.copy()
    # Map {col: (op, seuil)}
    rules = {
        "sharpe_d_252": (">=", min_sharpe),
        "sortino_d_252": (">=", min_sortino),
        "max_drawdown": ("<=", max_dd),
        "win_rate": (">=", min_win),
        "pct_green_days": (">=", min_green),
        "profit_factor": (">=", min_pf),
    }
    for col, (op, thr) in rules.items():
        if col not in out.columns:
            continue
        if op == ">=":
            out = out[(out[col].astype(float) >= float(thr)) | (out[col].isna())]  # garde NaN (option)
        elif op == "<=":
            out = out[(out[col].astype(float) <= float(thr)) | (out[col].isna())]
    return out


# ==============================
# UI — Barre latérale
# ==============================

st.sidebar.title("⚙️ Paramètres")
db_path = st.sidebar.text_input("Chemin DB (SQLite)", value=DEFAULT_DB_PATH, help="Chemin du fichier SQLite (ex: data/results.db)")

st.sidebar.markdown("---")
st.sidebar.subheader("Filtres KPI")

min_sharpe = st.sidebar.number_input("Sharpe (min)", value=1.0, step=0.1, format="%.2f")
min_sortino = st.sidebar.number_input("Sortino (min)", value=1.0, step=0.1, format="%.2f")
max_dd = st.sidebar.number_input("Max Drawdown (max, %)", value=20.0, step=1.0, format="%.1f")
min_win = st.sidebar.number_input("Win Rate (min, %)", value=50.0, step=1.0, format="%.1f")
min_green = st.sidebar.number_input("Pct Green Days (min, %)", value=50.0, step=1.0, format="%.1f")
min_pf = st.sidebar.number_input("Profit Factor (min)", value=1.50, step=0.05, format="%.2f")

st.sidebar.markdown("---")
sort_key = st.sidebar.selectbox(
    "Trier par",
    options=[
        "sharpe_d_252", "sortino_d_252", "max_drawdown", "profit_factor",
        "win_rate", "pct_green_days", "median_daily_pnl"
    ],
    index=0
)
ascending = st.sidebar.checkbox("Tri ascendant", value=False)

st.sidebar.markdown("---")
export_btn = st.sidebar.checkbox("Autoriser export CSV", value=True)


# ==============================
# Corps — Chargement & tableaux
# ==============================

st.title("📊 Dashboard — Intraday Strategies (Consistency-First)")

kpi = load_kpis(db_path)
if kpi.empty:
    st.warning(f"Base introuvable ou sans table 'kpi_by_backtest' : {db_path}")
    st.stop()

# Ajoute nb_trades si dispo
tr_counts = load_trades_counts(db_path)
if not tr_counts.empty:
    kpi = kpi.merge(tr_counts, on="backtest_id", how="left")

# Harmonisation unités (%)
for col in ["max_drawdown", "win_rate", "pct_green_days"]:
    if col in kpi.columns:
        kpi[col] = kpi[col].apply(_num)

# Filtres
filtered = _apply_filters(
    kpi,
    min_sharpe=min_sharpe,
    min_sortino=min_sortino,
    max_dd=max_dd / 100.0 if max_dd > 1.5 else max_dd,      # accepte 0.2 ou 20
    min_win=min_win / 100.0 if min_win > 1.5 else min_win,
    min_green=min_green / 100.0 if min_green > 1.5 else min_green,
    min_pf=min_pf
)

# Tri
if sort_key in filtered.columns:
    filtered = filtered.sort_values(sort_key, ascending=ascending, na_position="last")

# Affichage
st.markdown("### 🏆 Sélection filtrée")
show_cols = [c for c in [
    "backtest_id", "iteration", "sharpe_d_252", "sortino_d_252", "max_drawdown",
    "win_rate", "pct_green_days", "profit_factor", "median_daily_pnl", "nb_trades"
] if c in filtered.columns]

st.dataframe(filtered[show_cols], use_container_width=True, height=420)

# Export CSV
if export_btn:
    csv_bytes = filtered[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("💾 Export CSV", data=csv_bytes, file_name="good_strategies_filtered.csv", mime="text/csv")

# ==============================
# Détails stratégie (sélection)
# ==============================

st.markdown("---")
st.markdown("### 🔎 Détails d’une stratégie")

selected_id = st.selectbox(
    "Choisir un backtest_id",
    options=list(filtered["backtest_id"].astype(str).unique()),
    index=0 if len(filtered) else None
)

if selected_id:
    row = filtered[filtered["backtest_id"].astype(str) == str(selected_id)].head(1).to_dict(orient="records")
    if row:
        st.json(row[0])

# ==============================
# Distributions globales
# ==============================

st.markdown("---")
st.markdown("### 📈 Distributions (globales)")

cols_metrics = [c for c in ["sharpe_d_252", "sortino_d_252", "max_drawdown", "win_rate", "profit_factor"] if c in kpi.columns]
ncols = min(3, len(cols_metrics))
if cols_metrics:
    cols = st.columns(ncols)
    for i, col in enumerate(cols_metrics):
        sub = kpi[[col]].rename(columns={col: "value"}).dropna()
        if sub.empty:
            continue
        chart = alt.Chart(sub).mark_bar().encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=40), title=col),
            y=alt.Y("count()", title="Fréquence")
        ).properties(height=220)
        cols[i % ncols].altair_chart(chart, use_container_width=True)
else:
    st.info("Pas de colonnes KPI suffisantes pour tracer des distributions.")


# ==============================
# Footer
# ==============================

st.caption(
    "© Quant Dashboard — lit 'kpi_by_backtest' standardisée. "
    "Ajuste les colonnes/alias si tu as renommé des KPI en amont."
)

"""
====================================================================================
Fichier : backtest_db_manager.py
Objectif : Fonctions de gestion, transformation et diagnostic pour les logs et trades
====================================================================================

Description :
Ce module centralise les outils nécessaires à la création, gestion et extraction 
des **logs de stratégie** et **résultats de trades** dans un contexte de simulation 
ou d’apprentissage par renforcement appliqué au trading algorithmique.

Il fournit une couche d’abstraction cohérente pour :
    - Initialiser une base SQLite propre avec schéma structuré (logs/trades)
    - Nettoyer, convertir et synchroniser des données en DataFrames exploitables
    - Diagnostiquer l’état de la base (intégrité, couverture, vides)
    - Sauvegarder ou recharger des données de marché synchronisées (multi-timeframe)

Contenu principal :
    - `init_sqlite_database(...)`      → structure et création des tables `logs` & `trades`
    - `array_to_clean_dataframe(...)`  → conversion propre des arrays structurés (logs)
    - `save_market_data_to_sqlite(...)` → export multi-intervalle synchronisé en SQL
    - `diagnose_sqlite_backtest_db(...)` → audit complet des backtests en base
    - `is_sqlite_fusion_empty(...)`    → détection d’une base inutilisable ou vide
    - `load_market_data_from_sqlite(...)` → conversion SQL → dictionnaire numpy pour moteurs RL

Cas d’usage :
- Utilisé dans les pipelines de simulation de stratégie, optimisation bayésienne, PPO
- Compatible avec Dash, Streamlit ou tout moteur analytique sur base `good_iterations.db`
- Permet l’analyse rétroactive des signaux, états du moteur et qualité des trades

Références pertinentes :
- Stable-Baselines3 / EvalCallback
- NumPy structured arrays & pandas I/O
- Bases de données expérimentales pour la recherche en algorithmic trading

Contraintes :
- Les timestamps doivent être au format ISO ou `datetime64[ms]`
- `array_to_clean_dataframe()` suppose un champ-clé non nul pour identifier la fin des données
- La table `logs` peut contenir plusieurs centaines de colonnes : attention aux limites SQLite

Auteur : Moncoucut Brandon
Version : Juin 2025
"""

# === Imports fondamentaux ===
import os
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np

def init_sqlite_database(db_path):
    """
    Initialise la base SQLite et crée les tables nécessaires avec support du champ `backtest_id`.
    """
    # ✅ Initialisation idempotente : ne jamais supprimer la DB, assurer le schéma
    # Crée le dossier de la base si besoin (gère le cas d'un chemin sans dossier)
    base_dir = os.path.dirname(db_path) or "."
    os.makedirs(base_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # PRAGMA robustesse & concurrence
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA busy_timeout=5000;")

    # ✅ Table logs : inclut equity/pnl_step/drawdown/volume notionnel + stabilité & régime
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            backtest_id TEXT,
            timestamp TEXT,
            iteration REAL,

            -- Hyperparamètres
            ema_short_period REAL,
            ema_long_period REAL,
            rsi_period REAL,
            rsi_buy_zone REAL,
            rsi_sell_zone REAL,
            rsi_past_lookback REAL,
            atr_tp_multiplier REAL,
            atr_sl_multiplier REAL,
            atr_period REAL,
            macd_signal_period REAL,
            rsi_thresholds_1m REAL,
            rsi_thresholds_5m REAL,
            rsi_thresholds_15m REAL,
            rsi_thresholds_1h REAL,
            ewma_period REAL,
            weight_atr_combined_vol REAL,
            threshold_volume REAL,
            hist_volum_period REAL,
            detect_supp_resist_period REAL,
            trend_period REAL,
            threshold_factor REAL,
            min_qty REAL,
            step_size REAL,
            tick_size REAL,
            min_profit_margin REAL,
            resistance_buffer_margin REAL,
            risk_reward_ratio REAL,
            confidence_score_params REAL,
            signal_weight_bonus REAL,
            penalite_resistance_factor REAL,
            penalite_multi_tf_step REAL,
            override_score_threshold REAL,
            rsi_extreme_threshold REAL,
            signal_pure_threshold REAL,
            signal_pure_weight REAL,
            tp_mode TEXT,
            enable_early_stop REAL,
            mdd_stop_pct REAL,
            max_consecutive_losses REAL,
            max_steps_no_trade REAL,

            -- Indicateurs et stratégie
            atr_15m REAL,
            avg_atr REAL,
            required_confirmations REAL,
            adjusted_rsi_1m REAL,
            adjusted_rsi_5m REAL,
            adjusted_rsi_15m REAL,
            adjusted_rsi_1h REAL,
            rsi_1m REAL,
            rsi_1m_threshold REAL,
            rsi_5m REAL,
            rsi_5m_threshold REAL,
            rsi_15m REAL,
            rsi_15m_threshold REAL,
            rsi_1h REAL,
            rsi_1h_threshold REAL,
            total_confirmations REAL,
            multi_tf_confirmation REAL,
            ema_short REAL,
            ema_long REAL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            atr REAL,
            volatilite_combinee REAL,
            volume_signal REAL,
            support REAL,
            resistance REAL,
            vwap REAL,
            prix_actuel REAL,
            volume_avg REAL,
            rsi_past REAL,
            condition_marche REAL,
            confidence_score_real REAL,
            score_signal_pure REAL,
            signal_ema REAL,
            signal_rsi REAL,
            signal_macd REAL,
            penalite_resistance REAL,
            penalite_multi_tf REAL,
            distance_resistance_pct REAL,
            raison_refus TEXT,
            buy_reasons TEXT,
            refus_critique_structurel REAL,
            refus_critique_technique REAL,
            nb_signaux_valides REAL,
            position REAL,

            -- ===== Nouveaux champs equity/exécution =====
            balance REAL,
            equity REAL,
            pnl_step REAL,
            drawdown REAL,
            mkt_quote_vol REAL,
            price REAL,

            -- ===== Stabilité & Régimes =====
            signal_stability_var REAL,
            signal_stability_drift REAL,
            regime_tag TEXT
        )
    """)

    # Table trades enrichie : fees, slippage_bps, spread_bps_captured, fills, orders_sent, notional
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            backtest_id TEXT,
            iteration INTEGER,
            num_trade INTEGER,
            entry_time TEXT,
            entry_price REAL,
            exit_price REAL,
            pnl_net REAL,
            status TEXT,
            quantity REAL,
            tp REAL,
            sl REAL,
            exit_time TEXT,
            balance REAL,

            -- KPI à la volée (héritage, facultatif)
            sharpe_ratio REAL,
            max_drawdown REAL,
            profit_factor REAL,
            win_rate REAL,

            -- Exécution / audit
            fee REAL,
            slippage_bps REAL,
            spread_bps_captured REAL,
            fills REAL,
            orders_sent REAL,
            notional REAL,

            -- Early stop (facultatif)
            early_stop_reason TEXT,
            early_stop_step REAL
        )
    """)

    # ✅ Table KPI par backtest (clé: backtest_id + iteration)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kpi_by_backtest (
            backtest_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,

            -- Performance (annualisée/journalière)
            sharpe_d_252 REAL,          -- historique (marchés actions 252j)
            sharpe_d_365 REAL,          -- crypto 24/7 (recommandé)
            sortino_d_252 REAL,
            calmar_ratio REAL,
            mean_daily_return REAL,
            vol_daily_return REAL,
            return_cagr REAL,
            profit_factor REAL,

            -- Benchmark (optionnel)
            alpha_vs_bh REAL,
            information_ratio REAL,

            -- Risque & extrêmes
            max_drawdown REAL,
            ulcer_index REAL,
            dd_duration_max REAL,
            worst_day_return REAL,
            var_95_d REAL,
            cvar_95_d REAL,
            max_consecutive_losses REAL,
            tail_ratio REAL,

            -- Consistency
            nb_trades INTEGER,
            trades_per_day REAL,
            pct_green_days REAL,
            median_daily_pnl REAL,
            skew_daily_pnl REAL,
            win_rate REAL,
            avg_win REAL,
            avg_loss REAL,
            expectancy_per_trade REAL,
            time_to_recover_median REAL,
            intra_day_hold_time_median REAL,
            n_unique_days INTEGER,      -- nouveau : # de jours avec activité (utile pour valider Sharpe)
            std_daily REAL,             -- nouveau : écart-type des PnL journaliers (garde-fou Sharpe)

            -- Validation & Flags (anti-outliers / qualité échantillon)
            is_valid INTEGER DEFAULT 1,          -- 1 = KPI valide ; 0 = invalide (trop court, std_daily trop faible, etc.)
            invalid_reason TEXT,                 -- motif d’invalidation (diagnostic)
            flag_sharpe_outlier INTEGER DEFAULT 0, -- 1 = Sharpe jugé aberrant (cap/winsorize)

            -- Exécution (optionnel)
            fees_as_pct_gross REAL,
            slippage_bps REAL,
            fill_ratio REAL,
            avg_spread_bps_captured REAL,
            participation_rate_max REAL,
            capacity_flag REAL,

            -- Anti-lottery
            top5_share REAL,

            PRIMARY KEY (backtest_id, iteration)
        )
    """)

    # ✅ Nouvelle table pour la sélection de portefeuille (avant les index)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS selected_strategies (
            backtest_id TEXT NOT NULL,
            ts_selected INTEGER NOT NULL,       -- epoch ms au moment de la sélection
            batch_id INTEGER NOT NULL,          -- identifiant de batch (itération driver)
            portfolio_sharpe REAL,              -- Sharpe du portefeuille au moment de la sélection
            mean_pairwise_corr REAL,            -- Corr absolue moyenne sur le set retenu
            individual_sharpe REAL,             -- Sharpe individuel de la stratégie
            PRIMARY KEY (backtest_id, batch_id)
        )
    """)

    # ✅ Index utiles (créés une seule fois, après toutes les tables)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kpi_backtest ON kpi_by_backtest(backtest_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_backtest ON logs(backtest_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_backtest ON trades(backtest_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_selected_on_batch ON selected_strategies(batch_id)")

    conn.commit()
    conn.close()

def _ensure_columns(conn, table: str, expected: dict[str, str]) -> None:
    """
    Ajoute en douceur les colonnes manquantes à `table` selon `expected` (nom -> type SQL).
    Idempotent : ne casse pas si colonne déjà présente.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}  # set des noms de colonnes
    to_add = [(name, sqltype) for name, sqltype in expected.items() if name not in existing]
    for name, sqltype in to_add:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}")
    conn.commit()


def migrate_sqlite_schema(db_path: str) -> None:
    """
    Migration idempotente : garantit que les tables `logs`, `trades`, `kpi_by_backtest`
    disposent de toutes les colonnes attendues (ajout via ALTER TABLE si besoin).
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        # LOGS : colonnes additionnelles
        logs_expected = {
            # equity/exécution
            "balance": "REAL",
            "equity": "REAL",
            "pnl_step": "REAL",
            "drawdown": "REAL",
            "mkt_quote_vol": "REAL",
            "price": "REAL",
            # stabilité & régime
            "signal_stability_var": "REAL",
            "signal_stability_drift": "REAL",
            "regime_tag": "TEXT",
        }
        _ensure_columns(conn, "logs", logs_expected)

        # TRADES : colonnes additionnelles
        trades_expected = {
            "fee": "REAL",
            "slippage_bps": "REAL",
            "spread_bps_captured": "REAL",
            "fills": "REAL",
            "orders_sent": "REAL",
            "notional": "REAL",
        }

        _ensure_columns(conn, "trades", trades_expected)

        # KPI : créer table si absente
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='kpi_by_backtest'")
        if cur.fetchone() is None:
            # Si la table n'existe pas, on relance l'init qui la crée
            conn.close()
            init_sqlite_database(db_path)
            return
        
        # KPI : colonnes additionnelles (migration douce)
        # - 'sharpe_d_365' pour annualisation crypto 24/7
        # - 'n_unique_days' et 'std_daily' pour valider la robustesse du Sharpe
        kpi_expected = {
            # Annualisation crypto
            "sharpe_d_365": "REAL",

            # Validation & Flags (anti-outliers / qualité échantillon)
            "is_valid": "INTEGER DEFAULT 1",
            "invalid_reason": "TEXT",
            "flag_sharpe_outlier": "INTEGER DEFAULT 0",

            # Diagnostics pour robustifier le Sharpe
            "n_unique_days": "INTEGER",
            "std_daily": "REAL",
        }

        _ensure_columns(conn, "kpi_by_backtest", kpi_expected)


def array_to_clean_dataframe(struct_array, key_field=None):
    """
    Convertit un tableau structuré NumPy en DataFrame, en supprimant les lignes vides
    et en convertissant automatiquement les colonnes datetime en chaînes.

    Args:
        struct_array (np.ndarray): Tableau structuré (log_array, trade_history, etc.)
        key_field (str, optional): Champ utilisé pour détecter la fin (ex: 'num_trade', 'entry_price', 'timestamp')

    Returns:
        pd.DataFrame: DataFrame propre, sans les lignes non remplies, et datetime converti.
    """

    # Auto-détection du champ clé
    if key_field is None:
        possible_keys = ['num_trade', 'entry_price', 'timestamp']
        for key in possible_keys:
            if key in struct_array.dtype.names:
                key_field = key
                break
        if key_field is None:
            raise ValueError("❌ Impossible de déterminer un champ clé. Veuillez en fournir un via key_field.")

    # Détection des lignes valides (champ ≠ 0 ou vide)
    filled_mask = struct_array[key_field] != 0
    valid_rows = np.where(filled_mask)[0]

    if len(valid_rows) == 0:
        return pd.DataFrame(struct_array[:0])  # Aucun log/trade valide

    last_valid_index = valid_rows[-1] + 1
    trimmed_array = struct_array[:last_valid_index]

    # Conversion en DataFrame
    df = pd.DataFrame(trimmed_array)

    # Conversion automatique des datetime64 en str ISO
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return df

def insert_logs_from_array(db_path: str, backtest_id: str, iteration: int, log_array: np.ndarray) -> int:
    """
    Insère (append) un `log_array` structuré dans la table `logs`.
    - Convertit via `array_to_clean_dataframe`, ajoute backtest_id & iteration.
    - Conserve uniquement les colonnes qui existent en base (tolérant aux versions).
    Returns:
        int: nombre de lignes insérées.
    """
    if log_array is None or len(log_array) == 0:
        return 0
    df = array_to_clean_dataframe(log_array, key_field="timestamp").copy()
    if df.empty:
        return 0
    df["backtest_id"] = backtest_id
    df["iteration"] = iteration

    with sqlite3.connect(db_path) as conn:
        cols_sql = pd.read_sql_query("PRAGMA table_info(logs)", conn)["name"].tolist()
        keep = [c for c in df.columns if c in cols_sql]
        if not keep:
            return 0
        df[keep].to_sql("logs", conn, if_exists="append", index=False)
        return len(df)


def insert_trades_from_array(db_path: str, backtest_id: str, iteration: int, trade_array: np.ndarray) -> int:
    """
    Insère (append) un `trade_array` structuré dans la table `trades`.
    - Convertit via `array_to_clean_dataframe`, ajoute backtest_id & iteration.
    - Conserve uniquement les colonnes présentes en base.
    Returns:
        int: nombre de lignes insérées.
    """
    if trade_array is None or len(trade_array) == 0:
        return 0
    df = array_to_clean_dataframe(trade_array, key_field="num_trade").copy()
    if df.empty:
        return 0
    df["backtest_id"] = backtest_id
    df["iteration"] = iteration

    # Harmonise les datetime en iso string
    for col in ("entry_time", "exit_time"):
        if col in df.columns and not pd.api.types.is_string_dtype(df[col]):
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    with sqlite3.connect(db_path) as conn:
        cols_sql = pd.read_sql_query("PRAGMA table_info(trades)", conn)["name"].tolist()
        keep = [c for c in df.columns if c in cols_sql]
        if not keep:
            return 0
        df[keep].to_sql("trades", conn, if_exists="append", index=False)
        return len(df)


def upsert_kpis(db_path: str, backtest_id: str, iteration: int, kpi: dict) -> None:
    """
    UPSERT robuste (INSERT OR REPLACE) dans `kpi_by_backtest`.

    Notes
    -----
    - Utilise la clé primaire (backtest_id, iteration).
    - Tolère les clés KPI manquantes : elles seront NULL.
    - Suppose que `migrate_sqlite_schema`/`init_sqlite_database` ont déjà créé la table.
    """
    if not isinstance(kpi, dict):
        return

    with sqlite3.connect(db_path) as conn:
        # Colonnes de la table (et ordre)
        cols_sql = pd.read_sql_query("PRAGMA table_info(kpi_by_backtest)", conn)["name"].tolist()

        # Build la ligne complète avec les colonnes présentes
        payload = {c: None for c in cols_sql}
        payload.update(kpi)
        payload["backtest_id"] = str(backtest_id)
        payload["iteration"] = int(iteration)

        # Prépare la requête INSERT OR REPLACE
        cols = ", ".join(cols_sql)
        placeholders = ", ".join(["?"] * len(cols_sql))
        sql = f"INSERT OR REPLACE INTO kpi_by_backtest ({cols}) VALUES ({placeholders})"
        values = tuple(payload.get(c, None) for c in cols_sql)

        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()

def ensure_dict_arrays_consistency(data_dict):
    """
    S'assure que chaque entrée du dictionnaire est un np.ndarray de même longueur.
    Convertit automatiquement les scalaires ou chaînes en tableaux broadcastés.
    """
    lengths = [len(v) for v in data_dict.values() if isinstance(v, np.ndarray)]
    if not lengths:
        raise ValueError("Aucune colonne valide détectée.")
    target_len = lengths[0]

    for key, val in data_dict.items():
        if isinstance(val, np.ndarray):
            if len(val) != target_len:
                raise ValueError(f"La colonne '{key}' a une longueur différente ({len(val)} vs {target_len})")
        else:
            #print(f"[⚠️ Fix] Clé '{key}' convertie en np.ndarray broadcasté")
            data_dict[key] = np.full(target_len, val)

    return data_dict

def save_market_data_to_sqlite(data_dict, db_path, base_interval='5m'):
    """
    Enregistre les données synchronisées sur les timestamps du base_interval
    dans une base SQLite (une table par timeframe).
    """
    if base_interval not in data_dict:
        raise ValueError(f"⛔ Base interval '{base_interval}' manquant dans data_dict.")

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # ✅ Base temporelle (data_5m, par défaut)
    base_df = data_dict[base_interval].copy()
    if "timestamp" in base_df.columns:
        base_df["timestamp"] = pd.to_datetime(base_df["timestamp"], errors="coerce", utc=True)
    base_df = base_df.sort_values("timestamp").reset_index(drop=True)
    base_timestamps = base_df[["timestamp"]]

    with sqlite3.connect(db_path) as conn:
        for interval, df in data_dict.items():
            if df.empty:
                print(f"⚠️ Données vides pour l'intervalle {interval}, rien à sauvegarder.")
                continue

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)

            if interval == base_interval:
                df_synced = df.copy()
            else:
                # ✅ Synchronisation temporelle sur la base
                df_synced = pd.merge_asof(
                    base_timestamps, df,
                    on='timestamp', direction='backward'
                )

            df_synced["interval"] = interval
            table_name = f"data_{interval}"
            df_synced.to_sql(table_name, conn, if_exists="replace", index=False)

            print(f"✅ Données synchronisées '{interval}' enregistrées dans '{table_name}'")


def diagnose_sqlite_backtest_db(db_path):
    """
    Affiche un diagnostic détaillé de la base SQLite fusionnée :
    - nombre de backtests uniques
    - backtests avec trades, sans trades, sans logs
    """
    if not os.path.exists(db_path):
        print("❌ Base SQLite introuvable.")
        return

    with sqlite3.connect(db_path) as conn:
        try:
            trades = pd.read_sql_query("SELECT backtest_id FROM trades", conn)
            logs = pd.read_sql_query("SELECT backtest_id FROM logs", conn)
        except Exception as e:
            print(f"❌ Erreur lors de la lecture des tables : {e}")
            return

    trades_ids = set(trades["backtest_id"].dropna().unique())
    logs_ids = set(logs["backtest_id"].dropna().unique())
    all_ids = trades_ids.union(logs_ids)

    print(f"\n🔍 Diagnostic de la base '{os.path.basename(db_path)}'")
    print(f"📦 Total backtest_id uniques : {len(all_ids)}")
    print(f"📊 Avec trades               : {len(trades_ids)}")
    print(f"🗒️  Avec logs                : {len(logs_ids)}")
    print(f"🚫 Aucune donnée (vide)      : {len(all_ids - trades_ids - logs_ids)}")

    # Optionnel : liste les IDs sans trade
    only_logs = logs_ids - trades_ids
    if only_logs:
        print(f"ℹ️  Backtests avec logs mais sans trade :")
        for bid in sorted(only_logs):
            print(f"   - {bid}")

    if len(all_ids) == 0:
        print("⚠️ Aucun backtest trouvé.")
    elif len(trades_ids) == 0:
        print("⚠️ Aucun trade détecté. Vérifie la stratégie ou la période.")
    else:
        print("✅ La base contient des backtests exploitables.\n")

def is_sqlite_fusion_empty(db_path):
    """
    Vérifie si la base SQLite fusionnée contient au moins un backtest exploitable.
    Retourne True si vide, False sinon.
    """
    if not os.path.exists(db_path):
        return True  # base absente = vide

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Vérifie la présence de trades
        cursor.execute("SELECT COUNT(*) FROM trades")
        n_trades = cursor.fetchone()[0]
        
        # Vérifie la présence de logs (facultatif mais utile)
        cursor.execute("SELECT COUNT(*) FROM logs")
        n_logs = cursor.fetchone()[0]

    return (n_trades == 0 and n_logs == 0)

def load_market_data_from_sqlite(path_data, intervals=["1m", "5m", "15m", "1h"]):
    """
    Charge les données de marché depuis une base SQLite et les renvoie
    sous forme de dictionnaire {intervalle: dict[colonne: np.ndarray]}.

    Args:
        path_data (str): Chemin vers la base SQLite contenant les données historiques.
        intervals (list): Liste des intervalles à charger.

    Returns:
        dict: Dictionnaire contenant, pour chaque intervalle, un dict numpy-ready.
    """
    data = {}
    cols_to_convert = [
        "timestamp", "open", "high", "low", "close", "volume"
    ]

    with sqlite3.connect(path_data) as conn:
        for interval in intervals:
            table_name = f"data_{interval}"
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

                # Conversion en dict de np.ndarray
                data_np = {col: df[col].to_numpy() for col in cols_to_convert if col in df.columns}

                data[interval] = data_np
                print(f"✅ Données chargées pour '{interval}' ({len(df)} lignes)")

            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de '{table_name}': {e}")
                data[interval] = {}  # vide si erreur

    return data

def ensure_selected_table_exists(db_path: str):
    """
    Garantit l'existence de la table selected_strategies (idempotent).
    """
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS selected_strategies (
                backtest_id TEXT NOT NULL,
                ts_selected INTEGER NOT NULL,
                batch_id INTEGER NOT NULL,
                portfolio_sharpe REAL,
                mean_pairwise_corr REAL,
                individual_sharpe REAL,
                PRIMARY KEY (backtest_id, batch_id)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_selected_on_batch ON selected_strategies(batch_id)")
        conn.commit()


def insert_selected_strategies(db_path: str,
                               selected_ids: list[str],
                               portfolio_sharpe: float,
                               mean_pairwise_corr: float,
                               batch_id: int,
                               stats_df: pd.DataFrame):
    """
    Insère la sélection courante dans la table selected_strategies.
    - selected_ids : liste ordonnée (portefeuille retenu)
    - stats_df     : DataFrame 'stats' renvoyée par select_strategy_portfolio (contient sharpe individuel)
    """
    import time
    ts_now_ms = int(time.time() * 1000)

    # map sharpe individuel par id
    indiv = {}
    if isinstance(stats_df, pd.DataFrame) and "backtest_id" in stats_df and "individual_sharpe" in stats_df:
        indiv = dict(zip(stats_df["backtest_id"].astype(str), stats_df["individual_sharpe"].astype(float)))

    rows = [(sid, ts_now_ms, batch_id, float(portfolio_sharpe), float(mean_pairwise_corr), float(indiv.get(str(sid), np.nan)))
            for sid in selected_ids]

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS selected_strategies (
                backtest_id TEXT NOT NULL,
                ts_selected INTEGER NOT NULL,
                batch_id INTEGER NOT NULL,
                portfolio_sharpe REAL,
                mean_pairwise_corr REAL,
                individual_sharpe REAL,
                PRIMARY KEY (backtest_id, batch_id)
            )
        """)
        c.executemany("""
            INSERT OR REPLACE INTO selected_strategies
            (backtest_id, ts_selected, batch_id, portfolio_sharpe, mean_pairwise_corr, individual_sharpe)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()

def has_market_data_coverage(db_path: str, intervals: list[str], min_lookback_days: int, base_col: str = "timestamp") -> bool:
    """
    Vérifie que chaque table data_{interval} existe et couvre au moins `min_lookback_days`.
    Retourne True si tout est OK, False sinon.
    """
    if not os.path.exists(db_path):
        return False

    try:
        with sqlite3.connect(db_path) as conn:
            for itv in intervals:
                tbl = f"data_{itv}"
                try:
                    df = pd.read_sql_query(f"SELECT MIN({base_col}) AS tmin, MAX({base_col}) AS tmax FROM {tbl}", conn)
                except Exception:
                    return False
                if df.empty or df["tmin"].isna().all() or df["tmax"].isna().all():
                    return False
                tmin = pd.to_datetime(df.loc[0, "tmin"], errors="coerce", utc=True)
                tmax = pd.to_datetime(df.loc[0, "tmax"], errors="coerce", utc=True)
                if pd.isna(tmin) or pd.isna(tmax):
                    return False
                if (tmax - tmin).days < min_lookback_days:
                    return False
        return True
    except Exception:
        return False

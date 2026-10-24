"""
Creates the football database schema: leagues, teams, matches,
match_stats, odds, predictions, bets.
Safe to re-run: uses CREATE TABLE IF NOT EXISTS throughout.
"""

import sqlite3
import os

DB_PATH = os.path.join("..", "data", "over25.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Drop the old test table from Phase 1, no longer needed
cursor.execute("DROP TABLE IF EXISTS test_table")

# --- leagues ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS leagues (
    league_id INTEGER PRIMARY KEY AUTOINCREMENT,
    div_code TEXT UNIQUE NOT NULL,   -- e.g. 'E0', 'E1', 'E2', 'E3'
    league_name TEXT NOT NULL        -- e.g. 'Premier League'
)
""")

# --- teams ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT UNIQUE NOT NULL   -- standardised name
)
""")

# --- matches ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,       -- date + league + home + away
    league_id INTEGER NOT NULL,
    season TEXT NOT NULL,            -- e.g. '2024/25'
    match_date TEXT NOT NULL,        -- ISO format YYYY-MM-DD
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_goals INTEGER,
    away_goals INTEGER,
    total_goals INTEGER,
    over_2_5 INTEGER,                 -- 1 if total_goals >= 3, else 0
    result TEXT,                      -- 'H', 'D', 'A'
    behind_closed_doors INTEGER DEFAULT 0,  -- COVID flag, Phase 5
    FOREIGN KEY (league_id) REFERENCES leagues(league_id),
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
)
""")

# --- match_stats ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS match_stats (
    match_id TEXT PRIMARY KEY,
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_on_target INTEGER,
    away_shots_on_target INTEGER,
    home_corners INTEGER,
    away_corners INTEGER,
    home_fouls INTEGER,
    away_fouls INTEGER,
    home_yellow INTEGER,
    away_yellow INTEGER,
    home_red INTEGER,
    away_red INTEGER,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
)
""")

# --- odds (benchmarking/backtesting only, per Phase 2 notes) ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS odds (
    match_id TEXT PRIMARY KEY,
    avg_over_2_5 REAL,
    avg_under_2_5 REAL,
    bfe_over_2_5 REAL,         -- Betfair Exchange, only 2024/25-2025/26
    bfe_under_2_5 REAL,
    bfe_close_over_2_5 REAL,   -- closing price, where available
    bfe_close_under_2_5 REAL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
)
""")

# --- predictions ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_prob_over_2_5 REAL NOT NULL,
    prediction_date TEXT NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
)
""")

# --- bets ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS bets (
    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    prediction_id INTEGER,
    stake REAL NOT NULL,
    odds_taken REAL NOT NULL,
    bet_side TEXT NOT NULL,       -- 'over' or 'under'
    commission REAL NOT NULL,
    result TEXT,                  -- 'win', 'loss', 'pending'
    profit_loss REAL,
    bet_date TEXT NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
)
""")

conn.commit()

# Confirm what got created
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables created:", [row[0] for row in cursor.fetchall()])

conn.close()
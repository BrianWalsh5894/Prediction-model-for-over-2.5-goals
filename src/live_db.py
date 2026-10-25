"""
Database tables for live paper-tracking (Phase 11). Safe to call every run:
only adds what is missing, never drops or changes existing data.

  live_fixtures  - every upcoming fixture a prediction was made for
  predictions    - (existing table) + the two model components
  live_prices    - the Betfair prices YOU entered, with the time checked
  bets           - (existing table) paper bets; result 'pending'/'win'/'loss'/'void'
"""

import os
import sqlite3

DB_PATH = os.path.join("..", "data", "over25.db")
LIVE_DIR = os.path.join("..", "data", "live")


def connect():
    conn = sqlite3.connect(DB_PATH)
    ensure_live_schema(conn)
    return conn


def _columns(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def ensure_live_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_fixtures (
            match_id TEXT PRIMARY KEY,
            season TEXT NOT NULL,
            league_name TEXT NOT NULL,
            match_date TEXT NOT NULL,
            kickoff_time TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'scheduled'   -- 'scheduled', 'played', 'void'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_prices (
            match_id TEXT PRIMARY KEY,                 -- one price per match: the first you enter
            over_odds REAL NOT NULL,
            under_odds REAL NOT NULL,
            checked_at TEXT NOT NULL,                  -- when you looked at Betfair
            entered_at TEXT NOT NULL
        )
    """)
    for col, typ in [("p_poisson", "REAL"), ("p_dc", "REAL"),
                     ("lam_home", "REAL"), ("lam_away", "REAL")]:
        if col not in _columns(conn, "predictions"):
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
    for col, typ in [("expected_value", "REAL"), ("settled_at", "TEXT")]:
        if col not in _columns(conn, "bets"):
            conn.execute(f"ALTER TABLE bets ADD COLUMN {col} {typ}")
    conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_predictions_match_version
                    ON predictions(match_id, model_version)""")
    conn.commit()

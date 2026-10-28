"""
Shared functions for importing football-data.co.uk CSVs into the
database. Used by both the historical import (import_data.py) and
the weekly current-season refresh (update_current_season.py).
"""

import pandas as pd

DIVISIONS = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    # v3-eu-upper (European top divisions) - same football-data.co.uk format
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    # their second divisions - history for promoted teams (not predicted live)
    "SP2": "La Liga 2",
    "I2": "Serie B",
    "D2": "2. Bundesliga",
    "F2": "Ligue 2",
    # National League - tracked live only (v3-conf experiment, own group); never
    # exported into any model's training data by clean_data.py
    "EC": "National League",
}


def season_code_to_display(code):
    start = code[:2]
    end = code[2:]
    return f"20{start}/{end}"


def get_or_create_league(cursor, div_code, league_name):
    cursor.execute("SELECT league_id FROM leagues WHERE div_code = ?", (div_code,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute(
        "INSERT INTO leagues (div_code, league_name) VALUES (?, ?)",
        (div_code, league_name)
    )
    return cursor.lastrowid


def standard_team_name(cursor, raw_name):
    cursor.execute("SELECT standard_name FROM team_name_map WHERE raw_name = ?", (raw_name,))
    row = cursor.fetchone()
    return row[0] if row else raw_name


def get_or_create_team(cursor, raw_name):
    name = standard_team_name(cursor, raw_name)
    cursor.execute("SELECT team_id FROM teams WHERE team_name = ?", (name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute("INSERT INTO teams (team_name) VALUES (?)", (name,))
    return cursor.lastrowid


def build_match_id(date_str, div_code, home_raw, away_raw):
    date_obj = pd.to_datetime(date_str, dayfirst=True)
    iso_date = date_obj.strftime("%Y-%m-%d")
    match_id = f"{iso_date}_{div_code}_{home_raw}_{away_raw}"
    return match_id, iso_date


def safe_int(value):
    if pd.isna(value):
        return None
    return int(value)


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def import_dataframe(conn, cursor, div_code, season_display, df, covid_flag=0):
    league_id = get_or_create_league(cursor, div_code, DIVISIONS[div_code])

    matches_rows = []
    stats_rows = []
    odds_rows = []

    for _, row in df.iterrows():
        home_raw = row["HomeTeam"]
        away_raw = row["AwayTeam"]
        if pd.isna(home_raw) or pd.isna(away_raw) or pd.isna(row.get("Date")):
            continue

        match_id, iso_date = build_match_id(row["Date"], div_code, home_raw, away_raw)
        home_team_id = get_or_create_team(cursor, home_raw)
        away_team_id = get_or_create_team(cursor, away_raw)

        home_goals = safe_int(row.get("FTHG"))
        away_goals = safe_int(row.get("FTAG"))
        total_goals = home_goals + away_goals if home_goals is not None and away_goals is not None else None
        over_2_5 = int(total_goals >= 3) if total_goals is not None else None

        matches_rows.append((
            match_id, league_id, season_display, iso_date,
            home_team_id, away_team_id, home_goals, away_goals,
            total_goals, over_2_5, row.get("FTR"), covid_flag
        ))

        stats_rows.append((
            match_id,
            safe_int(row.get("HS")), safe_int(row.get("AS")),
            safe_int(row.get("HST")), safe_int(row.get("AST")),
            safe_int(row.get("HC")), safe_int(row.get("AC")),
            safe_int(row.get("HF")), safe_int(row.get("AF")),
            safe_int(row.get("HY")), safe_int(row.get("AY")),
            safe_int(row.get("HR")), safe_int(row.get("AR")),
        ))

        odds_rows.append((
            match_id,
            safe_float(row.get("Avg>2.5")), safe_float(row.get("Avg<2.5")),
            safe_float(row.get("BFE>2.5")), safe_float(row.get("BFE<2.5")),
            safe_float(row.get("BFEC>2.5")), safe_float(row.get("BFEC<2.5")),
        ))

    cursor.executemany("""
        INSERT OR REPLACE INTO matches
        (match_id, league_id, season, match_date, home_team_id, away_team_id,
         home_goals, away_goals, total_goals, over_2_5, result, behind_closed_doors)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, matches_rows)

    cursor.executemany("""
        INSERT OR REPLACE INTO match_stats
        (match_id, home_shots, away_shots, home_shots_on_target, away_shots_on_target,
         home_corners, away_corners, home_fouls, away_fouls,
         home_yellow, away_yellow, home_red, away_red)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, stats_rows)

    cursor.executemany("""
        INSERT OR REPLACE INTO odds
        (match_id, avg_over_2_5, avg_under_2_5, bfe_over_2_5, bfe_under_2_5,
         bfe_close_over_2_5, bfe_close_under_2_5)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, odds_rows)

    return len(matches_rows)
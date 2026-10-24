"""
Imports all 36 raw CSVs into the SQLite database: leagues, teams,
matches, match_stats, odds. Safe to re-run — uses a unique match_id
(date + league + home + away) so re-importing won't create duplicates.
Missing values from the source are stored as NULL, never guessed or filled.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join("..", "data", "over25.db")
RAW_DIR = os.path.join("..", "data", "raw")

DIVISIONS = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
}

# season code -> display season, e.g. "1718" -> "2017/18"
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
    return row[0] if row else raw_name  # fall back to raw name if not mapped


def get_or_create_team(cursor, raw_name):
    name = standard_team_name(cursor, raw_name)
    cursor.execute("SELECT team_id FROM teams WHERE team_name = ?", (name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute("INSERT INTO teams (team_name) VALUES (?)", (name,))
    return cursor.lastrowid


def build_match_id(date_str, div_code, home_raw, away_raw):
    # date_str comes in as DD/MM/YYYY from the source; normalise to YYYY-MM-DD
    date_obj = pd.to_datetime(date_str, dayfirst=True)
    iso_date = date_obj.strftime("%Y-%m-%d")
    match_id = f"{iso_date}_{div_code}_{home_raw}_{away_raw}"
    return match_id, iso_date


def safe_int(value):
    # Returns None (-> NULL in SQLite) for missing values, never a fabricated number
    if pd.isna(value):
        return None
    return int(value)


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def import_file(conn, cursor, div_code, season_code, filepath):
    league_id = get_or_create_league(cursor, div_code, DIVISIONS[div_code])
    season_display = season_code_to_display(season_code)
    covid_flag = 1 if season_code == "2021" else 0  # 2020/21, fully behind closed doors

    df = pd.read_csv(filepath)

    matches_rows = []
    stats_rows = []
    odds_rows = []

    for _, row in df.iterrows():
        home_raw = row["HomeTeam"]
        away_raw = row["AwayTeam"]
        if pd.isna(home_raw) or pd.isna(away_raw) or pd.isna(row.get("Date")):
            continue  # skip any genuinely blank row

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


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    total_imported = 0
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".csv") or fname.startswith("_"):
            continue
        div_code, season_code = fname.replace(".csv", "").split("_")
        filepath = os.path.join(RAW_DIR, fname)
        n = import_file(conn, cursor, div_code, season_code, filepath)
        conn.commit()
        print(f"{fname}: {n} matches imported")
        total_imported += n

    print(f"\nTotal matches imported: {total_imported}")
    conn.close()


if __name__ == "__main__":
    main()
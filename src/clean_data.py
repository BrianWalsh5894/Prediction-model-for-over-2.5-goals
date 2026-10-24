"""
Phase 4 - Data Cleaning / Validation.

Runs the Phase 4 checklist against the database:
  - standardised dates (one format, one timezone assumption: none needed —
    football-data.co.uk gives match dates only, no kickoff times, so there
    is no timezone ambiguity to resolve)
  - standardised league and team names
  - missing/incorrect values; score verification
  - duplicate matches
  - missing shots/odds handled explicitly (NULL, never filled)
  - target: total_goals, over_2_5
  - exports a cleaned, joined dataset to data/clean/matches_clean.csv

Safe to re-run any time (e.g. after the weekly update script). Exits with
a non-zero code and prints every failure if any check fails, so it can be
dropped into the weekly pipeline (Phase 11) as a guard before features are
rebuilt.
"""

import sqlite3
import os
import sys
import csv

DB_PATH = os.path.join("..", "data", "over25.db")
CLEAN_DIR = os.path.join("..", "data", "clean")
CLEAN_CSV = os.path.join(CLEAN_DIR, "matches_clean.csv")

EXPECTED_MATCHES = {
    "Premier League": 380,
    "Championship": 552,
    "League One": 552,
    "League Two": 552,
}

# Seasons known to legitimately deviate from the expected match count.
# Anything else deviating is treated as a real problem, not assumed fine.
KNOWN_SHORT_SEASONS = {
    ("League One", "2019/20"): "curtailed by COVID, season ended early on points-per-game",
    ("League Two", "2019/20"): "curtailed by COVID, season ended early on points-per-game",
}

# The current, in-progress season (Phase 9's "LIVE" season) is always
# short until the campaign finishes - that's expected, not a defect.
LIVE_SEASON = "2026/27"


def fail(msg):
    print(f"FAIL: {msg}")
    return False


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    ok = True

    # --- 1. Standardise dates -------------------------------------------
    # Dates are stored as match_date TEXT in ISO YYYY-MM-DD at import time
    # (import_helpers.build_match_id). No kickoff time is present in the
    # source data, so there is no timezone to standardise - just the
    # calendar date of the fixture.
    cur.execute("""
        SELECT COUNT(*) FROM matches
        WHERE match_date NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
    """)
    bad_dates = cur.fetchone()[0]
    if bad_dates:
        ok = fail(f"{bad_dates} match_date values are not in ISO YYYY-MM-DD format")
    else:
        print("OK: all match_date values are ISO YYYY-MM-DD")

    # --- 2. Standardise league and team names ---------------------------
    cur.execute("SELECT COUNT(*) FROM leagues")
    n_leagues = cur.fetchone()[0]
    if n_leagues != 4:
        ok = fail(f"expected 4 leagues, found {n_leagues}")
    else:
        print("OK: 4 leagues present")

    cur.execute("SELECT team_name, COUNT(*) FROM teams GROUP BY team_name HAVING COUNT(*) > 1")
    dup_team_names = cur.fetchall()
    if dup_team_names:
        ok = fail(f"duplicate team_name rows: {dup_team_names}")
    else:
        print("OK: no duplicate team names in teams table")

    # Every raw name used anywhere in the raw CSVs should resolve through
    # team_name_map to a name that actually exists in teams - i.e. no raw
    # name was silently left unmapped under a second, slightly different
    # team_id for the same real-world club. We can't re-derive "real
    # world club" automatically, so this check instead confirms every
    # mapped standard_name exists in teams (catches typos in the map).
    cur.execute("""
        SELECT DISTINCT standard_name FROM team_name_map
        WHERE standard_name NOT IN (SELECT team_name FROM teams)
    """)
    orphan_map = cur.fetchall()
    if orphan_map:
        ok = fail(f"team_name_map entries pointing to a team that doesn't exist: {orphan_map}")
    else:
        print("OK: every team_name_map standard_name exists in teams")

    # --- 3. Missing/incorrect values; verify scores ---------------------
    for col in ("home_goals", "away_goals", "total_goals", "over_2_5", "result", "match_date"):
        cur.execute(f"SELECT COUNT(*) FROM matches WHERE {col} IS NULL")
        n_null = cur.fetchone()[0]
        if n_null:
            ok = fail(f"{n_null} matches have NULL {col} (core field, should never be missing)")
    if ok:
        print("OK: no NULLs in core match fields (goals, target, result, date)")

    cur.execute("""
        SELECT COUNT(*) FROM matches
        WHERE (home_goals > away_goals AND result != 'H')
           OR (home_goals < away_goals AND result != 'A')
           OR (home_goals = away_goals AND result != 'D')
    """)
    bad_results = cur.fetchone()[0]
    if bad_results:
        ok = fail(f"{bad_results} matches where result (FTR) disagrees with the scoreline")
    else:
        print("OK: result (H/D/A) agrees with scoreline for every match")

    # Row-count sanity check per league/season, same idea as Phase 3's
    # check but now flags anything not already known/documented.
    cur.execute("""
        SELECT l.league_name, m.season, COUNT(*)
        FROM matches m JOIN leagues l ON m.league_id = l.league_id
        GROUP BY l.league_name, m.season
        ORDER BY l.league_name, m.season
    """)
    unexplained = []
    for league_name, season, count in cur.fetchall():
        expected = EXPECTED_MATCHES[league_name]
        if count == expected:
            continue
        if season == LIVE_SEASON:
            continue  # in-progress season, expected to be short
        reason = KNOWN_SHORT_SEASONS.get((league_name, season))
        if reason:
            continue  # known and documented
        unexplained.append((league_name, season, count, expected))
    if unexplained:
        ok = fail(f"unexplained match-count deviations (not current season, not known short season): {unexplained}")
    else:
        print("OK: all match-count deviations are either the in-progress 2026/27 season or the documented 2019/20 League One/Two curtailment")

    # --- 4. Duplicates ----------------------------------------------------
    cur.execute("SELECT match_id, COUNT(*) FROM matches GROUP BY match_id HAVING COUNT(*) > 1")
    dup_matches = cur.fetchall()
    if dup_matches:
        ok = fail(f"duplicate match_id rows: {dup_matches}")
    else:
        print("OK: no duplicate match_id rows (PK already enforces this)")

    # Same fixture (date + two teams) appearing under two different
    # match_ids would slip past the PK check (e.g. a team-name typo
    # creating a second team_id). Check for that case explicitly.
    cur.execute("""
        SELECT match_date, home_team_id, away_team_id, COUNT(*)
        FROM matches
        GROUP BY match_date, home_team_id, away_team_id
        HAVING COUNT(*) > 1
    """)
    dup_fixtures = cur.fetchall()
    if dup_fixtures:
        ok = fail(f"same fixture (date/home/away) appears more than once: {dup_fixtures}")
    else:
        print("OK: no fixture appears twice under different match_ids")

    # --- 5. Missing shots/odds handled explicitly ------------------------
    # These are allowed to be NULL (no fabricated values). Just report the
    # coverage per season so gaps are visible and documented, not silent.
    cur.execute("""
        SELECT m.season,
               SUM(CASE WHEN ms.home_shots IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN o.avg_over_2_5 IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN o.bfe_over_2_5 IS NULL THEN 1 ELSE 0 END),
               COUNT(*)
        FROM matches m
        LEFT JOIN match_stats ms ON m.match_id = ms.match_id
        LEFT JOIN odds o ON m.match_id = o.match_id
        GROUP BY m.season ORDER BY m.season
    """)
    print("\nShots/odds NULL coverage by season (home_shots_null, avg_odds_null, bfe_odds_null, total):")
    for row in cur.fetchall():
        print("  ", row)

    # --- 6. Target: total_goals, over_2_5 --------------------------------
    cur.execute("SELECT COUNT(*) FROM matches WHERE total_goals != home_goals + away_goals")
    bad_total = cur.fetchone()[0]
    if bad_total:
        ok = fail(f"{bad_total} matches where total_goals != home_goals + away_goals")
    else:
        print("OK: total_goals = home_goals + away_goals for every match")

    cur.execute("""
        SELECT COUNT(*) FROM matches
        WHERE (total_goals >= 3 AND over_2_5 != 1)
           OR (total_goals < 3 AND over_2_5 != 0)
    """)
    bad_target = cur.fetchone()[0]
    if bad_target:
        ok = fail(f"{bad_target} matches where over_2_5 disagrees with total_goals >= 3")
    else:
        print("OK: over_2_5 target correctly derived from total_goals for every match")

    # --- 7. Export cleaned, joined dataset -------------------------------
    os.makedirs(CLEAN_DIR, exist_ok=True)
    cur.execute("""
        SELECT
            m.match_id, m.season, l.league_name,
            m.match_date, ht.team_name AS home_team, at.team_name AS away_team,
            m.home_goals, m.away_goals, m.total_goals, m.over_2_5, m.result,
            m.behind_closed_doors,
            ms.home_shots, ms.away_shots, ms.home_shots_on_target, ms.away_shots_on_target,
            ms.home_corners, ms.away_corners, ms.home_fouls, ms.away_fouls,
            ms.home_yellow, ms.away_yellow, ms.home_red, ms.away_red,
            o.avg_over_2_5, o.avg_under_2_5,
            o.bfe_over_2_5, o.bfe_under_2_5,
            o.bfe_close_over_2_5, o.bfe_close_under_2_5
        FROM matches m
        JOIN leagues l ON m.league_id = l.league_id
        JOIN teams ht ON m.home_team_id = ht.team_id
        JOIN teams at ON m.away_team_id = at.team_id
        LEFT JOIN match_stats ms ON m.match_id = ms.match_id
        LEFT JOIN odds o ON m.match_id = o.match_id
        ORDER BY m.match_date, l.league_name
    """)
    rows = cur.fetchall()
    columns = [d[0] for d in cur.description]

    with open(CLEAN_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    print(f"\nExported {len(rows)} rows to {CLEAN_CSV}")

    conn.close()

    if not ok:
        print("\nPhase 4 cleaning FAILED one or more checks - see FAIL lines above.")
        sys.exit(1)
    print("\nPhase 4 cleaning: all checks passed.")


if __name__ == "__main__":
    main()

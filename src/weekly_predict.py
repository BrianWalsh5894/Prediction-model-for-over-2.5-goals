"""
Phase 11 - weekly step 1: predict upcoming League One / League Two fixtures
with the frozen model v3.

What it does, in order:
  1. Downloads the latest 2026/27 results into the database
     (update_current_season.py) and re-runs the Phase 4 checks/export
     (clean_data.py). Stops if any check fails.
  2. Downloads upcoming fixtures from football-data.co.uk/fixtures.csv
     (or reads a file you give it with --fixtures-file).
  3. Builds the same 103 features for each fixture, using only matches
     already played (build_features.py, unchanged logic).
  4. Predicts P(Over 2.5) with model v3 and stores every prediction in the
     database - one per match, the first one made. Re-running never
     overwrites an existing prediction.
  5. Writes data/live/odds_entry_<date>.csv for you to fill in the Betfair
     Over/Under 2.5 prices, then run weekly_bets.py on it.

Run from inside src/:
    python weekly_predict.py
    python weekly_predict.py --fixtures-file path\\to\\fixtures.csv   (if the download fails)
"""

import os
import sys
import argparse
from datetime import date, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402
import model_v3 as mv  # noqa: E402
import live_db  # noqa: E402
from build_features import build_feature_table  # noqa: E402
from import_helpers import DIVISIONS, build_match_id  # noqa: E402

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
CURRENT_SEASON = "2026/27"
LEAGUE_TO_DIV = {v: k for k, v in DIVISIONS.items()}


def min_back_odds(p, commission, min_edge):
    """Smallest decimal odds at which backing this side clears MIN_EDGE:
    EV = p*(o-1)*(1-c) - (1-p) >= min_edge  ->  o >= 1 + (min_edge + 1 - p) / (p*(1-c))"""
    return 1 + (min_edge + 1 - p) / (p * (1 - commission))


def refresh_results():
    import update_current_season
    import clean_data
    print("Updating 2026/27 results from football-data.co.uk...")
    update_current_season.main()
    print("\nRe-running Phase 4 checks + export...")
    try:
        clean_data.main()
    except SystemExit as e:
        if e.code:
            print("\nSTOP: Phase 4 checks failed - fix the data before predicting.")
            sys.exit(1)


def load_fixtures(path, conn, today):
    if path:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    else:
        print(f"\nDownloading {FIXTURES_URL}...")
        try:
            raw = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig")
        except Exception:
            raw = pd.read_csv(FIXTURES_URL, encoding="latin-1")
    raw.columns = [c.strip() for c in raw.columns]
    live_divs = [LEAGUE_TO_DIV[lg] for lg in config.LIVE_LEAGUES]
    raw = raw[raw["Div"].isin(live_divs)].dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    name_map = dict(conn.execute("SELECT raw_name, standard_name FROM team_name_map").fetchall())
    rows = []
    for r in raw.itertuples(index=False):
        r = r._asdict()
        match_id, iso_date = build_match_id(r["Date"], r["Div"], r["HomeTeam"], r["AwayTeam"])
        if pd.Timestamp(iso_date).date() < today:
            continue
        rows.append({
            "match_id": match_id, "season": CURRENT_SEASON,
            "league_name": DIVISIONS[r["Div"]], "match_date": pd.Timestamp(iso_date),
            "kickoff_time": r.get("Time"),
            "home_team": name_map.get(r["HomeTeam"], r["HomeTeam"]),
            "away_team": name_map.get(r["AwayTeam"], r["AwayTeam"]),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-file", help="use a local fixtures CSV instead of downloading")
    ap.add_argument("--skip-refresh", action="store_true",
                    help="don't download results / re-run clean_data (testing only)")
    ap.add_argument("--today", help="pretend today is YYYY-MM-DD (testing only)")
    args = ap.parse_args()

    today = date.fromisoformat(args.today) if args.today else date.today()
    if not args.skip_refresh:
        refresh_results()

    conn = live_db.connect()
    bundle = mv.load_bundle()

    fixtures = load_fixtures(args.fixtures_file, conn, today)
    clean = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    if len(fixtures):
        fixtures = fixtures[~fixtures["match_id"].isin(clean["match_id"])]   # already played
    if fixtures.empty:
        print(f"\nNo upcoming {' / '.join(config.LIVE_LEAGUES)} fixtures found from {today}.")
        return

    # Only each team's NEXT match. A later fixture would be predicted without
    # the result of the match before it (and the first prediction is the one
    # kept), so it waits for a later run, after that result is in.
    fixtures = fixtures.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    seen, keep = set(), []
    for r in fixtures.itertuples(index=False):
        keep.append(r.home_team not in seen and r.away_team not in seen)
        seen |= {r.home_team, r.away_team}
    later = fixtures[[not k for k in keep]]
    fixtures = fixtures[keep].reset_index(drop=True)
    if len(later):
        print(f"\n{len(later)} later fixture(s) held back until their teams' previous match is played "
              f"(next one from {later['match_date'].min():%Y-%m-%d}) - run this again before them.")
    unknown = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    unknown = [t for t in unknown if t not in set(clean["home_team"]) | set(clean["away_team"])]
    if unknown:
        print(f"\nNOTE: teams with no history (rated as league average): {unknown}")

    # Features: append fixtures (no result) to the played matches and build
    # the full table - every feature is shifted, so fixtures only see the past.
    fx_rows = fixtures.drop(columns=["kickoff_time"]).assign(behind_closed_doors=0)
    feats = build_feature_table(pd.concat([clean, fx_rows], ignore_index=True))
    fx_feats = feats[feats["match_id"].isin(fixtures["match_id"])].reset_index(drop=True)

    preds = mv.predict_v3(bundle, fx_feats, clean, today)
    preds = preds.merge(fixtures[["match_id", "kickoff_time"]], on="match_id")
    preds = preds.sort_values(["match_date", "league_name", "home_team"]).reset_index(drop=True)

    # --- Store (first prediction per match wins; never overwritten) ---------
    now = datetime.now().isoformat(timespec="seconds")
    new, kept = 0, 0
    for r in preds.itertuples(index=False):
        conn.execute("""INSERT OR IGNORE INTO live_fixtures
                        (match_id, season, league_name, match_date, kickoff_time, home_team, away_team)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                     (r.match_id, r.season, r.league_name, r.match_date.strftime("%Y-%m-%d"),
                      None if pd.isna(r.kickoff_time) else str(r.kickoff_time), r.home_team, r.away_team))
        cur = conn.execute("""INSERT OR IGNORE INTO predictions
                              (match_id, model_version, predicted_prob_over_2_5, prediction_date,
                               p_poisson, p_dc, lam_home, lam_away)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                           (r.match_id, mv.MODEL_VERSION, float(r.p_over), now, float(r.p_poisson),
                            float(r.p_dc), float(r.poisson_lam_home), float(r.poisson_lam_away)))
        new += cur.rowcount
        kept += 1 - cur.rowcount
    conn.commit()

    # Use the STORED probability (the first one) for the odds sheet
    stored = pd.read_sql_query(
        "SELECT match_id, predicted_prob_over_2_5 AS p_over FROM predictions WHERE model_version = ?",
        conn, params=(mv.MODEL_VERSION,))
    sheet = preds.drop(columns=["p_over"]).merge(stored, on="match_id")
    c, e = config.COMMISSION_RATE, config.MIN_EDGE
    sheet = pd.DataFrame({
        "match_id": sheet["match_id"],
        "date": sheet["match_date"].dt.strftime("%Y-%m-%d"),
        "time": sheet["kickoff_time"],
        "league": sheet["league_name"],
        "home": sheet["home_team"],
        "away": sheet["away_team"],
        "model_p_over": sheet["p_over"].round(4),
        "fair_odds_over": (1 / sheet["p_over"]).round(2),
        "fair_odds_under": (1 / (1 - sheet["p_over"])).round(2),
        "bet_over_if_odds_at_least": min_back_odds(sheet["p_over"], c, e).round(2),
        "bet_under_if_odds_at_least": min_back_odds(1 - sheet["p_over"], c, e).round(2),
        "betfair_over_odds": "",
        "betfair_under_odds": "",
        "checked_at": "",
    })
    os.makedirs(live_db.LIVE_DIR, exist_ok=True)
    out = os.path.join(live_db.LIVE_DIR, f"odds_entry_{today.isoformat()}.csv")
    if os.path.exists(out):
        out = os.path.join(live_db.LIVE_DIR, f"odds_entry_{today.isoformat()}_{datetime.now():%H%M%S}.csv")
    sheet.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(f"\n{len(preds)} fixtures: {new} new predictions stored, {kept} already predicted earlier (kept as-is).")
    print(sheet[["date", "league", "home", "away", "model_p_over",
                 "bet_over_if_odds_at_least", "bet_under_if_odds_at_least"]].to_string(index=False))
    print(f"\nNext: open {out}, fill in betfair_over_odds, betfair_under_odds and checked_at "
          f"(e.g. 2026-10-31 14:05), save, then run:\n    python weekly_bets.py {out}")


if __name__ == "__main__":
    main()

"""
Phase 11 - weekly step 1: predict upcoming fixtures for one league group
(default 'lower' = League One / League Two) with that group's frozen model.

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
    python weekly_predict.py                     (default group: lower = League One + Two)
    python weekly_predict.py --group lower
    python weekly_predict.py --group upper       (Premier League + Championship)
    python weekly_predict.py --group eu_upper    (La Liga, Serie A, Bundesliga, Ligue 1)
    python weekly_predict.py --fixtures-file path\\to\\fixtures.csv   (if the download fails)
League groups and their live models are set in config/config.py (LEAGUE_GROUPS).
"""

import os
import sys
import argparse
from datetime import date, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402
import league_groups  # noqa: E402
import live_db  # noqa: E402
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


def load_fixtures(path, conn, today, leagues):
    if path:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    else:
        print(f"\nDownloading {FIXTURES_URL}...")
        try:
            raw = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig")
        except Exception:
            raw = pd.read_csv(FIXTURES_URL, encoding="latin-1")
    raw.columns = [c.strip() for c in raw.columns]
    live_divs = [LEAGUE_TO_DIV[lg] for lg in leagues]
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


def _opt(row, name):
    """Optional model component (not every model has a Poisson/DC split)."""
    v = getattr(row, name, None)
    return None if v is None or pd.isna(v) else float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="lower",
                    help="league group from config.LEAGUE_GROUPS (default: lower = League One + Two)")
    ap.add_argument("--fixtures-file", help="use a local fixtures CSV instead of downloading")
    ap.add_argument("--skip-refresh", action="store_true",
                    help="don't download results / re-run clean_data (testing only)")
    ap.add_argument("--today", help="pretend today is YYYY-MM-DD (testing only)")
    ap.add_argument("--only-date", help="only predict fixtures on this date (YYYY-MM-DD) - use with --force "
                                        "when the last Saturday's results aren't published yet")
    ap.add_argument("--force", action="store_true",
                    help="predict even if last Saturday's results aren't in (only if there were no games)")
    args = ap.parse_args()

    today = date.fromisoformat(args.today) if args.today else date.today()
    leagues = league_groups.group(args.group)["leagues"]
    mv = league_groups.live_model(args.group)
    if not args.skip_refresh:
        refresh_results()
        if hasattr(mv, "refresh_data"):        # e.g. latest Understat xG for v3-eu-upper
            print("\nRefreshing extra data for this model...")
            mv.refresh_data()
    print(f"Group '{args.group}': {', '.join(leagues)} - live model {mv.MODEL_VERSION}")
    conn = live_db.connect()
    bundle = mv.load_bundle()

    fixtures = load_fixtures(args.fixtures_file, conn, today, leagues)
    clean = pd.read_csv(getattr(mv, "CLEAN_CSV", CLEAN_CSV), parse_dates=["match_date"])

    # Guard: the most recent Saturday's results must already be in the
    # database, otherwise a team's prediction would miss its last match.
    # (Football-Data usually publishes weekend results Sunday night /
    # Monday.) --force skips this, e.g. for a weekend with no games.
    last_sat = pd.Timestamp(today) - pd.Timedelta(days=(today.weekday() - 5) % 7 or 7)
    live = clean[clean["league_name"].isin(leagues) & (clean["season"] == CURRENT_SEASON)]
    latest = live["match_date"].max()
    if not args.force and (pd.isna(latest) or latest < last_sat):
        print(f"\nSTOP: latest {' / '.join(leagues)} result in the database is "
              f"{latest:%Y-%m-%d}, but there should be results from Saturday {last_sat:%Y-%m-%d}. "
              f"Football-Data hasn't published them yet - try again later (usually Sunday night or "
              f"Monday). If there were genuinely no games that Saturday, add --force.")
        sys.exit(1)
    if len(fixtures):
        fixtures = fixtures[~fixtures["match_id"].isin(clean["match_id"])]   # already played
    if len(fixtures) and args.only_date:
        fixtures = fixtures[fixtures["match_date"] == pd.Timestamp(args.only_date)]
    if len(fixtures):
        # Never predict a match that has already kicked off (fixtures.csv times are UK time)
        now = datetime.now()
        ko = pd.to_datetime(fixtures["match_date"].dt.strftime("%Y-%m-%d") + " "
                            + fixtures["kickoff_time"].fillna("23:59").astype(str), errors="coerce")
        started = ko.notna() & (ko <= now) & (fixtures["match_date"].dt.date == now.date())
        if started.any():
            print(f"\n{int(started.sum())} fixture(s) skipped - already kicked off.")
        fixtures = fixtures[~started]
    if fixtures.empty:
        print(f"\nNo upcoming {' / '.join(leagues)} fixtures found from {today}.")
        return

    # Midweek safety net: a team with an earlier predicted fixture whose
    # result still isn't in the database is held back until it is.
    awaiting = pd.read_sql_query(
        """SELECT match_id, home_team, away_team FROM live_fixtures
           WHERE status = 'scheduled' AND match_date < ? AND match_date >= ?""",
        conn, params=(today.isoformat(), (today - pd.Timedelta(days=7)).isoformat()))
    awaiting = awaiting[~awaiting["match_id"].isin(clean["match_id"])]
    waiting_teams = set(awaiting["home_team"]) | set(awaiting["away_team"])
    blocked = fixtures["home_team"].isin(waiting_teams) | fixtures["away_team"].isin(waiting_teams)
    if blocked.any():
        print(f"\n{int(blocked.sum())} fixture(s) held back - a team's previous result isn't published yet: "
              f"{sorted(waiting_teams & (set(fixtures['home_team']) | set(fixtures['away_team'])))}")
        fixtures = fixtures[~blocked]
        if fixtures.empty:
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
    feats = mv.prepare_features(pd.concat([clean, fx_rows], ignore_index=True))
    fx_feats = feats[feats["match_id"].isin(fixtures["match_id"])].reset_index(drop=True)

    preds = mv.predict(bundle, fx_feats, clean, today)
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
                           (r.match_id, mv.MODEL_VERSION, float(r.p_over), now,
                            _opt(r, "p_poisson"), _opt(r, "p_dc"),
                            _opt(r, "poisson_lam_home"), _opt(r, "poisson_lam_away")))
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
    out = os.path.join(live_db.LIVE_DIR, f"odds_entry_{args.group}_{today.isoformat()}.csv")
    if os.path.exists(out):
        out = os.path.join(live_db.LIVE_DIR, f"odds_entry_{args.group}_{today.isoformat()}_{datetime.now():%H%M%S}.csv")
    sheet.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(f"\n{len(preds)} fixtures: {new} new predictions stored, {kept} already predicted earlier (kept as-is).")
    print(sheet[["date", "league", "home", "away", "model_p_over",
                 "bet_over_if_odds_at_least", "bet_under_if_odds_at_least"]].to_string(index=False))
    print(f"\nNext: open {out}, fill in betfair_over_odds, betfair_under_odds and checked_at "
          f"(e.g. 2026-10-31 14:05), save, then run:\n    python weekly_bets.py {out}")


if __name__ == "__main__":
    main()

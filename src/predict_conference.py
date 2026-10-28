"""
National League - live paper-tracking for model v3-conf (EXPERIMENT).

v3-conf = rd_conference3.py's best variant "C+SOT+xG", frozen 28 Oct 2026.
It did NOT pass the +0.002 bar (+0.0042 vs Betfair close), so it is
tracked as an experiment, like v3-eu-upper. Never change it - a new idea
gets a new version and a new record.

Recipe (fixed):
  - features: goals-based v3 features + shots-on-target form + rolling xG,
    built on all English divisions + National League together
  - Poisson regression (alpha 1.3335) trained on League Two + National
    League matches from 2017/18 to 2025/26 ONLY - identical every run
    (the fingerprint printed below must never change)
  - Dixon-Coles on League Two + NL linked, re-fitted on the last 3 years
    before the match week (as v3)
  - 50/50 blend
Data (all read-only):
  data/raw/conference/EC_*.csv          football-data National League (re-run
                                        check_conference_data.py to refresh)
  data/raw/conference/bzzoiro_stats_nl_l2.csv   Bzzoiro NL shots on target + xG
  data/clean/matches_clean.csv          English divisions
Football-data's own NL shots on target are used where Bzzoiro has none.

Every run:
  1. re-downloads the current season's National League file (EC_2627.csv)
     from football-data (skip with --no-download)
  2. predicts the fixtures and appends them to data/live/conference_log.csv
     (timestamped; the first prediction per match is kept, never overwritten)
  3. stores them in the database (live_fixtures + predictions, group
     'conference') and writes data/live/odds_entry_conference_<date>.csv
     in the same format as weekly_predict.py - so weekly_bets.py,
     weekly_settle.py and the spreadsheet handle National League bets like
     every other group.
Before running, refresh Bzzoiro stats: scratch/pull_bzzoiro_national_league.py
and rebuild data/raw/conference/bzzoiro_stats_nl_l2.csv (Claude does this).

Fixtures file: Div,Date,Time,HomeTeam,AwayTeam (football-data names, Div EC)
Run from inside src/:
    python predict_conference.py ..\\data\\live\\fixtures_conference_2026-10-28.csv
    then fill the odds sheet it writes and run weekly_bets.py on it, as usual.
"""

import hashlib
import os
import sys
import urllib.request
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import rd_conference as r1  # noqa: E402
from rd_conference3 import xg_features, BZ_CSV  # noqa: E402
from rd_experiments3 import rolling_dc, NON_FEATURE_COLS, POISSON_ALPHA  # noqa: E402
from build_features import build_feature_table  # noqa: E402
from import_helpers import build_match_id, season_code_to_display  # noqa: E402

MODEL_VERSION = "v3-conf"
TRAIN_SEASONS = r1.ALL_SEASONS                      # 2017/18 .. 2025/26 - frozen
CURRENT_SEASON = "2026/27"
LOG_CSV = os.path.join("..", "data", "live", "conference_log.csv")
COMMISSION, MIN_EDGE = 0.02, 0.03
NL, L2 = r1.NL, r1.L2


def min_back_odds(p):
    return 1 + (MIN_EDGE + 1 - p) / (p * (1 - COMMISSION))


def load_nl_all():
    """National League from every EC file, including the current season."""
    saved = r1.ALL_SEASONS
    r1.ALL_SEASONS = saved + [CURRENT_SEASON]
    try:
        nl = r1.load_national_league()
    finally:
        r1.ALL_SEASONS = saved
    return nl


def refresh_current_season():
    url = "https://www.football-data.co.uk/mmz4281/2627/EC.csv"
    path = os.path.join(r1.CONF_DIR, "EC_2627.csv")
    try:
        urllib.request.urlretrieve(url, path)
        n = len(pd.read_csv(path, encoding="latin-1", on_bad_lines="skip").dropna(subset=["FTHG"]))
        print(f"Downloaded National League 2026/27 from football-data: {n} results")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not refresh EC_2627.csv ({e}) - using the copy on disk")


def store(log, match_ids, today):
    """Store in the database (first prediction kept) and write the odds sheet."""
    import live_db
    conn = live_db.connect()
    rows = log[log["match_id"].isin(match_ids)].copy()
    # Excel re-saves the log with UK dates (28/10/2026) - always store ISO dates
    rows["match_date"] = pd.to_datetime(rows["match_date"].astype(str).str[:10], dayfirst=True).dt.strftime("%Y-%m-%d")
    new = 0
    for r in rows.itertuples(index=False):
        conn.execute("""INSERT OR IGNORE INTO live_fixtures
                        (match_id, season, league_name, match_date, kickoff_time, home_team, away_team)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                     (r.match_id, CURRENT_SEASON, NL, r.match_date,
                      None if pd.isna(r.kickoff) else str(r.kickoff), r.home_team, r.away_team))
        conn.execute("UPDATE live_fixtures SET match_date = ? WHERE match_id = ? AND match_date <> ?",
                     (r.match_date, r.match_id, r.match_date))      # repairs any non-ISO date stored earlier
        cur = conn.execute("""INSERT OR IGNORE INTO predictions
                              (match_id, model_version, predicted_prob_over_2_5, prediction_date, p_poisson, p_dc)
                              VALUES (?, ?, ?, ?, ?, ?)""",
                           (r.match_id, MODEL_VERSION, float(r.p_over), r.predicted_at,
                            float(r.p_poisson), None if pd.isna(r.dc_p_over) else float(r.dc_p_over)))
        new += cur.rowcount
    conn.commit()
    stored = pd.read_sql_query(
        "SELECT match_id, predicted_prob_over_2_5 AS p_over FROM predictions WHERE model_version = ?",
        conn, params=(MODEL_VERSION,))
    sh = rows.drop(columns=["p_over"]).merge(stored, on="match_id")
    sheet = pd.DataFrame({
        "match_id": sh["match_id"], "date": sh["match_date"].astype(str).str[:10], "time": sh["kickoff"],
        "league": NL, "home": sh["home_team"], "away": sh["away_team"],
        "model_p_over": sh["p_over"].round(4),
        "fair_odds_over": (1 / sh["p_over"]).round(2), "fair_odds_under": (1 / (1 - sh["p_over"])).round(2),
        "bet_over_if_odds_at_least": sh["p_over"].map(min_back_odds).round(2),
        "bet_under_if_odds_at_least": (1 - sh["p_over"]).map(min_back_odds).round(2),
        "betfair_over_odds": "", "betfair_under_odds": "", "checked_at": ""})
    out = os.path.join(live_db.LIVE_DIR, f"odds_entry_conference_{today}.csv")
    if os.path.exists(out):
        out = os.path.join(live_db.LIVE_DIR, f"odds_entry_conference_{today}_{datetime.now():%H%M%S}.csv")
    sheet.to_csv(out, index=False)
    return new, out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print("Usage: python predict_conference.py <fixtures csv> [--no-download]")
        sys.exit(1)
    if "--no-download" not in sys.argv:
        refresh_current_season()
    now = datetime.now()
    fx = pd.read_csv(args[0])
    fx = fx[fx["Div"] == "EC"]
    rows = []
    for r in fx.to_dict("records"):
        mid, iso = build_match_id(r["Date"], "EC", r["HomeTeam"], r["AwayTeam"])
        rows.append({"match_id": mid, "season": CURRENT_SEASON, "league_name": NL,
                     "match_date": pd.Timestamp(iso), "kickoff": r.get("Time"),
                     "home_team": r["HomeTeam"], "away_team": r["AwayTeam"]})
    fixtures = pd.DataFrame(rows)

    eng = pd.read_csv(r1.CLEAN_CSV, parse_dates=["match_date"])
    nl = load_nl_all()
    # football-data's own NL shots on target (2026/27 onwards), used where Bzzoiro has none
    fd_sot = []
    for code in ["2627"]:
        path = os.path.join(r1.CONF_DIR, f"EC_{code}.csv")
        if os.path.exists(path):
            raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip").dropna(subset=["FTHG"])
            for r in raw.to_dict("records"):
                mid, _ = build_match_id(r["Date"], "EC", r["HomeTeam"], r["AwayTeam"])
                fd_sot.append((mid, r.get("HST"), r.get("AST")))
    fd_sot = pd.DataFrame(fd_sot, columns=["match_id", "fd_hst", "fd_ast"])

    unknown = sorted((set(fixtures["home_team"]) | set(fixtures["away_team"]))
                     - (set(nl["home_team"]) | set(nl["away_team"]) | set(eng["home_team"]) | set(eng["away_team"])))
    if unknown:
        print("STOP: unknown team names (use football-data's spelling):", unknown)
        sys.exit(1)
    fixtures = fixtures[~fixtures["match_id"].isin(nl["match_id"])]     # already played
    if fixtures.empty:
        print("No unplayed fixtures in the file.")
        return

    base_cols = [c for c in nl.columns if c in eng.columns]
    played = pd.concat([eng[base_cols], nl], ignore_index=True)
    fx_rows = fixtures.drop(columns=["kickoff"]).assign(behind_closed_doors=0)
    combined = pd.concat([played, fx_rows], ignore_index=True)

    bz = pd.read_csv(BZ_CSV)
    combined = combined.merge(bz[["match_id", "home_xg", "away_xg", "home_sot", "away_sot"]], on="match_id", how="left")
    combined = combined.merge(fd_sot, on="match_id", how="left")
    is_nl = combined["league_name"] == NL
    combined.loc[is_nl, "home_shots_on_target"] = combined.loc[is_nl, "home_sot"].fillna(combined.loc[is_nl, "fd_hst"])
    combined.loc[is_nl, "away_shots_on_target"] = combined.loc[is_nl, "away_sot"].fillna(combined.loc[is_nl, "fd_ast"])

    feats = build_feature_table(combined.drop(columns=["home_xg", "away_xg", "home_sot", "away_sot",
                                                       "fd_hst", "fd_ast"]))
    feats = feats.merge(xg_features(combined), on="match_id", how="left")
    base = [c for c in feats.columns if c not in NON_FEATURE_COLS and "shots" not in c and "sot_" not in c
            and not c.startswith("league_") and "xg" not in c] + ["league_National League"]
    cols = base + [c for c in feats.columns if "sot_" in c] + \
        [c for c in xg_features(combined.head(0)).columns if c != "match_id"]

    train = feats[feats["season"].isin(TRAIN_SEASONS) & feats["league_name"].isin([L2, NL])]
    cols = [c for c in cols if train[c].notna().any()]
    med = train[cols].median(numeric_only=True)
    sc = StandardScaler().fit(train[cols].fillna(med))
    goals = train[["match_id"]].merge(combined[["match_id", "home_goals", "away_goals"]], on="match_id")
    Xtr = sc.transform(train[cols].fillna(med))
    m_h = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000).fit(Xtr, goals["home_goals"])
    m_a = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000).fit(Xtr, goals["away_goals"])
    fp = hashlib.sha256(np.round(np.concatenate([m_h.coef_, m_a.coef_, [m_h.intercept_, m_a.intercept_]]), 8)
                        .tobytes()).hexdigest()[:16]
    print(f"Model {MODEL_VERSION} (experiment) - fingerprint {fp} ({len(train)} training matches, {len(cols)} features)")

    te = feats[feats["match_id"].isin(fixtures["match_id"])].copy()
    Xte = sc.transform(te[cols].fillna(med))
    lam = m_h.predict(Xte) + m_a.predict(Xte)
    te["p_poisson"] = 1 - poisson_dist.cdf(2, lam)
    dc = rolling_dc(combined[combined["league_name"].isin([L2, NL])], 0.0)[["match_id", "dc_p_over"]]
    te = te.merge(dc, on="match_id", how="left")
    te["p_over"] = 0.5 * te["p_poisson"] + 0.5 * te["dc_p_over"].fillna(te["p_poisson"])

    out = te[["match_id", "match_date", "home_team", "away_team", "p_poisson", "dc_p_over", "p_over"]].merge(
        fixtures[["match_id", "kickoff"]], on="match_id")
    out["bet_over_if_odds_at_least"] = out["p_over"].map(min_back_odds).round(2)
    out["bet_under_if_odds_at_least"] = (1 - out["p_over"]).map(min_back_odds).round(2)
    out["model_version"] = MODEL_VERSION
    out["fingerprint"] = fp
    out["predicted_at"] = now.isoformat(timespec="seconds")
    for c in ["betfair_over_odds", "betfair_under_odds", "checked_at", "bet_side", "home_goals", "away_goals",
              "result", "profit_loss"]:
        out[c] = np.nan

    if os.path.exists(LOG_CSV):
        log = pd.read_csv(LOG_CSV)
        new = out[~out["match_id"].isin(log["match_id"])]
        kept = len(out) - len(new)
        log = pd.concat([log, new], ignore_index=True)
    else:
        new, kept, log = out, 0, out
    log.to_csv(LOG_CSV, index=False)

    show = out[["match_date", "kickoff", "home_team", "away_team", "p_over",
                "bet_over_if_odds_at_least", "bet_under_if_odds_at_least"]].copy()
    show["match_date"] = pd.to_datetime(show["match_date"]).dt.strftime("%Y-%m-%d")
    show["p_over"] = show["p_over"].round(4)
    print(f"\n{len(out)} fixtures: {len(new)} new predictions logged, {kept} already logged earlier (kept as-is).")
    print(show.to_string(index=False))
    n_db, sheet_path = store(log, out["match_id"], now.date().isoformat())
    print(f"\nLog: {LOG_CSV}  |  database: {n_db} new prediction(s) stored (group 'conference')")
    print(f"\nNext: open {sheet_path}, fill in betfair_over_odds, betfair_under_odds and checked_at, "
          f"save, then run:\n    python weekly_bets.py {sheet_path}")


if __name__ == "__main__":
    main()

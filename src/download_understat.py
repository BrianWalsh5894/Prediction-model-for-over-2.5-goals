"""
Download match xG from Understat and link it to our matches: the Premier
League (step 4) and La Liga, Serie A, Bundesliga, Ligue 1 (v3-eu-upper).

Understat (understat.com) publishes match-level expected goals (its own
xG model) for the Premier League from 2014/15. We take 2017/18 - 2026/27
to line up with the rest of the project.

What it does:
  1. Downloads each season's league data (the same JSON endpoint the
     Understat website uses) and saves the raw file untouched to
     data/raw/understat/EPL_<year>.json (gitignored, like all raw data).
  2. Works out the Understat-name -> our-name mapping for every team
     AUTOMATICALLY: Understat and football-data.co.uk list the same
     matches, so a match on the same date with the same score pins down
     which team is which. The mapping is printed so you can eyeball it.
  3. Writes table match_xg (match_id, understat_id, home_xg, away_xg) to
     the database and reports any of our PL matches it could not link.

Only completed matches are stored. Safe to re-run any time (e.g. weekly):
existing rows are replaced with the latest values.

Run from inside src/ (about a minute - one request per season, with a
pause between requests to be polite to their server):
    python download_understat.py                       (all five leagues)
    python download_understat.py --league "La Liga"    (just one)
    python download_understat.py --offline    (re-link from saved raw files, no download)
"""

import os
import sys
import json
import time
import argparse
import sqlite3
from collections import Counter, defaultdict
import pandas as pd
import requests

DB_PATH = os.path.join("..", "data", "over25.db")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
RAW_DIR = os.path.join("..", "data", "raw", "understat")
UNDERSTAT = "https://understat.com"
HEADERS = {"X-Requested-With": "XMLHttpRequest",
           "User-Agent": "Mozilla/5.0 (over25 student project)"}
SEASONS = {f"20{y:02d}/{y + 1:02d}": 2000 + y for y in range(17, 27)}   # '2017/18' -> 2017
# our league name -> (Understat slug, our clean file)
LEAGUES = {
    "Premier League": ("EPL", CLEAN_CSV),
    "La Liga": ("La_liga", os.path.join("..", "data", "clean", "matches_clean_eu_upper.csv")),
    "Serie A": ("Serie_A", os.path.join("..", "data", "clean", "matches_clean_eu_upper.csv")),
    "Bundesliga": ("Bundesliga", os.path.join("..", "data", "clean", "matches_clean_eu_upper.csv")),
    "Ligue 1": ("Ligue_1", os.path.join("..", "data", "clean", "matches_clean_eu_upper.csv")),
}


def download(seasons, slug):
    os.makedirs(RAW_DIR, exist_ok=True)
    s = requests.Session()
    s.get(UNDERSTAT, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=30)   # sets cookies
    for season, year in seasons.items():
        url = f"{UNDERSTAT}/getLeagueData/{slug}/{year}"
        r = s.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        with open(os.path.join(RAW_DIR, f"{slug}_{year}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f)
        n_done = sum(1 for m in data["dates"] if m.get("isResult"))
        print(f"  {season}: {len(data['dates'])} fixtures, {n_done} played")
        time.sleep(1.5)


def load_raw(seasons, slug):
    rows = []
    for season, year in seasons.items():
        path = os.path.join(RAW_DIR, f"{slug}_{year}.json")
        if not os.path.exists(path):
            print(f"  missing {path} - skipped")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for m in data["dates"]:
            if not m.get("isResult"):
                continue
            rows.append({
                "season": season, "understat_id": int(m["id"]),
                "date": pd.Timestamp(m["datetime"]).normalize(),
                "u_home": m["h"]["title"], "u_away": m["a"]["title"],
                "home_goals": int(m["goals"]["h"]), "away_goals": int(m["goals"]["a"]),
                "home_xg": float(m["xG"]["h"]), "away_xg": float(m["xG"]["a"]),
            })
    return pd.DataFrame(rows)


def build_team_map(u, ours):
    """Vote: an Understat match and one of ours on the same date (+/- 1 day)
    with the same season and identical score is very likely the same match.
    Only unambiguous pairings vote; each Understat name takes its majority."""
    by_key = defaultdict(list)
    for r in ours.itertuples(index=False):
        for shift in (-1, 0, 1):
            by_key[(r.season, r.match_date + pd.Timedelta(days=shift), r.home_goals, r.away_goals)].append(r)
    votes = defaultdict(Counter)
    for r in u.itertuples(index=False):
        cands = by_key.get((r.season, r.date, r.home_goals, r.away_goals), [])
        if len(cands) == 1:
            votes[r.u_home][cands[0].home_team] += 1
            votes[r.u_away][cands[0].away_team] += 1
    mapping, report = {}, []
    for name, c in sorted(votes.items()):
        best, n = c.most_common(1)[0]
        mapping[name] = best
        report.append((name, best, n, sum(c.values())))
    return mapping, report


def process(league, offline):
    slug, clean_csv = LEAGUES[league]
    print(f"\n==================== {league} ({slug}) ====================")
    if not offline:
        print(f"Downloading {league} xG from Understat...")
        download(SEASONS, slug)

    u = load_raw(SEASONS, slug)
    if u.empty:
        print("No Understat data found - skipped.")
        return
    if not os.path.exists(clean_csv):
        print(f"{clean_csv} not found - run clean_data.py first. Skipped.")
        return
    ours = pd.read_csv(clean_csv, parse_dates=["match_date"])
    ours = ours[ours["league_name"] == league]

    mapping, report = build_team_map(u, ours)
    print("\nTeam name mapping (Understat -> ours, votes agreeing / total):")
    shaky = []
    for name, ours_name, agree, total in report:
        flag = "" if agree / total >= 0.9 and agree >= 3 else "   <-- CHECK"
        if flag:
            shaky.append(name)
        print(f"  {name:28s} -> {ours_name:28s} {agree}/{total}{flag}")
    unmapped = sorted((set(u["u_home"]) | set(u["u_away"])) - set(mapping))
    if unmapped:
        print(f"  NOT MAPPED: {unmapped}")

    u["home_team"] = u["u_home"].map(mapping)
    u["away_team"] = u["u_away"].map(mapping)

    # Link by season + home + away (each pairing happens once per season),
    # then sanity-check the date and the score.
    linked = ours[["match_id", "season", "match_date", "home_team", "away_team", "home_goals", "away_goals"]].merge(
        u[["season", "home_team", "away_team", "understat_id", "date", "home_goals", "away_goals",
           "home_xg", "away_xg"]],
        on=["season", "home_team", "away_team"], how="left", suffixes=("", "_u"))
    has = linked["understat_id"].notna()
    score_bad = has & ((linked["home_goals"] != linked["home_goals_u"]) | (linked["away_goals"] != linked["away_goals_u"]))
    date_gap = (linked["match_date"] - linked["date"]).dt.days.abs()
    date_bad = has & (date_gap > 3)
    ok = has & ~score_bad & ~date_bad

    print(f"\nOur {league} matches: {len(linked)}   linked: {int(ok.sum())}   "
          f"no Understat match: {int((~has).sum())}   score mismatch: {int(score_bad.sum())}   "
          f"date >3 days apart: {int(date_bad.sum())}")
    print(linked.assign(linked=ok).groupby("season")["linked"].agg(["sum", "size"])
          .rename(columns={"sum": "linked", "size": "our_matches"}).to_string())
    if (~ok).any():
        print("\nUnlinked examples:")
        print(linked.loc[~ok, ["match_id", "match_date", "date", "home_goals", "home_goals_u"]].head(10).to_string(index=False))

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS match_xg (
            match_id TEXT PRIMARY KEY,
            understat_id INTEGER,
            home_xg REAL,
            away_xg REAL,
            source TEXT NOT NULL DEFAULT 'understat',
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        )
    """)
    out = linked.loc[ok, ["match_id", "understat_id", "home_xg", "away_xg"]]
    conn.executemany("INSERT OR REPLACE INTO match_xg (match_id, understat_id, home_xg, away_xg) VALUES (?, ?, ?, ?)",
                     [(r.match_id, int(r.understat_id), float(r.home_xg), float(r.away_xg))
                      for r in out.itertuples(index=False)])
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM match_xg").fetchone()[0]
    conn.close()
    print(f"\nWrote {len(out)} rows to match_xg ({total} in table).")
    if shaky or unmapped or (~ok).sum() > 5:
        print("Some names/matches need a look - paste this output to check before running R&D.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="don't download; use saved raw files")
    ap.add_argument("--league", default="all", choices=["all"] + list(LEAGUES),
                    help="one league, or all (default)")
    args = ap.parse_args()
    for league in (LEAGUES if args.league == "all" else [args.league]):
        process(league, args.offline)


if __name__ == "__main__":
    main()

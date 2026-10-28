"""
National League (Conference) - STEP 0 data check. Read-only R&D:
does NOT touch the database, the live files or any model.

Downloads football-data.co.uk's National League file (code EC) for every
season 2017/18 - 2026/27 into data/raw/conference/ and reports, per season:
  - matches, and whether the scores are complete
  - STATS: shots, shots on target, corners, fouls, cards - % of matches filled
  - ODDS: bookmaker average / max / Pinnacle / Bet365 Over/Under 2.5,
          Betfair early and closing Over/Under 2.5 - % of matches filled
  - Over 2.5 rate
  - teams promoted in (not in the previous season's file) - their history is
    the weak spot to plan for

Run from inside src/ (needs internet, about a minute):
    python check_conference_data.py
"""

import os
import time
import urllib.request

import pandas as pd

BASE = "https://www.football-data.co.uk/mmz4281/{code}/EC.csv"
OUT_DIR = os.path.join("..", "data", "raw", "conference")
SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]

STATS = {"shots": ["HS", "AS"], "shots_on_target": ["HST", "AST"], "corners": ["HC", "AC"],
         "fouls": ["HF", "AF"], "yellows": ["HY", "AY"], "reds": ["HR", "AR"]}
ODDS = {"avg_OU": ["Avg>2.5", "Avg<2.5"], "max_OU": ["Max>2.5", "Max<2.5"],
        "pinnacle_OU": ["P>2.5", "P<2.5"], "bet365_OU": ["B365>2.5", "B365<2.5"],
        "betfair_OU_early": ["BFE>2.5", "BFE<2.5"],
        "avg_OU_close": ["AvgC>2.5", "AvgC<2.5"], "pinnacle_OU_close": ["PC>2.5", "PC<2.5"],
        "betfair_OU_close": ["BFEC>2.5", "BFEC<2.5"]}


def pct_filled(df, cols):
    if not all(c in df.columns for c in cols):
        return "missing"
    return f"{df[cols].notna().all(axis=1).mean():.0%}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, teams_by_season = [], {}
    for code in SEASONS:
        path = os.path.join(OUT_DIR, f"EC_{code}.csv")
        label = f"20{code[:2]}/{code[2:]}"
        try:
            urllib.request.urlretrieve(BASE.format(code=code), path)
            time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            print(f"{label}: download failed ({e})")
            rows.append({"season": label, "matches": 0})
            continue
        try:
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        except Exception as e:  # noqa: BLE001
            print(f"{label}: could not read file ({e})")
            continue
        df = df.dropna(subset=["HomeTeam", "AwayTeam"])
        played = df.dropna(subset=["FTHG", "FTAG"])
        teams = set(df["HomeTeam"]) | set(df["AwayTeam"])
        teams_by_season[label] = teams
        row = {"season": label, "matches": len(df), "with_score": len(played),
               "over25_rate": f"{((played['FTHG'] + played['FTAG']) > 2).mean():.0%}" if len(played) else ""}
        for k, cols in STATS.items():
            row[k] = pct_filled(df, cols)
        for k, cols in ODDS.items():
            row[k] = pct_filled(df, cols)
        rows.append(row)
        print(f"{label}: {len(df)} matches downloaded")

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== STATS (% of matches with the column filled) ===")
    print(res[["season", "matches", "with_score", "over25_rate"] + list(STATS)].to_string(index=False))
    print("\n=== ODDS (% of matches with Over AND Under filled) ===")
    print(res[["season"] + list(ODDS)].to_string(index=False))

    print("\n=== Teams new to the National League each season (promoted from North/South or relegated from League Two) ===")
    labels = list(teams_by_season)
    for prev, cur in zip(labels, labels[1:]):
        new = sorted(teams_by_season[cur] - teams_by_season[prev])
        print(f"{cur}: {len(new)} new - {', '.join(new)}")

    out = os.path.join("..", "reports", "conference_data_check.csv")
    res.to_csv(out, index=False)
    print(f"\nFiles saved in {OUT_DIR}. Summary: {out}")
    print("Tell Claude it's done - it will read the files and work out which teams came down from League Two.")


if __name__ == "__main__":
    main()

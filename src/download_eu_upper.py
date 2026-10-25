"""
v3-eu-upper - step 1: download and import La Liga, Serie A, Bundesliga and
Ligue 1, plus their second divisions (history for promoted teams), from
football-data.co.uk (2017/18 - 2025/26, same CSV format as the
English files). The current season (2026/27) comes in through
update_current_season.py, which now covers these leagues too.

Raw files are saved untouched to data/raw/eu_upper/ (gitignored) with a
download log, then imported with the same code as the English data
(import_helpers.import_dataframe), so match_id, NULL handling and the
COVID flag (2020/21) all work exactly the same way.

Model v3 is not affected: clean_data.py keeps exporting the English leagues
to matches_clean.csv unchanged, and these leagues to
matches_clean_eu_upper.csv.

Run from inside src/ (about a minute):
    python download_eu_upper.py
then:
    python clean_data.py
    python build_features_eu_upper.py
"""

import os
import io
import time
import sqlite3
import requests
import pandas as pd
from import_helpers import import_dataframe, season_code_to_display

DB_PATH = os.path.join("..", "data", "over25.db")
RAW_DIR = os.path.join("..", "data", "raw", "eu_upper")
BASE_URL = "https://www.football-data.co.uk/mmz4281"
EU_UPPER = {"SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1",
            # second divisions: history for promoted teams
            "SP2": "La Liga 2", "I2": "Serie B", "D2": "2. Bundesliga", "F2": "Ligue 2"}
SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    log = []
    for season in SEASONS:
        for div, name in EU_UPPER.items():
            url = f"{BASE_URL}/{season}/{div}.csv"
            path = os.path.join(RAW_DIR, f"{div}_{season}.csv")
            r = requests.get(url, timeout=30)
            if r.status_code != 200 or not r.content:
                print(f"MISS {div}_{season} (status {r.status_code})")
                log.append(f"MISS,{div}_{season}.csv,{url}")
                continue
            with open(path, "wb") as f:
                f.write(r.content)
            try:
                df = pd.read_csv(io.BytesIO(r.content), encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
            covid = 1 if season == "2021" else 0
            n = import_dataframe(conn, cursor, div, season_code_to_display(season), df, covid_flag=covid)
            conn.commit()
            print(f"OK   {name:13s} {season_code_to_display(season)}: {n} matches")
            log.append(f"OK,{div}_{season}.csv,{url}")
            time.sleep(0.5)
    with open(os.path.join(RAW_DIR, "_download_log.csv"), "w") as f:
        f.write("status,filename,url\n" + "\n".join(log))
    conn.close()
    print("\nDone. Next: python clean_data.py   then   python build_features_eu_upper.py")


if __name__ == "__main__":
    main()

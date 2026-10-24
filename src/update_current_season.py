"""
Weekly refresh: re-downloads the current season (2026/27) CSVs for
all four divisions and upserts any new completed matches into the
database. Safe to run repeatedly - existing matches get silently
replaced with the same data, new ones get added.
"""

import sqlite3
import pandas as pd
import os
from import_helpers import DIVISIONS, season_code_to_display, import_dataframe

DB_PATH = os.path.join("..", "data", "over25.db")
CURRENT_SEASON_CODE = "2627"
BASE_URL = "https://www.football-data.co.uk/mmz4281"


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    season_display = season_code_to_display(CURRENT_SEASON_CODE)

    for div_code in DIVISIONS:
        url = f"{BASE_URL}/{CURRENT_SEASON_CODE}/{div_code}.csv"
        df = pd.read_csv(url)
        n = import_dataframe(conn, cursor, div_code, season_display, df)
        conn.commit()
        print(f"{div_code} {season_display}: {n} matches up to date")

    conn.close()


if __name__ == "__main__":
    main()
"""
Downloads historical match data from Football-Data.co.uk for
Premier League, Championship, League One and League Two,
seasons 2017/18 through 2025/26.
"""

import requests
import os
import time

# Division codes used by football-data.co.uk
DIVISIONS = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
}

# Seasons as football-data.co.uk encodes them: 2017/18 -> "1718"
SEASONS = [
    "1718", "1819", "1920", "2021", "2122",
    "2223", "2324", "2425", "2526",
]

BASE_URL = "https://www.football-data.co.uk/mmz4281"
OUTPUT_DIR = os.path.join("..", "data", "raw")

os.makedirs(OUTPUT_DIR, exist_ok=True)

log_lines = []

for season in SEASONS:
    for div_code, div_name in DIVISIONS.items():
        url = f"{BASE_URL}/{season}/{div_code}.csv"
        filename = f"{div_code}_{season}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)

        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200 and len(response.content) > 0:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"OK   {filename}  ({len(response.content)} bytes)")
                log_lines.append(f"OK,{filename},{url}")
            else:
                print(f"MISS {filename}  (status {response.status_code})")
                log_lines.append(f"MISS,{filename},{url}")
        except requests.RequestException as e:
            print(f"FAIL {filename}  ({e})")
            log_lines.append(f"FAIL,{filename},{url}")

        time.sleep(0.5)  # be polite to their server

# Write a simple download log — source, URL, date for every file
with open(os.path.join(OUTPUT_DIR, "_download_log.csv"), "w") as f:
    f.write("status,filename,url\n")
    f.write("\n".join(log_lines))

print("\nDone.")
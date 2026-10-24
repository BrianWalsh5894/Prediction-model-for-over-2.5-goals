"""
Builds a raw-name -> standard-name mapping for every team seen in
the source data. Source names are already internally consistent
(football-data.co.uk), so this mostly expands shorthand to full
club names and gives a stable place to add aliases later (e.g.
when Betfair API names differ).
"""

import sqlite3
import os

DB_PATH = os.path.join("..", "data", "over25.db")

# raw_name (as it appears in the CSVs) -> standard_name
# Only entries that need expanding/correcting; anything not listed
# here is assumed to already be fine as-is.
NAME_MAP = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Weds": "Sheffield Wednesday",
    "Bristol Rvs": "Bristol Rovers",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    "QPR": "Queens Park Rangers",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Newport County": "Newport County",
    "Milton Keynes Dons": "MK Dons",
}

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS team_name_map (
    raw_name TEXT PRIMARY KEY,
    standard_name TEXT NOT NULL
)
""")

# All 103 raw names collected from the source files
raw_names = ['AFC Wimbledon', 'Accrington', 'Arsenal', 'Aston Villa', 'Barnet', 'Barnsley', 'Barrow', 'Birmingham', 'Blackburn', 'Blackpool', 'Bolton', 'Bournemouth', 'Bradford', 'Brentford', 'Brighton', 'Bristol City', 'Bristol Rvs', 'Bromley', 'Burnley', 'Burton', 'Bury', 'Cambridge', 'Cardiff', 'Carlisle', 'Charlton', 'Chelsea', 'Cheltenham', 'Chesterfield', 'Colchester', 'Coventry', 'Crawley Town', 'Crewe', 'Crystal Palace', 'Derby', 'Doncaster', 'Everton', 'Exeter', 'Fleetwood Town', 'Forest Green', 'Fulham', 'Gillingham', 'Grimsby', 'Harrogate', 'Hartlepool', 'Huddersfield', 'Hull', 'Ipswich', 'Leeds', 'Leicester', 'Leyton Orient', 'Lincoln', 'Liverpool', 'Luton', 'Macclesfield', 'Man City', 'Man United', 'Mansfield', 'Middlesbrough', 'Millwall', 'Milton Keynes Dons', 'Morecambe', 'Newcastle', 'Newport County', 'Northampton', 'Norwich', "Nott'm Forest", 'Notts County', 'Oldham', 'Oxford', 'Peterboro', 'Plymouth', 'Port Vale', 'Portsmouth', 'Preston', 'QPR', 'Reading', 'Rochdale', 'Rotherham', 'Salford', 'Scunthorpe', 'Sheffield United', 'Sheffield Weds', 'Shrewsbury', 'Southampton', 'Southend', 'Stevenage', 'Stockport', 'Stoke', 'Sunderland', 'Sutton', 'Swansea', 'Swindon', 'Tottenham', 'Tranmere', 'Walsall', 'Watford', 'West Brom', 'West Ham', 'Wigan', 'Wolves', 'Wrexham', 'Wycombe', 'Yeovil']

rows = [(name, NAME_MAP.get(name, name)) for name in raw_names]
cursor.executemany(
    "INSERT OR IGNORE INTO team_name_map (raw_name, standard_name) VALUES (?, ?)",
    rows
)

conn.commit()
cursor.execute("SELECT COUNT(*) FROM team_name_map")
print("Rows in team_name_map:", cursor.fetchone()[0])
conn.close()
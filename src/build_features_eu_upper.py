"""
v3-eu-upper - step 2: the same 103 Phase 5 features for La Liga, Serie A,
Bundesliga and Ligue 1, built with exactly the same code as England
(build_features.build_feature_table - leak-free, every feature shifted).

Reads data/clean/matches_clean_eu_upper.csv (from clean_data.py) and writes
data/clean/features_eu_upper.csv. The English features.csv is untouched.

Run from inside src/:
    python build_features_eu_upper.py
"""

import os
import pandas as pd
from build_features import build_feature_table

CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean_eu_upper.csv")
FEATURES_CSV = os.path.join("..", "data", "clean", "features_eu_upper.csv")


def main():
    df = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    final = build_feature_table(df)
    final.to_csv(FEATURES_CSV, index=False)
    print(f"Wrote {len(final)} rows, {len(final.columns)} columns to {FEATURES_CSV}")
    print(final.groupby(["league_name", "season"]).size().unstack(0).to_string())


if __name__ == "__main__":
    main()

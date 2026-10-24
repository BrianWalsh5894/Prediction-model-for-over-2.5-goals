"""
Phase 5 - Leakage audit.

Independently re-derives a sample of feature values straight from
data/clean/matches_clean.csv (NOT by re-running build_features.py) and
checks they match what's in data/clean/features.csv. This is deliberately
a separate, independent recomputation, not a second call into the same
code - the point is to catch a bug that build_features.py itself can't
see because it would make the same mistake twice.

Also runs the "too good to be true" check from Phase 0 / Phase 5: any
single feature with a suspiciously strong correlation to the Over 2.5
target gets flagged for manual review rather than trusted.

Run from inside src/, after build_features.py:
    python audit_features.py
"""

import os
import sys
import pandas as pd
import numpy as np

CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")

N_SAMPLE = 300
RANDOM_SEED = 424410  # last 6 digits of student ID G00424410, per the project's convention
TOLERANCE = 1e-6

CORRELATION_FLAG_THRESHOLD = 0.4  # anything stronger than this with the target
                                    # gets printed for manual review, not auto-failed -
                                    # see Phase 0 principle: assume a suspiciously
                                    # strong result is a bug until proven otherwise


def fail(msg):
    print(f"FAIL: {msg}")
    return False


def team_history_before(raw, team, match_date, match_id):
    """All of `team`'s matches (home or away) strictly before (match_date, match_id),
    in chronological order, as a DataFrame with columns: match_date, match_id,
    goals_for, goals_against, is_home."""
    as_home = raw[raw["home_team"] == team][["match_date", "match_id", "home_goals", "away_goals"]]
    as_home = as_home.rename(columns={"home_goals": "goals_for", "away_goals": "goals_against"})
    as_home["is_home"] = 1

    as_away = raw[raw["away_team"] == team][["match_date", "match_id", "away_goals", "home_goals"]]
    as_away = as_away.rename(columns={"away_goals": "goals_for", "home_goals": "goals_against"})
    as_away["is_home"] = 0

    hist = pd.concat([as_home, as_away], ignore_index=True)
    hist = hist.sort_values(["match_date", "match_id"])

    before = hist[(hist["match_date"] < match_date) | ((hist["match_date"] == match_date) & (hist["match_id"] < match_id))]
    return before


def audit_rolling_and_rest(raw, features, rng):
    ok = True
    sample_idx = rng.choice(len(features), size=min(N_SAMPLE, len(features)), replace=False)

    checked = {"roll5": 0, "roll10": 0, "games_played": 0, "rest_days": 0}

    for idx in sample_idx:
        row = features.iloc[idx]
        match_date, match_id = row["match_date"], row["match_id"]

        for side, team_col, prefix in (("home", "home_team", "home"), ("away", "away_team", "away")):
            team = row[team_col]
            hist = team_history_before(raw, team, match_date, match_id)

            # games played (career) before this match
            expected_games = len(hist)
            actual_games = row[f"{prefix}_games_played_career"]
            if expected_games != actual_games:
                ok = fail(
                    f"games_played_career mismatch for {team} in match {match_id}: "
                    f"recomputed {expected_games}, feature says {actual_games}"
                )
            checked["games_played"] += 1

            if len(hist) == 0:
                continue  # nothing else to check for a team's very first match

            # last 5 / last 10 goals_for rolling mean, overall (any venue)
            for w in (5, 10):
                last_n = hist.tail(w)
                expected = last_n["goals_for"].mean()
                actual = row[f"{prefix}_overall_goals_for_roll{w}"]
                if pd.isna(expected) or pd.isna(actual):
                    if not (pd.isna(expected) and pd.isna(actual)):
                        ok = fail(f"roll{w} NaN mismatch for {team} in match {match_id}")
                elif abs(expected - actual) > TOLERANCE:
                    ok = fail(
                        f"overall_goals_for_roll{w} mismatch for {team} in match {match_id}: "
                        f"recomputed {expected}, feature says {actual}"
                    )
                checked[f"roll{w}"] += 1

            # rest days since this team's previous match, any venue
            prev_date = hist["match_date"].max()
            expected_rest = (match_date - prev_date).days
            actual_rest = row[f"{prefix}_rest_days"]
            if abs(expected_rest - actual_rest) > TOLERANCE:
                ok = fail(
                    f"rest_days mismatch for {team} in match {match_id}: "
                    f"recomputed {expected_rest}, feature says {actual_rest}"
                )
            checked["rest_days"] += 1

    print(f"Rechecked (team, match) pairs: {checked}")
    return ok


def audit_elo_independently(raw):
    """Recompute Elo from scratch with an independently-written loop
    (not imported from build_features.py) and compare to features.csv."""
    from math import pow

    ELO_INITIAL, ELO_K, ELO_HOME_ADVANTAGE = 1500, 20, 65

    raw_sorted = raw.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    ratings = {}
    home_pre, away_pre = [], []
    for row in raw_sorted.itertuples():
        r_home = ratings.get(row.home_team, ELO_INITIAL)
        r_away = ratings.get(row.away_team, ELO_INITIAL)
        home_pre.append(r_home)
        away_pre.append(r_away)
        exp_home = 1.0 / (1.0 + pow(10, -((r_home + ELO_HOME_ADVANTAGE) - r_away) / 400.0))
        score_home = 1.0 if row.result == "H" else (0.5 if row.result == "D" else 0.0)
        ratings[row.home_team] = r_home + ELO_K * (score_home - exp_home)
        ratings[row.away_team] = r_away + ELO_K * ((1 - score_home) - (1 - exp_home))

    recomputed = raw_sorted[["match_id"]].copy()
    recomputed["home_elo_pre_check"] = home_pre
    recomputed["away_elo_pre_check"] = away_pre
    return recomputed


def audit_correlations(features):
    numeric = features.select_dtypes(include=[np.number]).copy()
    if "over_2_5" not in numeric.columns:
        print("FAIL: over_2_5 not found as a numeric column, can't run correlation audit")
        return False

    target = numeric["over_2_5"]
    corrs = numeric.drop(columns=["over_2_5"]).corrwith(target).dropna()
    corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)

    print("\nTop 15 features by |correlation| with over_2_5:")
    for name, val in corrs.head(15).items():
        print(f"  {name:45s} {val:+.3f}")

    flagged = corrs[corrs.abs() > CORRELATION_FLAG_THRESHOLD]
    if len(flagged):
        print(f"\nFLAGGED for manual review (|corr| > {CORRELATION_FLAG_THRESHOLD}):")
        for name, val in flagged.items():
            print(f"  {name}: {val:+.3f}")
    else:
        print(f"\nNo feature exceeds |corr| > {CORRELATION_FLAG_THRESHOLD} with the target - "
              f"consistent with there being no leaked or trivially-predictive feature.")
    return True  # informational, not a hard pass/fail - see Phase 0 principle


def main():
    raw = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    features = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])

    rng = np.random.default_rng(RANDOM_SEED)

    ok = True
    print(f"Auditing a random sample of {N_SAMPLE} (team, match) checks, seed={RANDOM_SEED}...\n")
    ok = audit_rolling_and_rest(raw, features, rng) and ok

    elo_check = audit_elo_independently(raw)
    merged = features[["match_id", "home_elo_pre", "away_elo_pre"]].merge(elo_check, on="match_id")
    elo_mismatches = merged[
        ((merged["home_elo_pre"] - merged["home_elo_pre_check"]).abs() > TOLERANCE)
        | ((merged["away_elo_pre"] - merged["away_elo_pre_check"]).abs() > TOLERANCE)
    ]
    if len(elo_mismatches):
        ok = fail(f"{len(elo_mismatches)} matches where independently-recomputed Elo doesn't match features.csv")
    else:
        print(f"OK: independently-recomputed Elo matches features.csv for all {len(merged)} matches")

    # Promoted/relegated teams: sanity-print clubs that appear in more
    # than one league across the dataset, confirming their history
    # carries across divisions under a single team identity (the schema
    # already guarantees this - team_id has no league attached - this
    # just makes it visible rather than assumed).
    multi_league = (
        raw.melt(id_vars=["league_name"], value_vars=["home_team", "away_team"], value_name="team")
        .groupby("team")["league_name"].nunique()
    )
    moved_teams = multi_league[multi_league > 1]
    print(f"\n{len(moved_teams)} teams have played in more than one of the four divisions "
          f"(promoted/relegated at some point) - their rolling form carries across divisions "
          f"under one team_id, by design.")

    audit_correlations(features)

    eligible = features["eligible_for_training"].sum()
    total = len(features)
    print(f"\n{eligible}/{total} matches flagged eligible_for_training "
          f"(both teams have played their minimum career games)")

    if not ok:
        print("\nLeakage audit FAILED one or more checks - see FAIL lines above. Do not train on this feature set yet.")
        sys.exit(1)
    print("\nLeakage audit: all checks passed.")


if __name__ == "__main__":
    main()

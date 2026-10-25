"""
Phase 5 - Feature Engineering.

Reads the Phase 4 cleaned dataset (data/clean/matches_clean.csv) and
builds one row of pre-kickoff features per match, saved to
data/clean/features.csv.

HARD RULE (checked by audit_features.py, not just assumed here):
every feature for a match uses ONLY information available before that
match's kickoff. All rolling/expanding stats are computed with
.shift(1) before the window, so a match never sees its own result -
the shift happens BEFORE the rolling window, not after, which is the
part that's easy to get backwards and would leak the current match's
result into its own features.

Run from inside src/, after clean_data.py has produced matches_clean.csv:
    python build_features.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402

CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")

ROLLING_WINDOWS = config.ROLLING_WINDOWS                      # [5, 10]
MIN_GAMES_SEASON_FOR_FULL_FORM = config.MIN_GAMES_SEASON_FOR_FULL_FORM
MIN_GAMES_CAREER_FOR_TRAINING = config.MIN_GAMES_CAREER_FOR_TRAINING
ELO_INITIAL = config.ELO_INITIAL
ELO_K = config.ELO_K
ELO_HOME_ADVANTAGE = config.ELO_HOME_ADVANTAGE

STAT_COLS = [
    "goals_for", "goals_against",
    "shots_for", "shots_against",
    "sot_for", "sot_against",
]


def load_clean():
    df = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return df


def compute_elo(df):
    """
    Sequential pre-match Elo rating per team. home_elo_pre/away_elo_pre
    are each team's rating BEFORE this match - the only point in time
    that's legitimate to use as a feature. Ratings are then updated
    using this match's actual result, for use in the NEXT match.
    """
    ratings = {}
    home_pre, away_pre = [], []
    for row in df.itertuples():
        r_home = ratings.get(row.home_team, ELO_INITIAL)
        r_away = ratings.get(row.away_team, ELO_INITIAL)
        home_pre.append(r_home)
        away_pre.append(r_away)

        # Upcoming fixture (no result yet, Phase 11): it gets a pre-match
        # rating like any match, but must not update anyone's rating.
        if pd.isna(row.result):
            continue

        exp_home = 1.0 / (1.0 + 10 ** (-((r_home + ELO_HOME_ADVANTAGE) - r_away) / 400.0))
        if row.result == "H":
            score_home = 1.0
        elif row.result == "D":
            score_home = 0.5
        else:
            score_home = 0.0

        ratings[row.home_team] = r_home + ELO_K * (score_home - exp_home)
        ratings[row.away_team] = r_away + ELO_K * ((1 - score_home) - (1 - exp_home))

    df["home_elo_pre"] = home_pre
    df["away_elo_pre"] = away_pre
    return df


def build_long_format(df):
    """One row per team per match (so every match contributes two rows:
    the home team's perspective and the away team's perspective)."""
    home = pd.DataFrame({
        "match_id": df["match_id"],
        "match_date": df["match_date"],
        "season": df["season"],
        "league_name": df["league_name"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "is_home": 1,
        "goals_for": df["home_goals"],
        "goals_against": df["away_goals"],
        "shots_for": df["home_shots"],
        "shots_against": df["away_shots"],
        "sot_for": df["home_shots_on_target"],
        "sot_against": df["away_shots_on_target"],
        "over_2_5": df["over_2_5"],
        "points": np.where(df["result"] == "H", 3, np.where(df["result"] == "D", 1, 0)),
    })
    away = pd.DataFrame({
        "match_id": df["match_id"],
        "match_date": df["match_date"],
        "season": df["season"],
        "league_name": df["league_name"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "is_home": 0,
        "goals_for": df["away_goals"],
        "goals_against": df["home_goals"],
        "shots_for": df["away_shots"],
        "shots_against": df["home_shots"],
        "sot_for": df["away_shots_on_target"],
        "sot_against": df["home_shots_on_target"],
        "over_2_5": df["over_2_5"],
        "points": np.where(df["result"] == "A", 3, np.where(df["result"] == "D", 1, 0)),
    })
    long = pd.concat([home, away], ignore_index=True)
    # Upcoming fixtures (Phase 11) have no result: points must be missing,
    # not 0, so a team's NEXT fixture doesn't count this one as a defeat.
    long.loc[long["goals_for"].isna(), "points"] = np.nan
    long = long.sort_values(["team", "match_date", "match_id"]).reset_index(drop=True)
    return long


def main():
    final = build_feature_table(load_clean())
    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
    final.to_csv(FEATURES_CSV, index=False)
    print(f"Wrote {len(final)} rows, {len(final.columns)} columns to {FEATURES_CSV}")
    print(f"Matches eligible for training (both teams >= {MIN_GAMES_CAREER_FOR_TRAINING} career games): "
          f"{final['eligible_for_training'].sum()} / {len(final)}")


def build_feature_table(df):
    """Builds the feature table from a matches_clean-style DataFrame.
    Phase 11 also calls this with upcoming fixtures appended (goals,
    result and stats all missing): every feature is shifted, so a
    fixture's row only ever uses matches played before it."""
    df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    df = compute_elo(df)

    long = build_long_format(df)

    # --- ATTACK / DEFENCE: overall rolling (every match, any venue) -----
    g = long.groupby("team")
    for col in STAT_COLS:
        shifted = g[col].apply(lambda s: s.shift(1)).reset_index(level=0, drop=True)
        long[f"overall_{col}_shift"] = shifted  # temp, dropped later
    for col in STAT_COLS:
        shift_col = f"overall_{col}_shift"
        for w in ROLLING_WINDOWS:
            long[f"overall_{col}_roll{w}"] = (
                long.groupby("team")[shift_col]
                .transform(lambda s: s.rolling(w, min_periods=1).mean())
            )

    # games played before this match (career) - also leak-free: cumcount()
    # counts PRIOR rows in the group, not including the current one.
    long["games_played_career"] = long.groupby("team").cumcount()

    # season-to-date expanding average ("longer-run average"), also shifted
    long["season_match_num"] = long.groupby(["team", "season"]).cumcount()
    for col in STAT_COLS:
        shift_col = f"overall_{col}_shift"
        long[f"overall_{col}_season_avg"] = (
            long.groupby(["team", "season"])[shift_col]
            .transform(lambda s: s.expanding().mean())
        )

    # FORM: Over 2.5 % and points, last 5 / 10
    over_shift = long.groupby("team")["over_2_5"].apply(lambda s: s.shift(1)).reset_index(level=0, drop=True)
    long["over25_shift"] = over_shift
    points_shift = long.groupby("team")["points"].apply(lambda s: s.shift(1)).reset_index(level=0, drop=True)
    long["points_shift"] = points_shift
    for w in ROLLING_WINDOWS:
        long[f"over25_roll{w}"] = long.groupby("team")["over25_shift"].transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        long[f"points_roll{w}"] = long.groupby("team")["points_shift"].transform(
            lambda s: s.rolling(w, min_periods=1).sum()
        )

    # Rest days since this team's previous match (any venue, any season -
    # a large value at the start of a season is real signal, not an error)
    long["prev_match_date"] = long.groupby("team")["match_date"].shift(1)
    long["rest_days"] = (long["match_date"] - long["prev_match_date"]).dt.days

    # --- ATTACK / DEFENCE: home-only and away-only rolling ---------------
    # Computed on each venue subset separately so "home-only" never mixes
    # in the team's away matches and vice versa.
    long["venue_label"] = np.where(long["is_home"] == 1, "home", "away")
    for label in ("home", "away"):
        mask = long["venue_label"] == label
        sub = long.loc[mask].sort_values(["team", "match_date", "match_id"])
        gv = sub.groupby("team")
        for col in STAT_COLS:
            shifted = gv[col].apply(lambda s: s.shift(1)).reset_index(level=0, drop=True)
            for w in ROLLING_WINDOWS:
                colname = f"venue_{col}_roll{w}"
                if colname not in long.columns:
                    long[colname] = np.nan
                rolled = shifted.groupby(sub["team"]).transform(
                    lambda s: s.rolling(w, min_periods=1).mean()
                )
                long.loc[mask, colname] = rolled.values

    # --- Prior-season average per team (whole previous season, no leakage:
    # the season has already finished) - used to blend early-season matches
    season_agg = (
        long.groupby(["team", "season"])[STAT_COLS + ["over_2_5"]]
        .mean()
        .reset_index()
        .sort_values(["team", "season"])
    )
    for col in STAT_COLS + ["over_2_5"]:
        season_agg[f"prior_season_{col}"] = season_agg.groupby("team")[col].shift(1)
    prior_cols = ["team", "season"] + [f"prior_season_{c}" for c in STAT_COLS + ["over_2_5"]]
    long = long.merge(season_agg[prior_cols], on=["team", "season"], how="left")

    # --- League-wide running average, leak-free (only matches strictly
    # before this one in the same league) - fallback when a team has no
    # history of its own at all (its very first match in the dataset).
    df_league = df.sort_values(["league_name", "match_date", "match_id"]).copy()
    df_league["league_over25_avg_to_date"] = (
        df_league.groupby("league_name")["over_2_5"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    df_league["league_goals_avg_to_date"] = (
        df_league.groupby("league_name")["total_goals"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    df = df.merge(
        df_league[["match_id", "league_over25_avg_to_date", "league_goals_avg_to_date"]],
        on="match_id", how="left",
    )

    # --- Blend early-season form with prior-season / league average -----
    # Only a small set of core stats are blended explicitly (goals for/against,
    # Over 2.5 rate) - the same pattern extends to the others if needed later.
    long = long.merge(
        df[["match_id", "league_over25_avg_to_date", "league_goals_avg_to_date"]],
        on="match_id", how="left",
    )
    enough_season_games = long["season_match_num"] >= MIN_GAMES_SEASON_FOR_FULL_FORM

    def blend(rolling_col, prior_col, league_col):
        return np.select(
            [enough_season_games, long[prior_col].notna()],
            [long[rolling_col], long[prior_col]],
            default=long[league_col],
        )

    long["blended_goals_for"] = blend("overall_goals_for_roll10", "prior_season_goals_for", "league_goals_avg_to_date")
    long["blended_goals_against"] = blend("overall_goals_against_roll10", "prior_season_goals_against", "league_goals_avg_to_date")
    long["blended_over25_rate"] = blend("over25_roll10", "prior_season_over_2_5", "league_over25_avg_to_date")

    # --- Drop temp/shift helper columns before splitting to match-level -
    # league_over25_avg_to_date / league_goals_avg_to_date are dropped here
    # too: they're already carried as match-level columns (same value for
    # both sides, since it's a property of the match's league, not of
    # either team) and were only needed per-row above to compute the
    # blended_* columns - keeping them again under home_/away_ prefixes
    # would just be a duplicate of the top-level column.
    drop_cols = [
        c for c in long.columns
        if c.endswith("_shift") or c in (
            "venue_label", "prev_match_date",
            "league_over25_avg_to_date", "league_goals_avg_to_date",
        )
    ]
    long = long.drop(columns=drop_cols)

    # --- Split back to match-level: one block of columns for the home
    # team's perspective, one for the away team's perspective ------------
    non_feature_cols = {
        "match_id", "match_date", "season", "league_name", "team", "opponent",
        "is_home", "goals_for", "goals_against", "shots_for", "shots_against",
        "sot_for", "sot_against", "over_2_5", "points",
    }
    feature_cols = [c for c in long.columns if c not in non_feature_cols]

    home_feat = long[long["is_home"] == 1].set_index("match_id")[feature_cols]
    away_feat = long[long["is_home"] == 0].set_index("match_id")[feature_cols]
    home_feat = home_feat.rename(columns=lambda c: f"home_{c}")
    away_feat = away_feat.rename(columns=lambda c: f"away_{c}")

    # Only keep match identity, the target, pre-match-knowable odds (for
    # Phase 7/10 benchmarking only, never as a model feature - see the
    # data dictionary), and the engineered features below. The raw
    # post-match box score (home_goals, away_goals, shots, corners,
    # fouls, cards, result...) is deliberately NOT carried into
    # features.csv: those columns are what determine over_2_5, so if a
    # training script picked one up by accident it would look like a
    # perfect model instead of a bug. They're already in
    # matches_clean.csv if ever needed for backtesting/debugging.
    base_cols = [
        "match_id", "match_date", "season", "league_name",
        "home_team", "away_team", "behind_closed_doors", "over_2_5",
        "home_elo_pre", "away_elo_pre",
        "avg_over_2_5", "avg_under_2_5",
        "bfe_over_2_5", "bfe_under_2_5",
        "bfe_close_over_2_5", "bfe_close_under_2_5",
        "league_over25_avg_to_date", "league_goals_avg_to_date",
    ]
    final = df[base_cols].set_index("match_id").join(home_feat).join(away_feat).copy()

    # --- MATCH-LEVEL features --------------------------------------------
    final["combined_home_goal_rate"] = (
        final["home_overall_goals_for_roll10"] + final["away_overall_goals_against_roll10"]
    ) / 2
    final["combined_away_goal_rate"] = (
        final["away_overall_goals_for_roll10"] + final["home_overall_goals_against_roll10"]
    ) / 2

    league_dummies = pd.get_dummies(final["league_name"], prefix="league")
    final = pd.concat([final, league_dummies], axis=1)

    # --- EDGE CASES --------------------------------------------------------
    final["eligible_for_training"] = (
        (final["home_games_played_career"] >= MIN_GAMES_CAREER_FOR_TRAINING)
        & (final["away_games_played_career"] >= MIN_GAMES_CAREER_FOR_TRAINING)
    ).astype(int)

    final = final.reset_index()
    final = final.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return final


if __name__ == "__main__":
    main()

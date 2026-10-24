"""
Phase 7 - Baselines.

Builds four Over 2.5 probability estimators and compares them on
VALIDATION_SEASONS only (config.py: 2023/24). TEST_SEASONS (2024/25,
2025/26) and the live 2026/27 season are never read by this script -
per Phase 9's rule, test is "touched ONCE, at the end". That's
enforced below, not just promised: rows outside TRAIN/VALIDATION are
dropped before anything is computed.

Baseline A - league historical Over 2.5 rate (fit on TRAIN only)
Baseline B - Poisson goals model, using the Phase 5 combined-goal-rate
             features (already pre-match / leak-free by construction)
Baseline C - market baseline: de-margined bookmaker-average odds
             (avg_over_2_5/avg_under_2_5 - the only odds column with
             coverage in 2023/24; Betfair Exchange odds don't start
             until 2024/25, which is why this isn't BFE here even
             though the checklist calls it "closing odds")
Logistic Regression - trained on the Phase 5 engineered features
             (TRAIN only), evaluated on VALIDATION

Primary metric: Log Loss and Brier Score (Phase 0 principle 1/2),
both lower-is-better, both compared against the market (Baseline C) -
not against a coin flip.

Run from inside src/:
    python baselines.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402

FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")
REPORT_MD = os.path.join("..", "reports", "phase7_baselines.md")
COMPARISON_CSV = os.path.join("..", "reports", "phase7_comparison.csv")

TRAIN_SEASONS = set(config.TRAIN_SEASONS)
VALIDATION_SEASONS = set(config.VALIDATION_SEASONS)
TEST_SEASONS = set(config.TEST_SEASONS)

# Columns that must never be used as a model feature: raw identity,
# the target itself, and anything derived from betting odds (those are
# for Baseline C / Phase 10 only, never a training feature - see the
# Phase 5 data dictionary).
NON_FEATURE_COLS = {
    "match_id", "match_date", "season", "league_name", "home_team", "away_team",
    "over_2_5", "behind_closed_doors",
    "avg_over_2_5", "avg_under_2_5", "bfe_over_2_5", "bfe_under_2_5",
    "bfe_close_over_2_5", "bfe_close_under_2_5",
    "league_over25_avg_to_date", "league_goals_avg_to_date",
}


def md_table(df, float_fmt="{:.4f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: float_fmt.format(v) if pd.notna(v) else "")
        else:
            df[c] = df[c].astype(str)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep] + rows)


def load_split():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])

    # Hard guard: TEST_SEASONS and anything beyond it must not even be
    # loaded into memory here, let alone used - Phase 9 touches test once,
    # at the end, and this script isn't that step.
    df = df[df["season"].isin(TRAIN_SEASONS | VALIDATION_SEASONS)].copy()

    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val = df[df["season"].isin(VALIDATION_SEASONS)].copy()
    return train, val


def baseline_a_league_rate(train, val):
    """League historical Over 2.5 rate, fit on TRAIN only."""
    league_rate = train.groupby("league_name")["over_2_5"].mean()
    preds = val["league_name"].map(league_rate)
    return preds


def baseline_b_poisson(val):
    """Poisson goals model using Phase 5's combined-goal-rate features
    (already pre-match, no fitting needed - they're rolling averages)."""
    lam = val["combined_home_goal_rate"] + val["combined_away_goal_rate"]
    # P(total goals >= 3) = 1 - P(X <= 2) for X ~ Poisson(lambda)
    p_over = 1 - poisson.cdf(2, lam)
    return pd.Series(p_over, index=val.index)


def baseline_c_market(val):
    """De-margined bookmaker-average odds. avg_over_2_5/avg_under_2_5 is
    the only odds column with full coverage in the 2023/24 validation
    season - Betfair Exchange prices don't start until 2024/25 (TEST),
    so this is NOT the Betfair Exchange closing price the checklist
    names; it's the closest equivalent actually available pre-test."""
    implied_over = 1 / val["avg_over_2_5"]
    implied_under = 1 / val["avg_under_2_5"]
    p_over = implied_over / (implied_over + implied_under)
    return p_over


def logistic_regression(train, val):
    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]

    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()

    # Median imputation (fit on TRAIN only, applied to both) - NaNs only
    # occur for a team's literal first-ever match in the dataset, a
    # small, early, TRAIN-only occurrence; nothing in VALIDATION should
    # need it, but the same imputer is applied for safety/consistency.
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    y_train = train["over_2_5"]

    model = LogisticRegression(max_iter=2000, random_state=424410)
    model.fit(X_train_scaled, y_train)

    p_over = model.predict_proba(X_val_scaled)[:, 1]
    return pd.Series(p_over, index=val.index), feature_cols, model, scaler


def evaluate(name, y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, 1e-6, 1 - 1e-6)  # log loss is undefined at exactly 0 or 1
    return {
        "model": name,
        "log_loss": log_loss(y_true, y_pred_clipped),
        "brier_score": brier_score_loss(y_true, y_pred_clipped),
        "n_matches": len(y_true),
    }


def main():
    train, val = load_split()
    print(f"Train: {len(train)} matches ({sorted(TRAIN_SEASONS)})")
    print(f"Validation: {len(val)} matches ({sorted(VALIDATION_SEASONS)})")

    results = []

    pred_a = baseline_a_league_rate(train, val)
    results.append(evaluate("Baseline A - league historical rate", val["over_2_5"], pred_a))

    pred_b = baseline_b_poisson(val)
    mask_b = pred_b.notna()
    if mask_b.sum() < len(val):
        print(f"Baseline B: dropping {len(val) - mask_b.sum()} validation rows with no Poisson rate available")
    results.append(evaluate("Baseline B - Poisson goals model", val.loc[mask_b, "over_2_5"], pred_b[mask_b]))

    val_c = val.dropna(subset=["avg_over_2_5", "avg_under_2_5"])
    pred_c = baseline_c_market(val_c)
    results.append(evaluate("Baseline C - market (de-margined avg odds)", val_c["over_2_5"], pred_c))

    pred_lr, feature_cols, model, scaler = logistic_regression(train, val)
    results.append(evaluate("Logistic Regression (engineered features)", val["over_2_5"], pred_lr))

    comparison = pd.DataFrame(results).sort_values("log_loss")
    comparison.to_csv(COMPARISON_CSV, index=False)

    print("\n" + comparison.to_string(index=False))

    beats_market = comparison[comparison["model"] != "Baseline C - market (de-margined avg odds)"]["log_loss"].min() < \
        comparison[comparison["model"] == "Baseline C - market (de-margined avg odds)"]["log_loss"].values[0]

    lines = []
    lines.append("# Phase 7 - Baselines\n")
    lines.append(f"Evaluated on VALIDATION_SEASONS only ({sorted(VALIDATION_SEASONS)}, "
                  f"{len(val)} matches). TRAIN_SEASONS ({sorted(TRAIN_SEASONS)}, {len(train)} "
                  f"matches) used to fit Baseline A and the Logistic Regression. "
                  f"TEST_SEASONS {sorted(TEST_SEASONS)} were not loaded by this script at all.\n")
    lines.append("## Results (sorted by Log Loss, lower is better)\n")
    lines.append(md_table(comparison) + "\n")
    lines.append(f"**Brier Score** (also lower is better) tells the same story as Log Loss here "
                  f"but penalises confident wrong predictions less harshly - included per Phase 0's "
                  f"primary metric pair.\n")
    lines.append("## Notes\n")
    lines.append("- Baseline A is fit on TRAIN only (per-league Over 2.5 rate), then applied "
                  "as a constant per league to every validation match.\n")
    lines.append("- Baseline B needs no separate fitting: it converts the Phase 5 "
                  "`combined_home_goal_rate`/`combined_away_goal_rate` features (already "
                  "pre-match, shift-based) into a Poisson P(total goals >= 3).\n")
    lines.append("- Baseline C uses `avg_over_2_5`/`avg_under_2_5` (bookmaker average), not "
                  "Betfair Exchange, because BFE odds don't exist yet in the 2023/24 "
                  "validation season (data_dictionary.md) - the real BFE closing-price "
                  "comparison only becomes possible in Phase 9/10 on the TEST seasons.\n")
    lines.append("- Logistic Regression excludes all odds columns from its feature set - "
                  "it only sees the Phase 5 engineered features, never the market's own price, "
                  "so beating Baseline C with it is a genuine test of the feature engineering, "
                  "not the model re-deriving the market's answer from the market's own odds.\n")
    verdict = ("beats" if beats_market else "does NOT beat")
    lines.append(f"\n**Does anything beat the market here?** The best non-market model "
                  f"{verdict} Baseline C on Log Loss in this validation run. Per Phase 0 "
                  f"principle 6 (\"assume any edge above ~5% is a bug until proven otherwise\"), "
                  f"a result where Logistic Regression clearly beats the market this early, with "
                  f"no tuning, would itself be a reason to go back and check for a leak before "
                  f"believing it - not a reason to celebrate.\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {REPORT_MD}")
    print(f"Wrote {COMPARISON_CSV}")


if __name__ == "__main__":
    main()

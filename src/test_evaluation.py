"""
Phase 9 - Proper Testing.

THIS IS THE ONE SCRIPT IN THE PROJECT THAT OPENS TEST_SEASONS. Per
Phase 0/9's rule, test is touched once, at the end - so the model and
its hyperparameters are not tuned here. They're fixed, taken from
Phase 8's extended search (reports/phase8_tuning.md / models/manifest_v2.json):
Random Forest, {max_depth: 15, max_features: 'sqrt', min_samples_leaf: 36,
min_samples_split: 22, n_estimators: 200}, uncalibrated (calibration
failed Phase 8's own check twice).

Final model is trained on TRAIN_SEASONS + VALIDATION_SEASONS combined
(standard practice once a model is selected: validation's job - tuning
- is done, so it's folded back into training data to give the deployed
model as much pre-test history as possible). This is a deliberate choice,
not an accident - flagged here and in the report.

Checklist coverage:
  - Log Loss, Brier Score (primary) vs the market
  - ROC-AUC, Accuracy, Precision, Recall, F1 (secondary)
  - Calibration curve + table by probability bucket
  - Performance by league and by season
  - Walk-forward validation (train on seasons up to N, test on N+1)
  - The headline question: when the model says 70-75%, do ~70-75%
    of those matches actually go Over?

Run from inside src/ (budget a few minutes - walk-forward retrains
the model multiple times):
    python test_evaluation.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    log_loss, brier_score_loss, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402

FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")
REPORT_MD = os.path.join("..", "reports", "phase9_test.md")
FIG_DIR = os.path.join("..", "reports", "figures")
MODELS_DIR = os.path.join("..", "models")

TRAIN_SEASONS = list(config.TRAIN_SEASONS)
VALIDATION_SEASONS = list(config.VALIDATION_SEASONS)
TEST_SEASONS = list(config.TEST_SEASONS)
SEED = 424410

# Locked from Phase 8's extended search (reports/phase8_tuning.md) - NOT
# re-tuned here. Re-tuning against test would be exactly the mistake
# Phase 9's "touched once" rule exists to prevent.
RF_PARAMS = {
    "max_depth": 15,
    "max_features": "sqrt",
    "min_samples_leaf": 36,
    "min_samples_split": 22,
    "n_estimators": 200,
}

NON_FEATURE_COLS = {
    "match_id", "match_date", "season", "league_name", "home_team", "away_team",
    "over_2_5", "behind_closed_doors",
    "avg_over_2_5", "avg_under_2_5", "bfe_over_2_5", "bfe_under_2_5",
    "bfe_close_over_2_5", "bfe_close_under_2_5",
    "league_over25_avg_to_date", "league_goals_avg_to_date",
}

ALL_SEASONS_CHRONOLOGICAL = [
    "2017/18", "2018/19", "2019/20", "2020/21", "2021/22",
    "2022/23", "2023/24", "2024/25", "2025/26",
]
WALK_FORWARD_MIN_TRAIN_SEASONS = 3  # don't bother predicting a season off <3 seasons of history


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


def load_all():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])
    return df


def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def fit_predict(train_df, pred_df, feature_cols, medians=None):
    if medians is None:
        medians = train_df[feature_cols].median(numeric_only=True)
    X_train = train_df[feature_cols].fillna(medians)
    X_pred = pred_df[feature_cols].fillna(medians)
    model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **RF_PARAMS)
    model.fit(X_train, train_df["over_2_5"])
    pred = model.predict_proba(X_pred)[:, 1]
    return model, pred, medians


def full_metrics(y_true, y_pred, threshold=0.5):
    p = np.clip(y_pred, 1e-6, 1 - 1e-6)
    pred_class = (p >= threshold).astype(int)
    return {
        "log_loss": log_loss(y_true, p),
        "brier_score": brier_score_loss(y_true, p),
        "roc_auc": roc_auc_score(y_true, p),
        "accuracy": accuracy_score(y_true, pred_class),
        "precision": precision_score(y_true, pred_class, zero_division=0),
        "recall": recall_score(y_true, pred_class, zero_division=0),
        "f1": f1_score(y_true, pred_class, zero_division=0),
        "n_matches": len(y_true),
    }


def bucket_reliability(probs, outcomes, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "pred_mid": probs[mask].mean(),
            "actual_rate": outcomes[mask].mean(),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_all()
    feature_cols = get_feature_cols(df)

    final_train = df[df["season"].isin(TRAIN_SEASONS + VALIDATION_SEASONS)].copy()
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    print(f"Final train (TRAIN+VALIDATION): {len(final_train)} matches "
          f"({sorted(TRAIN_SEASONS + VALIDATION_SEASONS)})")
    print(f"TEST: {len(test)} matches ({sorted(TEST_SEASONS)})")

    model, pred_test, medians = fit_predict(final_train, test, feature_cols)
    y_test = test["over_2_5"].values

    # --- Primary + secondary metrics, overall ----------------------------
    overall = full_metrics(y_test, pred_test)
    print("\nOverall TEST metrics:")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Market comparison: real Betfair Exchange CLOSING odds, finally -
    # available (bfe_close_over_2_5/bfe_close_under_2_5) only from 2024/25
    # onward - this is the first point in the project where the checklist's
    # literal "closing odds" comparison is actually possible.
    market = test.dropna(subset=["bfe_close_over_2_5", "bfe_close_under_2_5"]).copy()
    implied_over = 1 / market["bfe_close_over_2_5"]
    implied_under = 1 / market["bfe_close_under_2_5"]
    market_pred = implied_over / (implied_over + implied_under)
    market_metrics = full_metrics(market["over_2_5"].values, market_pred.values)
    print(f"\nMarket (Betfair Exchange closing odds) on the {len(market)} matches with coverage:")
    for k, v in market_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Model's own metrics restricted to the same subset, for a fair apples-
    # to-apples comparison against the market (market has less coverage
    # than the full test set - see data_dictionary.md).
    model_on_market_subset = full_metrics(market["over_2_5"].values, pred_test[test.index.isin(market.index)])

    # --- Performance by season --------------------------------------------
    by_season_rows = []
    for season in sorted(test["season"].unique()):
        mask = test["season"] == season
        m = full_metrics(y_test[mask.values], pred_test[mask.values])
        by_season_rows.append({"season": season, **m})
    by_season = pd.DataFrame(by_season_rows)

    # --- Performance by league ----------------------------------------------
    by_league_rows = []
    for league in sorted(test["league_name"].unique()):
        mask = test["league_name"] == league
        m = full_metrics(y_test[mask.values], pred_test[mask.values])
        by_league_rows.append({"league_name": league, **m})
    by_league = pd.DataFrame(by_league_rows)

    # --- Calibration: curve + table by probability bucket -----------------
    reliability = bucket_reliability(pred_test, y_test)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.scatter(reliability["pred_mid"], reliability["actual_rate"], s=reliability["n"] / 3, alpha=0.75)
    ax.set_xlabel("Predicted probability (Random Forest)")
    ax.set_ylabel("Actual Over 2.5 rate")
    ax.set_title("Phase 9 - TEST calibration (2024/25 + 2025/26)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "phase9_test_calibration.png"), dpi=120)
    plt.close(fig)

    # --- Walk-forward validation --------------------------------------------
    print("\nWalk-forward validation...")
    wf_rows = []
    for i in range(WALK_FORWARD_MIN_TRAIN_SEASONS, len(ALL_SEASONS_CHRONOLOGICAL)):
        train_seasons_wf = ALL_SEASONS_CHRONOLOGICAL[:i]
        test_season_wf = ALL_SEASONS_CHRONOLOGICAL[i]
        wf_train = df[df["season"].isin(train_seasons_wf)]
        wf_test = df[df["season"] == test_season_wf]
        if len(wf_test) == 0:
            continue
        _, wf_pred, _ = fit_predict(wf_train, wf_test, feature_cols)
        m = full_metrics(wf_test["over_2_5"].values, wf_pred)
        wf_rows.append({"train_through": train_seasons_wf[-1], "predicts": test_season_wf, **m})
        print(f"  train through {train_seasons_wf[-1]} ({len(wf_train)} matches) "
              f"-> predict {test_season_wf}: log_loss={m['log_loss']:.4f}")
    walk_forward = pd.DataFrame(wf_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(walk_forward["predicts"], walk_forward["log_loss"], marker="o", label="Random Forest")
    ax.axhline(0.693, color="gray", linestyle=":", label="coin-flip floor (ln 2)")
    ax.set_xlabel("Season predicted")
    ax.set_ylabel("Log Loss")
    ax.set_title("Walk-forward stability: train on seasons up to N, predict N+1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "phase9_walk_forward.png"), dpi=120)
    plt.close(fig)

    # --- Save the frozen test-evaluated model -------------------------------
    joblib.dump(model, os.path.join(MODELS_DIR, "random_forest_phase9_test.joblib"))
    joblib.dump(medians, os.path.join(MODELS_DIR, "feature_medians_phase9_test.joblib"))

    does_it_beat_market = overall["log_loss"] < market_metrics["log_loss"]

    results_summary = {
        "trained_on": sorted(TRAIN_SEASONS + VALIDATION_SEASONS),
        "tested_on": sorted(TEST_SEASONS),
        "model": "Random Forest",
        "hyperparameters": RF_PARAMS,
        "calibration_applied": False,
        "overall_test_metrics": overall,
        "market_closing_odds_metrics": market_metrics,
        "model_on_market_coverage_subset": model_on_market_subset,
        "beats_market_on_log_loss": bool(does_it_beat_market),
        "by_season": by_season.to_dict(orient="records"),
        "by_league": by_league.to_dict(orient="records"),
        "reliability_table": reliability.to_dict(orient="records"),
        "walk_forward": walk_forward.to_dict(orient="records"),
    }
    with open(os.path.join(MODELS_DIR, "phase9_test_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # --- Report --------------------------------------------------------------
    lines = []
    lines.append("# Phase 9 - Proper Testing\n")
    lines.append(f"**This is the one evaluation against TEST_SEASONS ({sorted(TEST_SEASONS)}) in "
                  f"the whole project.** Model and hyperparameters were locked before this ran "
                  f"(Phase 8 extended search), not tuned here.\n")
    lines.append(f"Final model trained on TRAIN+VALIDATION combined "
                  f"({sorted(TRAIN_SEASONS + VALIDATION_SEASONS)}, {len(final_train)} matches): "
                  f"Random Forest, `{RF_PARAMS}`, uncalibrated.\n")

    lines.append("## Overall TEST metrics\n")
    lines.append(md_table(pd.DataFrame([{"metric": k, "value": v} for k, v in overall.items()])) + "\n")

    lines.append("## vs. the market (Betfair Exchange CLOSING odds)\n")
    lines.append(f"The real closing-price comparison the checklist asks for only becomes possible "
                  f"from 2024/25 onward ({len(market)} of {len(test)} test matches have `bfe_close_*` "
                  f"coverage - see `data_dictionary.md`).\n")
    comp_table = pd.DataFrame([
        {"source": "Market (BFE closing odds)", **market_metrics},
        {"source": "Random Forest (same matches)", **model_on_market_subset},
    ])
    lines.append(md_table(comp_table) + "\n")
    verdict = "BEATS" if model_on_market_subset["log_loss"] < market_metrics["log_loss"] else "does NOT beat"
    lines.append(f"\n**On Log Loss, the model {verdict} the market** on the matches where both "
                  f"can be compared. Per Phase 0 principle 6, a win here should be treated with "
                  f"suspicion, not celebration, until it holds up over many more live matches - "
                  f"this is {len(market)} matches, a small sample for a market-beating claim.\n")

    lines.append("\n## Performance by season\n")
    lines.append(md_table(by_season) + "\n")
    lines.append("\n## Performance by league\n")
    lines.append(md_table(by_league) + "\n")

    lines.append("\n## Calibration - the most important question\n")
    lines.append("**When the model says 70-75%, do roughly 70-75% of those matches actually go "
                  "Over?** The table below is the direct answer, bucket by bucket:\n")
    lines.append(md_table(reliability) + "\n")
    lines.append("\n![TEST calibration](figures/phase9_test_calibration.png)\n")

    lines.append("\n## Walk-forward validation (stability across seasons)\n")
    lines.append("Same locked hyperparameters throughout - only the training window grows. "
                  "Shows whether performance is consistently close to the market or swings "
                  "wildly season to season (a swing would suggest luck, not skill).\n")
    lines.append(md_table(walk_forward[["train_through", "predicts", "log_loss", "brier_score", "roc_auc", "n_matches"]]) + "\n")
    lines.append("\n![Walk-forward](figures/phase9_walk_forward.png)\n")

    lines.append("\n## Saved\n")
    lines.append("`models/random_forest_phase9_test.joblib` (the exact model evaluated above) + "
                  "`models/phase9_test_results.json` (every number in this report, machine-readable).\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {REPORT_MD}, 2 figures, models/random_forest_phase9_test.joblib, "
          f"models/phase9_test_results.json")


if __name__ == "__main__":
    main()

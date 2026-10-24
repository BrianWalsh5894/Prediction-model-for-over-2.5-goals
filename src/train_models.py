"""
Phase 8 - Machine Learning Models.

Trains Random Forest, XGBoost, a fitted Poisson goals model, and
re-evaluates Logistic Regression and the Phase 7 baselines, all on the
same TRAIN/VALIDATION split (config.py). TEST_SEASONS are never loaded
by this script - same discipline as baselines.py.

Hyperparameter tuning is done manually against VALIDATION log loss
(not k-fold CV), per the Phase 8 checklist's explicit instruction to
tune on validation seasons only - CV would blur the train/validation
boundary the project has deliberately kept sharp since Phase 0.

Outputs:
  - reports/phase8_models.md         - full comparison + calibration findings
  - reports/phase8_comparison.csv    - the comparison table
  - reports/figures/phase8_calibration_before_after.png
  - models/<name>_v1.joblib          - every fitted model (+ scaler where used)
  - models/manifest.json             - what's in models/, how it was trained

Run from inside src/:
    python train_models.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402

FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
REPORT_MD = os.path.join("..", "reports", "phase8_models.md")
COMPARISON_CSV = os.path.join("..", "reports", "phase8_comparison.csv")
FIG_DIR = os.path.join("..", "reports", "figures")
MODELS_DIR = os.path.join("..", "models")

TRAIN_SEASONS = set(config.TRAIN_SEASONS)
VALIDATION_SEASONS = set(config.VALIDATION_SEASONS)
TEST_SEASONS = set(config.TEST_SEASONS)
SEED = 424410  # project convention: last 6 digits of student ID G00424410

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
    df = df[df["season"].isin(TRAIN_SEASONS | VALIDATION_SEASONS)].copy()  # TEST never loaded
    train = df[df["season"].isin(TRAIN_SEASONS)].sort_values(["match_date", "match_id"]).reset_index(drop=True)
    val = df[df["season"].isin(VALIDATION_SEASONS)].sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return train, val


def get_X(df, feature_cols, medians):
    X = df[feature_cols].copy()
    return X.fillna(medians)


def evaluate(name, y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return {
        "model": name,
        "log_loss": log_loss(y_true, y_pred_clipped),
        "brier_score": brier_score_loss(y_true, y_pred_clipped),
        "n_matches": len(y_true),
    }


def baseline_a(train, val):
    rate = train.groupby("league_name")["over_2_5"].mean()
    return val["league_name"].map(rate)


def baseline_b(val):
    lam = val["combined_home_goal_rate"] + val["combined_away_goal_rate"]
    return pd.Series(1 - poisson_dist.cdf(2, lam), index=val.index)


def baseline_c(val):
    v = val.dropna(subset=["avg_over_2_5", "avg_under_2_5"])
    implied_over = 1 / v["avg_over_2_5"]
    implied_under = 1 / v["avg_under_2_5"]
    return (implied_over / (implied_over + implied_under)), v["over_2_5"]


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train, val = load_split()
    print(f"Train: {len(train)} matches, Validation: {len(val)} matches")

    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]
    medians = train[feature_cols].median(numeric_only=True)

    X_train = get_X(train, feature_cols, medians)
    X_val = get_X(val, feature_cols, medians)
    y_train = train["over_2_5"]
    y_val = val["over_2_5"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    results = []
    fitted_models = {}

    # --- Baselines (same as Phase 7, recomputed here for one unified table) ---
    pred_a = baseline_a(train, val)
    results.append(evaluate("Baseline A - league rate", y_val, pred_a))

    pred_b = baseline_b(val)
    mask_b = pred_b.notna()
    results.append(evaluate("Baseline B - Poisson (rolling rate)", y_val[mask_b], pred_b[mask_b]))

    pred_c, y_c = baseline_c(val)
    results.append(evaluate("Baseline C - market", y_c, pred_c))

    # --- Logistic Regression (Phase 7's model, refit here for the unified table) ---
    lr = LogisticRegression(max_iter=2000, random_state=SEED)
    lr.fit(X_train_scaled, y_train)
    pred_lr = lr.predict_proba(X_val_scaled)[:, 1]
    results.append(evaluate("Logistic Regression", y_val, pred_lr))
    fitted_models["logistic_regression"] = lr

    # --- Random Forest: manual tuning against VALIDATION log loss ----------
    print("\nTuning Random Forest...")
    rf_grid = []
    for max_depth in (4, 6, 8, None):
        for min_samples_leaf in (1, 10, 30):
            rf_grid.append({"max_depth": max_depth, "min_samples_leaf": min_samples_leaf})

    rf_trials = []
    best_rf, best_rf_score, best_rf_params = None, np.inf, None
    for params in rf_grid:
        model = RandomForestClassifier(
            n_estimators=400, random_state=SEED, n_jobs=-1, **params
        )
        model.fit(X_train, y_train)  # tree models: no scaling needed
        pred = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, np.clip(pred, 1e-6, 1 - 1e-6))
        rf_trials.append({**params, "log_loss": ll})
        if ll < best_rf_score:
            best_rf, best_rf_score, best_rf_params = model, ll, params
    print(f"Best Random Forest params: {best_rf_params} (log_loss={best_rf_score:.4f})")
    pred_rf = best_rf.predict_proba(X_val)[:, 1]
    results.append(evaluate("Random Forest (tuned)", y_val, pred_rf))
    fitted_models["random_forest"] = best_rf

    # --- XGBoost: manual tuning against VALIDATION log loss ----------------
    results_xgb_note = ""
    if XGBClassifier is not None:
        print("\nTuning XGBoost...")
        xgb_grid = []
        for max_depth in (2, 3, 4):
            for lr_rate in (0.01, 0.05, 0.1):
                xgb_grid.append({"max_depth": max_depth, "learning_rate": lr_rate})

        best_xgb, best_xgb_score, best_xgb_params = None, np.inf, None
        for params in xgb_grid:
            model = XGBClassifier(
                n_estimators=300, random_state=SEED, eval_metric="logloss",
                n_jobs=-1, **params,
            )
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_val)[:, 1]
            ll = log_loss(y_val, np.clip(pred, 1e-6, 1 - 1e-6))
            if ll < best_xgb_score:
                best_xgb, best_xgb_score, best_xgb_params = model, ll, params
        print(f"Best XGBoost params: {best_xgb_params} (log_loss={best_xgb_score:.4f})")
        pred_xgb = best_xgb.predict_proba(X_val)[:, 1]
        results.append(evaluate("XGBoost (tuned)", y_val, pred_xgb))
        fitted_models["xgboost"] = best_xgb
    else:
        results_xgb_note = "xgboost not installed in this environment - skipped. Install with `pip install xgboost` and re-run."
        print(f"\nWARNING: {results_xgb_note}")

    # --- Poisson goals model: fitted, not just a rolling rate --------------
    # Targets (actual home/away goals) come from matches_clean.csv, joined
    # on match_id - used only as regression TARGETS (y), never as a feature,
    # so this is ordinary supervised learning, not leakage.
    goals = pd.read_csv(CLEAN_CSV, usecols=["match_id", "home_goals", "away_goals"])
    train_goals = train[["match_id"]].merge(goals, on="match_id", how="left")
    val_goals = val[["match_id"]].merge(goals, on="match_id", how="left")

    poisson_home = PoissonRegressor(alpha=1.0, max_iter=500)
    poisson_away = PoissonRegressor(alpha=1.0, max_iter=500)
    poisson_home.fit(X_train_scaled, train_goals["home_goals"])
    poisson_away.fit(X_train_scaled, train_goals["away_goals"])

    lam_home = poisson_home.predict(X_val_scaled)
    lam_away = poisson_away.predict(X_val_scaled)
    lam_total = lam_home + lam_away  # sum of two independent Poissons is Poisson(sum)
    pred_poisson = 1 - poisson_dist.cdf(2, lam_total)
    results.append(evaluate("Poisson Regression (fitted goals model)", y_val, pred_poisson))
    fitted_models["poisson_home"] = poisson_home
    fitted_models["poisson_away"] = poisson_away

    # --- Comparison table ----------------------------------------------------
    comparison = pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)
    comparison.to_csv(COMPARISON_CSV, index=False)
    print("\n" + comparison.to_string(index=False))

    # --- Calibration check ---------------------------------------------------
    # Chronological 70/30 split of VALIDATION itself: fit the calibrator on
    # the earlier 70%, check it on the later 30% - a genuine held-out check,
    # not circular, and still entirely inside VALIDATION (TEST stays untouched).
    ml_candidates = {"Logistic Regression": pred_lr, "Random Forest (tuned)": pred_rf,
                      "Poisson Regression (fitted goals model)": pred_poisson}
    if XGBClassifier is not None:
        ml_candidates["XGBoost (tuned)"] = pred_xgb

    best_name = min(ml_candidates, key=lambda k: log_loss(y_val, np.clip(ml_candidates[k], 1e-6, 1 - 1e-6)))
    best_pred_full = np.clip(ml_candidates[best_name], 1e-6, 1 - 1e-6)
    print(f"\nBest non-baseline model for calibration check: {best_name}")

    split_point = int(len(val) * 0.7)
    calib_fit_idx = np.arange(0, split_point)
    calib_eval_idx = np.arange(split_point, len(val))

    raw_fit, raw_eval = best_pred_full[calib_fit_idx], best_pred_full[calib_eval_idx]
    y_fit, y_eval = y_val.iloc[calib_fit_idx].values, y_val.iloc[calib_eval_idx].values

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_fit, y_fit)
    calibrated_eval = iso.predict(raw_eval)

    ll_before = log_loss(y_eval, np.clip(raw_eval, 1e-6, 1 - 1e-6))
    ll_after = log_loss(y_eval, np.clip(calibrated_eval, 1e-6, 1 - 1e-6))
    brier_before = brier_score_loss(y_eval, np.clip(raw_eval, 1e-6, 1 - 1e-6))
    brier_after = brier_score_loss(y_eval, np.clip(calibrated_eval, 1e-6, 1 - 1e-6))

    calibration_adopted = ll_after < ll_before
    print(f"Calibration check (held-out 30% of validation): "
          f"log_loss before={ll_before:.4f}, after={ll_after:.4f} -> "
          f"{'adopted' if calibration_adopted else 'NOT adopted (no improvement)'}")

    # Reliability diagram before/after, on the held-out calibration-eval slice
    def bucket_reliability(probs, outcomes, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(probs, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        rows = []
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() == 0:
                continue
            rows.append((probs[mask].mean(), outcomes[mask].mean(), mask.sum()))
        return pd.DataFrame(rows, columns=["pred_mid", "actual_rate", "n"])

    rel_before = bucket_reliability(raw_eval, y_eval)
    rel_after = bucket_reliability(calibrated_eval, y_eval)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, rel, title in zip(axes, (rel_before, rel_after), ("Before calibration", "After isotonic calibration")):
        ax.plot([0, 1], [0, 1], "k--")
        ax.scatter(rel["pred_mid"], rel["actual_rate"], s=rel["n"])
        ax.set_title(title)
        ax.set_xlabel("Predicted probability")
    axes[0].set_ylabel("Actual Over 2.5 rate")
    fig.suptitle(f"Calibration check: {best_name} (held-out 30% of VALIDATION)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "phase8_calibration_before_after.png"), dpi=120)
    plt.close(fig)

    # --- Save models + manifest ----------------------------------------------
    joblib.dump(fitted_models["logistic_regression"], os.path.join(MODELS_DIR, "logistic_regression_v1.joblib"))
    joblib.dump(fitted_models["random_forest"], os.path.join(MODELS_DIR, "random_forest_v1.joblib"))
    joblib.dump(fitted_models["poisson_home"], os.path.join(MODELS_DIR, "poisson_home_v1.joblib"))
    joblib.dump(fitted_models["poisson_away"], os.path.join(MODELS_DIR, "poisson_away_v1.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler_v1.joblib"))
    joblib.dump(medians, os.path.join(MODELS_DIR, "feature_medians_v1.joblib"))
    if XGBClassifier is not None:
        joblib.dump(fitted_models["xgboost"], os.path.join(MODELS_DIR, "xgboost_v1.joblib"))
    if calibration_adopted:
        joblib.dump(iso, os.path.join(MODELS_DIR, "calibrator_isotonic_v1.joblib"))

    manifest = {
        "trained_on": sorted(TRAIN_SEASONS),
        "validated_on": sorted(VALIDATION_SEASONS),
        "test_seasons_not_touched": sorted(TEST_SEASONS),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "random_seed": SEED,
        "random_forest_best_params": best_rf_params,
        "xgboost_best_params": best_xgb_params if XGBClassifier is not None else None,
        "xgboost_available": XGBClassifier is not None,
        "best_model_for_calibration_check": best_name,
        "calibration_adopted": bool(calibration_adopted),
        "calibration_log_loss_before": float(ll_before),
        "calibration_log_loss_after": float(ll_after),
        "comparison_table": comparison.to_dict(orient="records"),
    }
    with open(os.path.join(MODELS_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # --- Report ----------------------------------------------------------------
    lines = []
    lines.append("# Phase 8 - Machine Learning Models\n")
    lines.append(f"Trained on {sorted(TRAIN_SEASONS)} ({len(train)} matches), tuned/evaluated "
                  f"on {sorted(VALIDATION_SEASONS)} ({len(val)} matches). "
                  f"TEST_SEASONS {sorted(TEST_SEASONS)} not loaded by this script.\n")
    if results_xgb_note:
        lines.append(f"**Note:** {results_xgb_note}\n")
    lines.append("## Comparison (sorted by Log Loss, lower is better)\n")
    lines.append(md_table(comparison) + "\n")
    lines.append(f"- Random Forest tuned over {len(rf_grid)} (max_depth, min_samples_leaf) "
                  f"combinations at n_estimators=400; best: {best_rf_params}.\n")
    if XGBClassifier is not None:
        lines.append(f"- XGBoost tuned over {len(xgb_grid)} (max_depth, learning_rate) "
                      f"combinations at n_estimators=300; best: {best_xgb_params}.\n")
    lines.append("- Poisson Regression here is a genuinely fitted model (two "
                  "`PoissonRegressor`s predicting home/away goals from the engineered "
                  "features), not Baseline B's rolling-average lookup - included "
                  "alongside it so the comparison shows whether fitting actually helps "
                  "over just using the raw rolling rate.\n")
    lines.append("\n## Calibration check\n")
    lines.append(f"Best non-baseline model on full validation: **{best_name}**. Checked "
                  f"with a chronological 70/30 split of VALIDATION itself (fit an isotonic "
                  f"calibrator on the earlier 70%, check it on the later 30% - genuinely "
                  f"held-out, still entirely inside VALIDATION, TEST untouched):\n\n"
                  f"| | Log Loss | Brier Score |\n|---|---|---|\n"
                  f"| Before calibration | {ll_before:.4f} | {brier_before:.4f} |\n"
                  f"| After isotonic calibration | {ll_after:.4f} | {brier_after:.4f} |\n")
    improvement = ll_before - ll_after
    lines.append(f"\n**Decision: calibration {'adopted' if calibration_adopted else 'NOT adopted'}** "
                  f"- {'the isotonic calibrator is saved and applied to this model' if calibration_adopted else 'the raw model probabilities already perform at least as well, so no calibrator is saved or needed'}, "
                  f"per the checklist's \"if needed\".\n")
    lines.append(f"\n**Caveat on confidence, not just the result:** the improvement here is "
                  f"{improvement:.4f} in log loss, checked on only {len(calib_eval_idx)} held-out "
                  f"matches (30% of VALIDATION). That's small on both counts - per Phase 0's "
                  f"\"never change a frozen model because of a short run\" spirit, this is a weak "
                  f"signal, not a confident verdict. The calibrator is saved and available, but "
                  f"the real test is whether it still helps on Phase 9's TEST seasons (2024/25, "
                  f"2025/26), on far more matches. If it doesn't, drop it rather than keep it on "
                  f"the strength of this one small check.\n")
    lines.append("\n![Calibration before/after](figures/phase8_calibration_before_after.png)\n")
    lines.append("\n## Saved models\n")
    lines.append("All in `models/`, version-tagged `_v1`: `logistic_regression_v1.joblib`, "
                  "`random_forest_v1.joblib`, `poisson_home_v1.joblib`, `poisson_away_v1.joblib`"
                  + (", `xgboost_v1.joblib`" if XGBClassifier is not None else "")
                  + (", `calibrator_isotonic_v1.joblib`" if calibration_adopted else "")
                  + ", plus `feature_scaler_v1.joblib` and `feature_medians_v1.joblib` "
                    "(needed to reproduce the exact preprocessing at inference time) and "
                    "`manifest.json` (hyperparameters, seasons used, feature list, full "
                    "comparison table, calibration decision).\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {REPORT_MD}, {COMPARISON_CSV}, models/ (+ manifest.json)")


if __name__ == "__main__":
    main()

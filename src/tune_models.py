"""
Phase 8 (extended) - wider hyperparameter search.

Supersedes train_models.py's narrow grids (12 Random Forest combos,
9 XGBoost combos, untuned Poisson, untuned Logistic Regression) with a
much wider random search over all four models, still scored purely on
VALIDATION_SEASONS log loss - same discipline as before, just more of
it. TEST_SEASONS are never loaded.

Why: Random Forest won the narrow Phase 8 search, but that search gave
XGBoost, Poisson and Logistic Regression far less room to show what
they can do than Random Forest got. This widens all four before
anything gets locked in for Phase 9's one-shot test evaluation.

Outputs:
  - reports/phase8_tuning.md            - full search + final comparison
  - reports/phase8_tuning_comparison.csv
  - reports/figures/phase8_tuning_calibration.png
  - models/<name>_v2.joblib             - best of each model type
  - models/manifest_v2.json

Run from inside src/ (takes several minutes - this is a much wider
search than train_models.py):
    python tune_models.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.stats import poisson as poisson_dist, loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
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
REPORT_MD = os.path.join("..", "reports", "phase8_tuning.md")
COMPARISON_CSV = os.path.join("..", "reports", "phase8_tuning_comparison.csv")
FIG_DIR = os.path.join("..", "reports", "figures")
MODELS_DIR = os.path.join("..", "models")

TRAIN_SEASONS = set(config.TRAIN_SEASONS)
VALIDATION_SEASONS = set(config.VALIDATION_SEASONS)
TEST_SEASONS = set(config.TEST_SEASONS)
SEED = 424410

RF_N_ITER = 30
XGB_N_ITER = 25

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


def clean_params(params):
    """Convert numpy scalar types to plain Python types so they print and
    JSON-serialize cleanly (np.float64(0.123) instead of 0.123, etc.)."""
    cleaned = {}
    for k, v in params.items():
        if isinstance(v, np.floating):
            cleaned[k] = round(float(v), 6)
        elif isinstance(v, np.integer):
            cleaned[k] = int(v)
        else:
            cleaned[k] = v
    return cleaned


def load_split():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])
    df = df[df["season"].isin(TRAIN_SEASONS | VALIDATION_SEASONS)].copy()
    train = df[df["season"].isin(TRAIN_SEASONS)].sort_values(["match_date", "match_id"]).reset_index(drop=True)
    val = df[df["season"].isin(VALIDATION_SEASONS)].sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return train, val


def evaluate(name, y_true, y_pred):
    p = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return {"model": name, "log_loss": log_loss(y_true, p), "brier_score": brier_score_loss(y_true, p), "n_matches": len(y_true)}


def tune_random_forest(X_train, y_train, X_val, y_val):
    param_dist = {
        "n_estimators": [200, 400],
        "max_depth": [3, 4, 5, 6, 8, 10, 15, None],
        "min_samples_leaf": randint(1, 60),
        "min_samples_split": randint(2, 60),
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
    }
    sampler = list(ParameterSampler(param_dist, n_iter=RF_N_ITER, random_state=SEED))

    best_model, best_score, best_params, trials = None, np.inf, None, []
    for i, params in enumerate(sampler):
        model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, np.clip(pred, 1e-6, 1 - 1e-6))
        trials.append({**params, "log_loss": ll})
        if ll < best_score:
            best_model, best_score, best_params = model, ll, clean_params(params)
    return best_model, best_score, best_params, trials


def tune_xgboost(X_train, y_train, X_val, y_val):
    if XGBClassifier is None:
        return None, None, None, []
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": loguniform(0.005, 0.3),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(0.1, 10),
        "min_child_weight": randint(1, 20),
    }
    sampler = list(ParameterSampler(param_dist, n_iter=XGB_N_ITER, random_state=SEED))

    best_model, best_score, best_params, trials = None, np.inf, None, []
    for params in sampler:
        model = XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1, **params)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, np.clip(pred, 1e-6, 1 - 1e-6))
        trials.append({**params, "log_loss": ll})
        if ll < best_score:
            best_model, best_score, best_params = model, ll, clean_params(params)
    return best_model, best_score, best_params, trials


def tune_poisson(X_train_scaled, y_home, y_away, X_val_scaled, y_val):
    alphas = np.logspace(-3, 2, 25)
    best = (None, None, np.inf, None)
    trials = []
    for alpha in alphas:
        m_home = PoissonRegressor(alpha=alpha, max_iter=1000)
        m_away = PoissonRegressor(alpha=alpha, max_iter=1000)
        m_home.fit(X_train_scaled, y_home)
        m_away.fit(X_train_scaled, y_away)
        lam = m_home.predict(X_val_scaled) + m_away.predict(X_val_scaled)
        pred = 1 - poisson_dist.cdf(2, lam)
        ll = log_loss(y_val, np.clip(pred, 1e-6, 1 - 1e-6))
        trials.append({"alpha": alpha, "log_loss": ll})
        if ll < best[2]:
            best = (m_home, m_away, ll, alpha)
    return best, trials


def tune_logistic(X_train_scaled, y_train, X_val_scaled, y_val):
    Cs = np.logspace(-3, 3, 13)
    penalties = ["l1", "l2"]
    best_model, best_score, best_params, trials = None, np.inf, None, []
    for C in Cs:
        for penalty in penalties:
            model = LogisticRegression(C=C, penalty=penalty, solver="liblinear", max_iter=2000, random_state=SEED)
            model.fit(X_train_scaled, y_train)
            pred = model.predict_proba(X_val_scaled)[:, 1]
            ll = log_loss(y_val, np.clip(pred, 1e-6, 1 - 1e-6))
            trials.append({"C": C, "penalty": penalty, "log_loss": ll})
            if ll < best_score:
                best_model, best_score, best_params = model, ll, clean_params({"C": C, "penalty": penalty})
    return best_model, best_score, best_params, trials


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train, val = load_split()
    print(f"Train: {len(train)} matches, Validation: {len(val)} matches")

    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]
    medians = train[feature_cols].median(numeric_only=True)
    X_train = train[feature_cols].fillna(medians)
    X_val = val[feature_cols].fillna(medians)
    y_train = train["over_2_5"]
    y_val = val["over_2_5"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    goals = pd.read_csv(CLEAN_CSV, usecols=["match_id", "home_goals", "away_goals"])
    train_goals = train[["match_id"]].merge(goals, on="match_id", how="left")

    results = []

    t0 = time.time()
    print(f"\nTuning Random Forest ({RF_N_ITER} random configs)...")
    rf_model, rf_score, rf_params, rf_trials = tune_random_forest(X_train, y_train, X_val, y_val)
    print(f"  best: {rf_params} -> log_loss={rf_score:.4f} ({time.time()-t0:.0f}s)")
    results.append(evaluate("Random Forest (wide search)", y_val, rf_model.predict_proba(X_val)[:, 1]))

    t0 = time.time()
    print(f"\nTuning XGBoost ({XGB_N_ITER} random configs)...")
    xgb_model, xgb_score, xgb_params, xgb_trials = tune_xgboost(X_train, y_train, X_val, y_val)
    if xgb_model is not None:
        print(f"  best: {xgb_params} -> log_loss={xgb_score:.4f} ({time.time()-t0:.0f}s)")
        results.append(evaluate("XGBoost (wide search)", y_val, xgb_model.predict_proba(X_val)[:, 1]))
    else:
        print("  xgboost not installed - skipped")

    t0 = time.time()
    print("\nTuning Poisson Regression (25-point alpha grid)...")
    (poisson_home, poisson_away, poisson_score, poisson_alpha), poisson_trials = tune_poisson(
        X_train_scaled, train_goals["home_goals"], train_goals["away_goals"], X_val_scaled, y_val
    )
    print(f"  best alpha={poisson_alpha:.4f} -> log_loss={poisson_score:.4f} ({time.time()-t0:.0f}s)")
    lam_val = poisson_home.predict(X_val_scaled) + poisson_away.predict(X_val_scaled)
    pred_poisson = 1 - poisson_dist.cdf(2, lam_val)
    results.append(evaluate("Poisson Regression (wide search)", y_val, pred_poisson))

    t0 = time.time()
    print("\nTuning Logistic Regression (C x L1/L2 grid)...")
    lr_model, lr_score, lr_params, lr_trials = tune_logistic(X_train_scaled, y_train, X_val_scaled, y_val)
    print(f"  best: {lr_params} -> log_loss={lr_score:.4f} ({time.time()-t0:.0f}s)")
    results.append(evaluate("Logistic Regression (wide search)", y_val, lr_model.predict_proba(X_val_scaled)[:, 1]))

    # --- baselines, unchanged, for context -----------------------------
    league_rate = train.groupby("league_name")["over_2_5"].mean()
    pred_a = val["league_name"].map(league_rate)
    results.append(evaluate("Baseline A - league rate", y_val, pred_a))

    lam_b = val["combined_home_goal_rate"] + val["combined_away_goal_rate"]
    pred_b = pd.Series(1 - poisson_dist.cdf(2, lam_b), index=val.index)
    mask_b = pred_b.notna()
    results.append(evaluate("Baseline B - Poisson (rolling rate)", y_val[mask_b], pred_b[mask_b]))

    val_c = val.dropna(subset=["avg_over_2_5", "avg_under_2_5"])
    implied_over = 1 / val_c["avg_over_2_5"]
    implied_under = 1 / val_c["avg_under_2_5"]
    pred_c = implied_over / (implied_over + implied_under)
    results.append(evaluate("Baseline C - market", val_c["over_2_5"], pred_c))

    comparison = pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)
    comparison.to_csv(COMPARISON_CSV, index=False)
    print("\n" + comparison.to_string(index=False))

    # --- best candidate overall (excluding baselines) -----------------
    candidate_names = {
        "Random Forest (wide search)": rf_model.predict_proba(X_val)[:, 1],
        "Poisson Regression (wide search)": pred_poisson,
        "Logistic Regression (wide search)": lr_model.predict_proba(X_val_scaled)[:, 1],
    }
    if xgb_model is not None:
        candidate_names["XGBoost (wide search)"] = xgb_model.predict_proba(X_val)[:, 1]

    best_name = min(candidate_names, key=lambda k: log_loss(y_val, np.clip(candidate_names[k], 1e-6, 1 - 1e-6)))
    best_pred_full = np.clip(candidate_names[best_name], 1e-6, 1 - 1e-6)
    print(f"\nBest candidate after the wide search: {best_name}")

    # --- calibration check (same 70/30 chronological validation split) -
    split_point = int(len(val) * 0.7)
    raw_fit, raw_eval = best_pred_full[:split_point], best_pred_full[split_point:]
    y_fit, y_eval = y_val.iloc[:split_point].values, y_val.iloc[split_point:].values

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_fit, y_fit)
    calibrated_eval = iso.predict(raw_eval)

    ll_before = log_loss(y_eval, np.clip(raw_eval, 1e-6, 1 - 1e-6))
    ll_after = log_loss(y_eval, np.clip(calibrated_eval, 1e-6, 1 - 1e-6))
    brier_before = brier_score_loss(y_eval, np.clip(raw_eval, 1e-6, 1 - 1e-6))
    brier_after = brier_score_loss(y_eval, np.clip(calibrated_eval, 1e-6, 1 - 1e-6))
    calibration_adopted = ll_after < ll_before
    print(f"Calibration check: before={ll_before:.4f}, after={ll_after:.4f} -> "
          f"{'adopted' if calibration_adopted else 'NOT adopted'}")

    def bucket_reliability(probs, outcomes, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
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
    fig.savefig(os.path.join(FIG_DIR, "phase8_tuning_calibration.png"), dpi=120)
    plt.close(fig)

    # --- save v2 models --------------------------------------------------
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest_v2.joblib"))
    if xgb_model is not None:
        joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost_v2.joblib"))
    joblib.dump(poisson_home, os.path.join(MODELS_DIR, "poisson_home_v2.joblib"))
    joblib.dump(poisson_away, os.path.join(MODELS_DIR, "poisson_away_v2.joblib"))
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression_v2.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler_v2.joblib"))
    joblib.dump(medians, os.path.join(MODELS_DIR, "feature_medians_v2.joblib"))
    if calibration_adopted:
        joblib.dump(iso, os.path.join(MODELS_DIR, "calibrator_isotonic_v2.joblib"))

    manifest = {
        "trained_on": sorted(TRAIN_SEASONS),
        "validated_on": sorted(VALIDATION_SEASONS),
        "test_seasons_not_touched": sorted(TEST_SEASONS),
        "random_seed": SEED,
        "random_forest_n_iter": RF_N_ITER,
        "random_forest_best_params": rf_params,
        "xgboost_n_iter": XGB_N_ITER if xgb_model is not None else None,
        "xgboost_best_params": xgb_params,
        "xgboost_available": xgb_model is not None,
        "poisson_best_alpha": float(poisson_alpha),
        "logistic_regression_best_params": lr_params,
        "best_candidate_after_wide_search": best_name,
        "calibration_adopted": bool(calibration_adopted),
        "calibration_log_loss_before": float(ll_before),
        "calibration_log_loss_after": float(ll_after),
        "comparison_table": comparison.to_dict(orient="records"),
    }
    with open(os.path.join(MODELS_DIR, "manifest_v2.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    # --- report ------------------------------------------------------------
    lines = []
    lines.append("# Phase 8 (extended) - wider hyperparameter search\n")
    lines.append(f"Supersedes the narrow grids in `reports/phase8_models.md`. Random Forest: "
                  f"{RF_N_ITER} random configs. XGBoost: {XGB_N_ITER} random configs. Poisson: "
                  f"25-point alpha grid. Logistic Regression: 13 C values x 2 penalties (26 configs). "
                  f"All scored on VALIDATION ({sorted(VALIDATION_SEASONS)}, {len(val)} matches) only - "
                  f"TEST ({sorted(TEST_SEASONS)}) was not loaded by this script.\n")
    lines.append("## Comparison (sorted by Log Loss, lower is better)\n")
    lines.append(md_table(comparison) + "\n")
    lines.append(f"- Random Forest best: `{rf_params}`\n")
    if xgb_model is not None:
        lines.append(f"- XGBoost best: `{xgb_params}`\n")
    lines.append(f"- Poisson Regression best alpha: `{poisson_alpha:.4f}`\n")
    lines.append(f"- Logistic Regression best: `{lr_params}`\n")
    lines.append(f"\n**Best candidate after the wide search: {best_name}**\n")
    lines.append("\n## Calibration check\n")
    lines.append(f"Same held-out-30%-of-VALIDATION method as the narrow search: "
                  f"log loss before={ll_before:.4f}, after={ll_after:.4f} "
                  f"({'adopted' if calibration_adopted else 'not adopted'}). "
                  f"Checked on {len(y_eval)} matches - still a small sample, same caveat as before: "
                  f"treat as a weak signal, confirm properly on Phase 9's test set.\n")
    lines.append("\n![Calibration](figures/phase8_tuning_calibration.png)\n")
    lines.append("\n## Saved models\n")
    lines.append("`models/*_v2.joblib` + `models/manifest_v2.json` (v1 from the narrow search "
                  "is left in place for comparison, not deleted).\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {REPORT_MD}, {COMPARISON_CSV}, models/*_v2.joblib + manifest_v2.json")


if __name__ == "__main__":
    main()

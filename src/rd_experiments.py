"""
R&D round (post-Phase 9) - trying to improve on Random Forest's Phase 9
result using walk-forward evaluation across ALL historical seasons, not
a single validation season. This is the methodology fix for what the
Phase 9 walk-forward chart exposed: tuning against one season (2023/24)
let the model partly fit that season's quirks rather than learn
something that generalizes.

This script is free to use all of 2017/18-2025/26 (it's historical
"sandbox" data now, already seen once in Phase 9) - but nothing here is
a claim of a proven live edge. Any candidate that looks best here still
has to prove itself against 2026/27 as it's actually played before any
real money follows it (see reports/rd_round1.md's closing note).

Hyperparameters are kept FIXED at Phase 8's values throughout - this
round is testing IDEAS (feature pruning, ensembling), not re-running a
hyperparameter search, so a win here is attributable to the idea, not
to searching harder.

Candidates:
  1. Random Forest (Phase 8 params, full feature set) - the reference
  2. Random Forest, redundant features pruned (|corr| > 0.9 pairs - keep
     the one more correlated with the target, drop the other)
  3. Random Forest, weak features also pruned (|corr with target| < 0.03)
  4. XGBoost (Phase 8 params, full feature set)
  5. Poisson Regression (Phase 8 alpha, full feature set)
  6. Ensemble: simple average of candidates 1, 4, 5's probabilities

Evaluated via walk-forward: train on all seasons up to N, predict N+1,
for N+1 in {2020/21, ..., 2025/26} (6 folds) - same folds as Phase 9,
so this is directly comparable to that report.

Run from inside src/ (a few minutes):
    python rd_experiments.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402

FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
REPORT_MD = os.path.join("..", "reports", "rd_round1.md")
COMPARISON_CSV = os.path.join("..", "reports", "rd_round1_comparison.csv")
FIG_DIR = os.path.join("..", "reports", "figures")

SEED = 424410
ALL_SEASONS_CHRONOLOGICAL = [
    "2017/18", "2018/19", "2019/20", "2020/21", "2021/22",
    "2022/23", "2023/24", "2024/25", "2025/26",
]
WALK_FORWARD_MIN_TRAIN_SEASONS = 3

RF_PARAMS = {
    "max_depth": 15, "max_features": "sqrt",
    "min_samples_leaf": 36, "min_samples_split": 22, "n_estimators": 200,
}
XGB_PARAMS = {
    "colsample_bytree": 0.628020, "learning_rate": 0.038649, "max_depth": 2,
    "min_child_weight": 2, "n_estimators": 300, "reg_lambda": 2.507368,
    "subsample": 0.654868,
}
POISSON_ALPHA = 1.3335

NON_FEATURE_COLS = {
    "match_id", "match_date", "season", "league_name", "home_team", "away_team",
    "over_2_5", "behind_closed_doors",
    "avg_over_2_5", "avg_under_2_5", "bfe_over_2_5", "bfe_under_2_5",
    "bfe_close_over_2_5", "bfe_close_under_2_5",
    "league_over25_avg_to_date", "league_goals_avg_to_date",
}

REDUNDANCY_THRESHOLD = 0.9
WEAK_THRESHOLD = 0.03


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


def pick_pruned_feature_sets(df, feature_cols):
    """Computed fresh over the full historical dataset (not reusing Phase
    6's numbers, which were only over TRAIN+VALIDATION at the time)."""
    numeric = df[feature_cols].select_dtypes(include=[np.number])
    numeric = numeric.loc[:, numeric.std() > 0]
    target_corr = df[numeric.columns.tolist() + ["over_2_5"]].corr()["over_2_5"].drop("over_2_5")

    corr_matrix = numeric.corr()
    cols = corr_matrix.columns
    to_drop_redundant = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a in to_drop_redundant or b in to_drop_redundant:
                continue
            v = corr_matrix.loc[a, b]
            if pd.notna(v) and abs(v) > REDUNDANCY_THRESHOLD:
                # drop whichever of the pair is less correlated with the target
                drop = a if abs(target_corr.get(a, 0)) < abs(target_corr.get(b, 0)) else b
                to_drop_redundant.add(drop)

    redundant_pruned = [c for c in feature_cols if c not in to_drop_redundant]

    weak_cols = set(target_corr[target_corr.abs() < WEAK_THRESHOLD].index)
    weak_and_redundant_pruned = [c for c in redundant_pruned if c not in weak_cols]

    print(f"Feature pruning: {len(feature_cols)} full -> {len(redundant_pruned)} "
          f"redundancy-pruned -> {len(weak_and_redundant_pruned)} redundancy+weak-pruned")
    return redundant_pruned, weak_and_redundant_pruned


def fit_rf(train, feature_cols, medians):
    X = train[feature_cols].fillna(medians)
    m = RandomForestClassifier(random_state=SEED, n_jobs=-1, **RF_PARAMS)
    m.fit(X, train["over_2_5"])
    return m


def fit_xgb(train, feature_cols, medians):
    if XGBClassifier is None:
        return None
    X = train[feature_cols].fillna(medians)
    m = XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1, **XGB_PARAMS)
    m.fit(X, train["over_2_5"])
    return m


def fit_poisson(train, feature_cols, medians, scaler, goals_df):
    X = train[feature_cols].fillna(medians)
    X_scaled = scaler.fit_transform(X)
    tg = train[["match_id"]].merge(goals_df, on="match_id", how="left")
    m_home = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000)
    m_away = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000)
    m_home.fit(X_scaled, tg["home_goals"])
    m_away.fit(X_scaled, tg["away_goals"])
    return m_home, m_away


def predict_poisson(m_home, m_away, test, feature_cols, medians, scaler):
    X = test[feature_cols].fillna(medians)
    X_scaled = scaler.transform(X)
    lam = m_home.predict(X_scaled) + m_away.predict(X_scaled)
    return 1 - poisson_dist.cdf(2, lam)


def run_walk_forward(df, feature_cols, goals_df, candidate_name, build_fn, predict_fn):
    rows = []
    for i in range(WALK_FORWARD_MIN_TRAIN_SEASONS, len(ALL_SEASONS_CHRONOLOGICAL)):
        train_seasons = ALL_SEASONS_CHRONOLOGICAL[:i]
        test_season = ALL_SEASONS_CHRONOLOGICAL[i]
        train = df[df["season"].isin(train_seasons)]
        test = df[df["season"] == test_season]
        if len(test) == 0:
            continue
        medians = train[feature_cols].median(numeric_only=True)
        state = build_fn(train, feature_cols, medians)
        pred = predict_fn(state, test, feature_cols, medians)
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        ll = log_loss(test["over_2_5"], p)
        brier = brier_score_loss(test["over_2_5"], p)
        rows.append({"candidate": candidate_name, "predicts": test_season, "log_loss": ll, "brier_score": brier, "n": len(test)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])
    goals_df = pd.read_csv(CLEAN_CSV, usecols=["match_id", "home_goals", "away_goals"])
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    redundant_pruned, weak_and_redundant_pruned = pick_pruned_feature_sets(df, feature_cols)

    all_folds = []

    # 1. Random Forest, full features
    all_folds.append(run_walk_forward(
        df, feature_cols, goals_df, "RF - full features",
        lambda tr, fc, md: fit_rf(tr, fc, md),
        lambda model, te, fc, md: model.predict_proba(te[fc].fillna(md))[:, 1],
    ))

    # 2. Random Forest, redundancy-pruned features
    all_folds.append(run_walk_forward(
        df, redundant_pruned, goals_df, "RF - redundancy-pruned",
        lambda tr, fc, md: fit_rf(tr, fc, md),
        lambda model, te, fc, md: model.predict_proba(te[fc].fillna(md))[:, 1],
    ))

    # 3. Random Forest, redundancy+weak-pruned features
    all_folds.append(run_walk_forward(
        df, weak_and_redundant_pruned, goals_df, "RF - redundancy+weak-pruned",
        lambda tr, fc, md: fit_rf(tr, fc, md),
        lambda model, te, fc, md: model.predict_proba(te[fc].fillna(md))[:, 1],
    ))

    # 4. XGBoost, full features
    if XGBClassifier is not None:
        all_folds.append(run_walk_forward(
            df, feature_cols, goals_df, "XGBoost - full features",
            lambda tr, fc, md: fit_xgb(tr, fc, md),
            lambda model, te, fc, md: model.predict_proba(te[fc].fillna(md))[:, 1],
        ))

    # 5. Poisson, full features
    scaler_holder = {}

    def build_poisson(tr, fc, md):
        scaler = StandardScaler()
        scaler_holder["s"] = scaler
        return fit_poisson(tr, fc, md, scaler, goals_df)

    def predict_poisson_fn(state, te, fc, md):
        m_home, m_away = state
        return predict_poisson(m_home, m_away, te, fc, md, scaler_holder["s"])

    all_folds.append(run_walk_forward(
        df, feature_cols, goals_df, "Poisson Regression - full features",
        build_poisson, predict_poisson_fn,
    ))

    # 6. Ensemble: average RF + XGBoost + Poisson (full features), per fold
    ensemble_rows = []
    for i in range(WALK_FORWARD_MIN_TRAIN_SEASONS, len(ALL_SEASONS_CHRONOLOGICAL)):
        train_seasons = ALL_SEASONS_CHRONOLOGICAL[:i]
        test_season = ALL_SEASONS_CHRONOLOGICAL[i]
        train = df[df["season"].isin(train_seasons)]
        test = df[df["season"] == test_season]
        if len(test) == 0:
            continue
        medians = train[feature_cols].median(numeric_only=True)

        rf = fit_rf(train, feature_cols, medians)
        pred_rf = rf.predict_proba(test[feature_cols].fillna(medians))[:, 1]

        preds_to_average = [pred_rf]
        if XGBClassifier is not None:
            xgb = fit_xgb(train, feature_cols, medians)
            pred_xgb = xgb.predict_proba(test[feature_cols].fillna(medians))[:, 1]
            preds_to_average.append(pred_xgb)

        scaler = StandardScaler()
        m_home, m_away = fit_poisson(train, feature_cols, medians, scaler, goals_df)
        pred_poisson = predict_poisson(m_home, m_away, test, feature_cols, medians, scaler)
        preds_to_average.append(pred_poisson)

        ensemble_pred = np.mean(preds_to_average, axis=0)
        p = np.clip(ensemble_pred, 1e-6, 1 - 1e-6)
        ll = log_loss(test["over_2_5"], p)
        brier = brier_score_loss(test["over_2_5"], p)
        ensemble_rows.append({"candidate": "Ensemble (RF+XGB+Poisson avg)", "predicts": test_season,
                               "log_loss": ll, "brier_score": brier, "n": len(test)})
    all_folds.append(pd.DataFrame(ensemble_rows))

    # --- market, same folds, for reference --------------------------------
    market_rows = []
    for season in ALL_SEASONS_CHRONOLOGICAL[WALK_FORWARD_MIN_TRAIN_SEASONS:]:
        test = df[df["season"] == season]
        if season in ("2024/25", "2025/26"):
            m = test.dropna(subset=["bfe_close_over_2_5", "bfe_close_under_2_5"])
            io, iu = 1 / m["bfe_close_over_2_5"], 1 / m["bfe_close_under_2_5"]
        else:
            m = test.dropna(subset=["avg_over_2_5", "avg_under_2_5"])
            io, iu = 1 / m["avg_over_2_5"], 1 / m["avg_under_2_5"]
        if len(m) == 0:
            continue
        pred = io / (io + iu)
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        ll = log_loss(m["over_2_5"], p)
        brier = brier_score_loss(m["over_2_5"], p)
        market_rows.append({"candidate": "Market (reference)", "predicts": season, "log_loss": ll, "brier_score": brier, "n": len(m)})
    all_folds.append(pd.DataFrame(market_rows))

    detail = pd.concat(all_folds, ignore_index=True)
    detail.to_csv(COMPARISON_CSV, index=False)

    summary = (
        detail.groupby("candidate")
        .agg(mean_log_loss=("log_loss", "mean"), std_log_loss=("log_loss", "std"),
             mean_brier=("brier_score", "mean"), n_folds=("log_loss", "count"))
        .reset_index()
        .sort_values("mean_log_loss")
    )
    print("\n" + summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in detail["candidate"].unique():
        sub = detail[detail["candidate"] == name].sort_values("predicts")
        style = "--" if name == "Market (reference)" else "-"
        ax.plot(sub["predicts"], sub["log_loss"], marker="o", linestyle=style, label=name)
    ax.axhline(0.693, color="gray", linestyle=":", linewidth=1, label="coin-flip floor")
    ax.set_xlabel("Season predicted")
    ax.set_ylabel("Log Loss")
    ax.set_title("R&D round 1 - walk-forward comparison across all candidates")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rd_round1_walk_forward.png"), dpi=120)
    plt.close(fig)

    # --- report --------------------------------------------------------------
    lines = []
    lines.append("# R&D round 1\n")
    lines.append("Judged by walk-forward log loss averaged across 6 folds (predicting "
                  "2020/21 through 2025/26 in turn, training on everything before each one) - "
                  "not a single validation season, specifically to avoid repeating the mistake "
                  "Phase 9's walk-forward chart exposed. Hyperparameters fixed at Phase 8's "
                  "values throughout, so any difference here is attributable to the idea being "
                  "tested (feature pruning, ensembling), not to searching harder.\n")
    lines.append(f"Feature pruning: {len(feature_cols)} engineered features -> "
                  f"{len(redundant_pruned)} after dropping redundant pairs (|corr| > "
                  f"{REDUNDANCY_THRESHOLD}, recomputed fresh over all historical data) -> "
                  f"{len(weak_and_redundant_pruned)} after also dropping weak features "
                  f"(|corr with target| < {WEAK_THRESHOLD}).\n")
    lines.append("## Summary (sorted by mean Log Loss across folds, lower is better)\n")
    lines.append(md_table(summary) + "\n")
    lines.append("\n## Fold-by-fold detail\n")
    pivot = detail.pivot_table(index="predicts", columns="candidate", values="log_loss")
    pivot = pivot.reset_index()
    lines.append(md_table(pivot) + "\n")
    lines.append("\n![Walk-forward comparison](figures/rd_round1_walk_forward.png)\n")

    models_only = summary[summary["candidate"] != "Market (reference)"].reset_index(drop=True)
    best_candidate = models_only.iloc[0]["candidate"]
    best_ll = models_only.iloc[0]["mean_log_loss"]
    market_ll = summary[summary["candidate"] == "Market (reference)"]["mean_log_loss"].values[0]
    beats_market = best_ll < market_ll
    lines.append(f"\n**Best model candidate: {best_candidate}** (mean log loss {best_ll:.4f} "
                  f"vs market's {market_ll:.4f} across the same folds - "
                  f"{'beats' if beats_market else 'does NOT beat'} the market).\n")
    lines.append("\n**This is still not proof of a live edge.** Every number above comes from "
                  "historical data already used in Phase 9 - it tells us which approach "
                  "generalizes best across past seasons, which is a much better signal than a "
                  "single validation season, but it is still not out-of-sample in the way that "
                  "matters for staking real money. Whatever candidate is chosen from this round "
                  "needs to be frozen, unchanged, and proven against 2026/27 as it's actually "
                  "played before any bet is placed on it.\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {REPORT_MD}, {COMPARISON_CSV}, figures/rd_round1_walk_forward.png")


if __name__ == "__main__":
    main()

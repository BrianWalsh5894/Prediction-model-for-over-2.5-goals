"""
R&D round 2 - pushing specifically on Poisson Regression, since round 1
showed it's the most consistent individual model across 6 walk-forward
seasons (narrowly ahead of Random Forest, which Phase 8 spent much more
tuning effort on).

Same evaluation method as round 1: walk-forward across 6 folds
(predicting 2020/21 through 2025/26 in turn, training on everything
before each one) - never a single validation season.

Two ideas tested:

  A) Wider alpha search. Phase 8 only tried a 25-point grid scored on
     ONE validation season (2023/24). Here a much wider grid is scored
     on the 6-fold walk-forward MEAN, the same selection method used in
     round 1 to compare whole models - so this is consistent with that
     upgrade, not a step back to single-season tuning.

  B) Dixon-Coles-style attack/defense ratings as extra features, on top
     of the existing feature set. Each walk-forward fold fits attack,
     defense and home-advantage parameters by maximum likelihood on
     that fold's TRAIN matches only (teams unseen in training get
     average attack/defense as a fallback), then those ratings are held
     FROZEN for every match in that fold's test season - no match ever
     uses a rating derived from itself or from the future. This is a
     simplification of real Dixon-Coles (no time-decay weighting of
     older matches) - noted so it isn't mistaken for the full method.

Run from inside src/ (a few minutes - the Dixon-Coles fit is the slow
part, one optimization per fold):
    python rd_experiments2.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist
from scipy.optimize import minimize
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402

FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
REPORT_MD = os.path.join("..", "reports", "rd_round2.md")
COMPARISON_CSV = os.path.join("..", "reports", "rd_round2_comparison.csv")
ALPHA_CSV = os.path.join("..", "reports", "rd_round2_alpha_search.csv")
FIG_DIR = os.path.join("..", "reports", "figures")

ALL_SEASONS_CHRONOLOGICAL = [
    "2017/18", "2018/19", "2019/20", "2020/21", "2021/22",
    "2022/23", "2023/24", "2024/25", "2025/26",
]
WALK_FORWARD_MIN_TRAIN_SEASONS = 3
CURRENT_ALPHA = 1.3335  # Phase 8's pick, for comparison

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


# ---------------------------------------------------------------------------
# Dixon-Coles-style attack/defense ratings (simplified: no time decay)
# ---------------------------------------------------------------------------

def fit_dixon_coles(train, goals_df):
    """Fits one attack and one defense rating per team, plus a single home
    advantage term, by maximum likelihood on independent-Poisson goal
    counts. log(lambda_home) = attack[home] + defense[away] + home_adv
    log(lambda_away) = attack[away] + defense[home]
    Identifiability constraint: mean(attack) = 0."""
    g = train[["match_id", "home_team", "away_team"]].merge(goals_df, on="match_id", how="left")
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = g["home_team"].map(idx).values
    away_idx = g["away_team"].map(idx).values
    home_goals = g["home_goals"].values
    away_goals = g["away_goals"].values

    def unpack(params):
        attack = params[:n]
        defense = params[n:2 * n]
        home_adv = params[-1]
        return attack, defense, home_adv

    def neg_log_lik(params):
        attack, defense, home_adv = unpack(params)
        lam_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
        lam_away = np.exp(attack[away_idx] + defense[home_idx])
        lam_home = np.clip(lam_home, 1e-6, 50)
        lam_away = np.clip(lam_away, 1e-6, 50)
        ll = (
            home_goals * np.log(lam_home) - lam_home
            + away_goals * np.log(lam_away) - lam_away
        )
        # soft identifiability penalty instead of a hard constraint
        penalty = 100 * (attack.mean() ** 2)
        return -ll.sum() + penalty

    x0 = np.zeros(2 * n + 1)
    result = minimize(neg_log_lik, x0, method="L-BFGS-B", options={"maxiter": 300})
    attack, defense, home_adv = unpack(result.x)
    ratings = {t: (attack[idx[t]], defense[idx[t]]) for t in teams}
    return ratings, home_adv, float(attack.mean()), float(defense.mean())


def add_dc_features(df_slice, ratings, home_adv, avg_attack, avg_defense):
    out = df_slice.copy()

    def get_rating(team):
        return ratings.get(team, (avg_attack, avg_defense))

    home_r = out["home_team"].map(get_rating)
    away_r = out["away_team"].map(get_rating)
    out["dc_home_attack"] = home_r.map(lambda r: r[0])
    out["dc_home_defense"] = home_r.map(lambda r: r[1])
    out["dc_away_attack"] = away_r.map(lambda r: r[0])
    out["dc_away_defense"] = away_r.map(lambda r: r[1])
    out["dc_home_adv"] = home_adv
    return out


# ---------------------------------------------------------------------------
# Poisson fit/predict helpers
# ---------------------------------------------------------------------------

def fit_poisson(train, feature_cols, medians, alpha, goals_df):
    X = train[feature_cols].fillna(medians)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tg = train[["match_id"]].merge(goals_df, on="match_id", how="left")
    m_home = PoissonRegressor(alpha=alpha, max_iter=1000)
    m_away = PoissonRegressor(alpha=alpha, max_iter=1000)
    m_home.fit(X_scaled, tg["home_goals"])
    m_away.fit(X_scaled, tg["away_goals"])
    return m_home, m_away, scaler


def predict_poisson(state, test, feature_cols, medians):
    m_home, m_away, scaler = state
    X = test[feature_cols].fillna(medians)
    X_scaled = scaler.transform(X)
    lam = m_home.predict(X_scaled) + m_away.predict(X_scaled)
    return 1 - poisson_dist.cdf(2, lam)


def walk_forward_eval(df, feature_cols, goals_df, alpha, use_dc=False):
    rows = []
    for i in range(WALK_FORWARD_MIN_TRAIN_SEASONS, len(ALL_SEASONS_CHRONOLOGICAL)):
        train_seasons = ALL_SEASONS_CHRONOLOGICAL[:i]
        test_season = ALL_SEASONS_CHRONOLOGICAL[i]
        train = df[df["season"].isin(train_seasons)]
        test = df[df["season"] == test_season]
        if len(test) == 0:
            continue

        cols = list(feature_cols)
        train_use, test_use = train, test
        if use_dc:
            ratings, home_adv, avg_a, avg_d = fit_dixon_coles(train, goals_df)
            train_use = add_dc_features(train, ratings, home_adv, avg_a, avg_d)
            test_use = add_dc_features(test, ratings, home_adv, avg_a, avg_d)
            cols = cols + ["dc_home_attack", "dc_home_defense", "dc_away_attack", "dc_away_defense", "dc_home_adv"]

        medians = train_use[cols].median(numeric_only=True)
        state = fit_poisson(train_use, cols, medians, alpha, goals_df)
        pred = predict_poisson(state, test_use, cols, medians)
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        ll = log_loss(test["over_2_5"], p)
        brier = brier_score_loss(test["over_2_5"], p)
        rows.append({"predicts": test_season, "log_loss": ll, "brier_score": brier, "n": len(test)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])
    goals_df = pd.read_csv(CLEAN_CSV, usecols=["match_id", "home_goals", "away_goals"])
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    # --- A: wide alpha search (current feature set, no DC features) --------
    alpha_grid = sorted(set(
        list(np.logspace(-3, 2, 30)) + [CURRENT_ALPHA]
    ))
    alpha_rows = []
    print(f"Searching {len(alpha_grid)} alpha values via 6-fold walk-forward...")
    for a in alpha_grid:
        res = walk_forward_eval(df, feature_cols, goals_df, a, use_dc=False)
        alpha_rows.append({"alpha": a, "mean_log_loss": res["log_loss"].mean(), "std_log_loss": res["log_loss"].std()})
    alpha_df = pd.DataFrame(alpha_rows).sort_values("mean_log_loss")
    alpha_df.to_csv(ALPHA_CSV, index=False)
    best_alpha = float(alpha_df.iloc[0]["alpha"])
    print(alpha_df.head(10).to_string(index=False))
    print(f"\nBest alpha: {best_alpha} (current Phase 8 alpha: {CURRENT_ALPHA})")

    # --- B: Dixon-Coles features, at both the current and the new best alpha
    print("\nFitting Dixon-Coles attack/defense ratings per fold (this is the slow part)...")
    candidates = {
        f"Poisson - current alpha ({CURRENT_ALPHA})": (CURRENT_ALPHA, False),
        f"Poisson - best alpha ({best_alpha:.4f})": (best_alpha, False),
        f"Poisson - current alpha + Dixon-Coles features": (CURRENT_ALPHA, True),
        f"Poisson - best alpha + Dixon-Coles features": (best_alpha, True),
    }

    all_rows = []
    for name, (a, use_dc) in candidates.items():
        res = walk_forward_eval(df, feature_cols, goals_df, a, use_dc=use_dc)
        res["candidate"] = name
        all_rows.append(res)
        print(f"  {name}: mean log loss = {res['log_loss'].mean():.6f}")

    # market reference, same folds
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
        market_rows.append({
            "candidate": "Market (reference)", "predicts": season,
            "log_loss": log_loss(m["over_2_5"], p),
            "brier_score": brier_score_loss(m["over_2_5"], p), "n": len(m),
        })
    all_rows.append(pd.DataFrame(market_rows))

    detail = pd.concat(all_rows, ignore_index=True)
    detail.to_csv(COMPARISON_CSV, index=False)

    summary = (
        detail.groupby("candidate")
        .agg(mean_log_loss=("log_loss", "mean"), std_log_loss=("log_loss", "std"),
             mean_brier=("brier_score", "mean"), n_folds=("log_loss", "count"))
        .reset_index()
        .sort_values("mean_log_loss")
    )
    print("\n" + summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(alpha_df["alpha"], alpha_df["mean_log_loss"], marker="o", markersize=3)
    axes[0].axvline(CURRENT_ALPHA, color="gray", linestyle="--", label=f"Phase 8 alpha ({CURRENT_ALPHA})")
    axes[0].axvline(best_alpha, color="green", linestyle="--", label=f"best alpha ({best_alpha:.3f})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("alpha (log scale)")
    axes[0].set_ylabel("Mean walk-forward Log Loss")
    axes[0].set_title("Poisson alpha search (6-fold walk-forward)")
    axes[0].legend(fontsize=8)

    for name in detail["candidate"].unique():
        sub = detail[detail["candidate"] == name].sort_values("predicts")
        style = "--" if name == "Market (reference)" else "-"
        axes[1].plot(sub["predicts"], sub["log_loss"], marker="o", linestyle=style, label=name, markersize=4)
    axes[1].axhline(0.693, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Season predicted")
    axes[1].set_ylabel("Log Loss")
    axes[1].set_title("Poisson variants vs market")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rd_round2_poisson.png"), dpi=120)
    plt.close(fig)

    lines = []
    lines.append("# R&D round 2 - pushing on Poisson Regression\n")
    lines.append("Same evaluation method as round 1: mean Log Loss across 6 walk-forward "
                  "folds (predicting 2020/21 through 2025/26 in turn). Two ideas tested on "
                  "top of the existing Poisson model: a much wider alpha search (scored on "
                  "the walk-forward mean, not one validation season), and Dixon-Coles-style "
                  "attack/defense ratings as extra features (fit per fold on that fold's "
                  "training matches only, then held frozen for the whole test season - no "
                  "leakage, but a simplified Dixon-Coles with no time-decay weighting).\n")
    lines.append(f"## Alpha search ({len(alpha_grid)} values tried)\n")
    lines.append(md_table(alpha_df.head(10)) + "\n")
    lines.append(f"\nBest alpha: **{best_alpha:.4f}** vs Phase 8's **{CURRENT_ALPHA}**.\n")
    lines.append("\n## Comparison with Dixon-Coles features\n")
    lines.append(md_table(summary) + "\n")
    lines.append("\n![Poisson round 2](figures/rd_round2_poisson.png)\n")

    models_only = summary[summary["candidate"] != "Market (reference)"].reset_index(drop=True)
    best_name = models_only.iloc[0]["candidate"]
    best_ll = models_only.iloc[0]["mean_log_loss"]
    market_ll = summary[summary["candidate"] == "Market (reference)"]["mean_log_loss"].values[0]
    beats_market = best_ll < market_ll
    lines.append(f"\n**Best: {best_name}** (mean log loss {best_ll:.4f} vs market's "
                  f"{market_ll:.4f} - {'beats' if beats_market else 'does NOT beat'} the market).\n")
    lines.append("\nStill historical data, still not proof of a live edge - same caveat as "
                  "round 1. Whatever is picked from this needs to be frozen and proven "
                  "against 2026/27 as it's actually played before any money follows it.\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {REPORT_MD}, {COMPARISON_CSV}, {ALPHA_CSV}, figures/rd_round2_poisson.png")


if __name__ == "__main__":
    main()

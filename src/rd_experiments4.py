"""
R&D round 4 - market-anchored model.

Rounds 1-3 tried to beat the market from scratch using public stats
(goals, shots, Elo, Dixon-Coles) and all landed ~0.004 log loss behind it.
The market already prices those stats in, and also knows things we don't
(team news, line-ups, late money).

So this round flips the question. Instead of building a probability from
scratch, START from the market's early price and let a small model learn
only systematic corrections to it, e.g. "the early price under-rates
Overs when our Poisson / Dixon-Coles models disagree with it".

The anchor
----------
`avg_over_2_5` / `avg_under_2_5` = Football-Data's average bookmaker
price (the "Avg>2.5" columns). These are collected BEFORE closing
(Football-Data collects them ahead of the matchday), so they are an
"early" price, available back to 2019/20. Margin removed by
normalising the two implied probabilities.

Fairness rules (no leakage)
---------------------------
  - Poisson predictions used as an input are produced out-of-sample: for
    each season, Poisson is trained on ALL earlier seasons only (expanding
    window), so training rows never see a Poisson fitted on themselves.
  - Dixon-Coles probabilities come from round 3's rolling weekly fit
    (xi = 0, 3-year window) - out-of-sample by construction.
  - Walk-forward: train the correction model on seasons with odds
    (2019/20 onward) up to N-1, predict N. 5 folds: 2021/22 - 2025/26.

What counts as success
----------------------
  1. Beats its own anchor (the early bookmaker average) across the folds -
     i.e. the corrections are real, not noise.
  2. The honest test: beats the Betfair Exchange CLOSING price on
     2024/25 + 2025/26 (the only seasons with Betfair prices), with a
     bootstrap confidence interval on the difference.

Run from inside src/ (about 2 minutes):
    python rd_experiments4.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from rd_experiments3 import rolling_dc  # noqa: E402  (round 3's weekly Dixon-Coles)

FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
REPORT_MD = os.path.join("..", "reports", "rd_round4.md")
COMPARISON_CSV = os.path.join("..", "reports", "rd_round4_comparison.csv")
FIG_DIR = os.path.join("..", "reports", "figures")

ALL_SEASONS = [
    "2017/18", "2018/19", "2019/20", "2020/21", "2021/22",
    "2022/23", "2023/24", "2024/25", "2025/26",
]
ODDS_SEASONS = ALL_SEASONS[2:]          # 2019/20 onward have the anchor price
FOLD_SEASONS = ALL_SEASONS[4:]          # 2021/22 - 2025/26 (>= 2 seasons of odds to train on)
BETFAIR_SEASONS = ["2024/25", "2025/26"]
POISSON_ALPHA = 1.3335
DC_XI = 0.0                             # round 3's pick
SEED = 424410
N_BOOT = 2000

NON_FEATURE_COLS = {
    "match_id", "match_date", "season", "league_name", "home_team", "away_team",
    "over_2_5", "behind_closed_doors",
    "avg_over_2_5", "avg_under_2_5", "bfe_over_2_5", "bfe_under_2_5",
    "bfe_close_over_2_5", "bfe_close_under_2_5",
    "league_over25_avg_to_date", "league_goals_avg_to_date",
}
LEAGUE_DUMMIES = ["league_Championship", "league_League One", "league_League Two"]


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


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def logit(p):
    p = clip(p)
    return np.log(p / (1 - p))


def ll(y, p):
    return log_loss(y, clip(p), labels=[0, 1])


def demargin(over_odds, under_odds):
    io, iu = 1 / over_odds, 1 / under_odds
    return io / (io + iu)


def oos_poisson(df, base_cols, goals_df):
    """Out-of-sample Poisson P(Over 2.5) for every season with odds: each
    season predicted by a model trained on all EARLIER seasons only."""
    out = pd.Series(np.nan, index=df.index)
    for i, season in enumerate(ALL_SEASONS):
        if season not in ODDS_SEASONS:
            continue
        train = df[df["season"].isin(ALL_SEASONS[:i])]
        test = df[df["season"] == season]
        med = train[base_cols].median(numeric_only=True)
        sc = StandardScaler()
        Xtr = sc.fit_transform(train[base_cols].fillna(med))
        Xte = sc.transform(test[base_cols].fillna(med))
        tg = train[["match_id"]].merge(goals_df, on="match_id", how="left")
        mh = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000).fit(Xtr, tg["home_goals"])
        ma = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000).fit(Xtr, tg["away_goals"])
        lam = mh.predict(Xte) + ma.predict(Xte)
        out.loc[test.index] = 1 - poisson_dist.cdf(2, lam)
    return out


def fit_lr(train, test, cols):
    sc = StandardScaler()
    Xtr = sc.fit_transform(train[cols])
    Xte = sc.transform(test[cols])
    m = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, train["over_2_5"])
    coefs = dict(zip(cols, m.coef_[0] / sc.scale_))   # per-unit coefficients, easier to read
    return m.predict_proba(Xte)[:, 1], coefs


def bootstrap_diff(y, p_model, p_ref, rng):
    """Mean per-match log-loss difference (model - reference) with a 95%
    bootstrap interval. Negative = model better."""
    y = np.asarray(y)
    lm = -(y * np.log(clip(p_model)) + (1 - y) * np.log(1 - clip(p_model)))
    lr = -(y * np.log(clip(p_ref)) + (1 - y) * np.log(1 - clip(p_ref)))
    d = lm - lr
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    t0 = time.time()
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])
    df = df[df["season"].isin(ALL_SEASONS)].reset_index(drop=True).copy()
    clean = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    clean = clean[clean["season"].isin(ALL_SEASONS)]
    goals_df = clean[["match_id", "home_goals", "away_goals"]]
    base_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    print("Building inputs (all out-of-sample)...")
    df["p_anchor"] = demargin(df["avg_over_2_5"], df["avg_under_2_5"])
    df["p_bfe_early"] = demargin(df["bfe_over_2_5"], df["bfe_under_2_5"])
    df["p_bfe_close"] = demargin(df["bfe_close_over_2_5"], df["bfe_close_under_2_5"])
    df["p_poisson"] = oos_poisson(df, base_cols, goals_df)
    dc = rolling_dc(clean, DC_XI)[["match_id", "dc_p_over"]]
    df = df.merge(dc, on="match_id", how="left")
    df["p_dc"] = df["dc_p_over"].fillna(df["p_poisson"])
    df["p_blend_r3"] = 0.5 * df["p_poisson"] + 0.5 * df["p_dc"]

    df["x_anchor"] = logit(df["p_anchor"])
    df["x_pois_gap"] = logit(df["p_poisson"]) - df["x_anchor"]   # how far Poisson disagrees with the price
    df["x_dc_gap"] = logit(df["p_dc"]) - df["x_anchor"]
    df["early_season"] = (np.minimum(df["home_season_match_num"], df["away_season_match_num"]) <= 5).astype(float)

    d = df[df["season"].isin(ODDS_SEASONS) & df["p_anchor"].notna() & df["p_poisson"].notna()].copy()
    print(f"{len(d)} matches with an anchor price ({ODDS_SEASONS[0]} - {ODDS_SEASONS[-1]})")

    specs = {
        "A. Anchor recalibrated": ["x_anchor"],
        "B. Anchor + Poisson/DC gaps": ["x_anchor", "x_pois_gap", "x_dc_gap"],
        "C. B + league + early-season": ["x_anchor", "x_pois_gap", "x_dc_gap", "early_season"] + LEAGUE_DUMMIES,
    }

    rows, coef_rows, test_preds = [], [], []
    for season in FOLD_SEASONS:
        i = ODDS_SEASONS.index(season)
        train = d[d["season"].isin(ODDS_SEASONS[:i])]
        test = d[d["season"] == season].copy()
        y = test["over_2_5"].values

        preds = {
            "Anchor: early bookmaker average (raw)": test["p_anchor"].values,
            "Round 3 blend (no market)": test["p_blend_r3"].values,
        }
        for name, cols in specs.items():
            p, coefs = fit_lr(train, test, cols)
            preds[name] = p
            coef_rows.append({"fold": season, "candidate": name, **coefs})
            test[name] = p

        has_bfe = test["p_bfe_close"].notna().values
        for name, p in preds.items():
            rows.append({"candidate": name, "predicts": season, "log_loss": ll(y, p),
                         "brier_score": brier_score_loss(y, clip(p)), "n": len(y)})
        if season in BETFAIR_SEASONS:
            for name, col in [("Betfair EARLY price", "p_bfe_early"), ("Betfair CLOSING price", "p_bfe_close")]:
                m = test[col].notna().values
                rows.append({"candidate": name, "predicts": season,
                             "log_loss": ll(y[m], test[col].values[m]),
                             "brier_score": brier_score_loss(y[m], clip(test[col].values[m])),
                             "n": int(m.sum())})
            test_preds.append(test)
        print(f"  {season}: " + ", ".join(f"{r['candidate'][:24]}={r['log_loss']:.4f}"
                                          for r in rows if r["predicts"] == season))

    detail = pd.DataFrame(rows)
    detail.to_csv(COMPARISON_CSV, index=False)

    anchor_ll = detail[detail["candidate"].str.startswith("Anchor:")].set_index("predicts")["log_loss"]
    summary = (
        detail[~detail["candidate"].str.startswith("Betfair")]
        .groupby("candidate")
        .agg(mean_log_loss=("log_loss", "mean"), std_log_loss=("log_loss", "std"),
             mean_brier=("brier_score", "mean"), n_folds=("log_loss", "count"))
        .reset_index().sort_values("mean_log_loss")
    )
    beats_anchor = (
        detail[~detail["candidate"].str.startswith(("Betfair", "Anchor:"))]
        .assign(b=lambda x: x["log_loss"] < x["predicts"].map(anchor_ll))
        .groupby("candidate")["b"].sum()
    )
    summary["folds_beating_anchor"] = summary["candidate"].map(beats_anchor)
    pd.set_option("display.width", 200)
    print("\n5-fold walk-forward (2021/22 - 2025/26):")
    print(summary.to_string(index=False))

    # --- The honest test: vs Betfair closing, 2024/25 + 2025/26 pooled ------
    bf = pd.concat(test_preds)
    bf = bf[bf["p_bfe_close"].notna()]
    yb = bf["over_2_5"].values
    bf_rows = []
    for name in ["Anchor: early bookmaker average (raw)", "Round 3 blend (no market)",
                 *specs.keys(), "Betfair EARLY price"]:
        col = {"Anchor: early bookmaker average (raw)": "p_anchor",
               "Round 3 blend (no market)": "p_blend_r3",
               "Betfair EARLY price": "p_bfe_early"}.get(name, name)
        m = bf[col].notna().values
        diff, lo, hi = bootstrap_diff(yb[m], bf[col].values[m], bf["p_bfe_close"].values[m], rng)
        bf_rows.append({"candidate": name, "log_loss": ll(yb[m], bf[col].values[m]),
                        "betfair_close_log_loss": ll(yb[m], bf["p_bfe_close"].values[m]),
                        "diff_vs_close": diff, "ci_low": lo, "ci_high": hi, "n": int(m.sum())})
    bf_df = pd.DataFrame(bf_rows).sort_values("log_loss")
    print("\nvs Betfair CLOSING price, 2024/25 + 2025/26 pooled (diff < 0 = better than close):")
    print(bf_df.to_string(index=False))

    coef_df = pd.DataFrame(coef_rows)
    print("\nCorrection-model coefficients for model C (per unit, by fold):")
    print(coef_df[coef_df["candidate"].str.startswith("C.")].drop(columns="candidate").round(3).to_string(index=False))

    # --- Figure -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in detail["candidate"].unique():
        sub = detail[detail["candidate"] == name]
        style = ":" if name.startswith("Betfair") else ("--" if name.startswith("Anchor") else "-")
        ax.plot(sub["predicts"], sub["log_loss"], marker="o", linestyle=style, label=name, markersize=4)
    ax.set_xlabel("Season predicted")
    ax.set_ylabel("Log Loss")
    ax.set_title("Round 4 - market-anchored models")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rd_round4_market_anchored.png"), dpi=120)
    plt.close(fig)

    # --- Report ---------------------------------------------------------------
    best_bf = bf_df[~bf_df["candidate"].str.startswith("Betfair")].iloc[0]
    significant = best_bf["ci_high"] < 0
    lines = [
        "# R&D round 4 - market-anchored model\n",
        "Start from the early bookmaker-average price (Football-Data `Avg>2.5`, margin removed) "
        "and let a logistic regression learn corrections from out-of-sample Poisson and rolling "
        "Dixon-Coles predictions (as gaps from the price), league and early-season. Walk-forward "
        "over 5 folds (2021/22 - 2025/26), training only on seasons with odds (2019/20+).\n",
        "## Walk-forward (5 folds)\n",
        md_table(summary) + "\n",
        "\n## The honest test - vs Betfair CLOSING price (2024/25 + 2025/26 pooled)\n",
        "`diff_vs_close` = mean per-match log loss minus Betfair close's; negative = better. "
        "95% bootstrap interval in `ci_low` / `ci_high`.\n",
        md_table(bf_df) + "\n",
        "\n## Correction coefficients (model C, per fold)\n",
        md_table(coef_df[coef_df["candidate"].str.startswith("C.")].drop(columns="candidate")) + "\n",
        "\n![Round 4](figures/rd_round4_market_anchored.png)\n",
        f"\n**Best vs Betfair close: {best_bf['candidate']}** - diff {best_bf['diff_vs_close']:+.4f} "
        f"(95% CI {best_bf['ci_low']:+.4f} to {best_bf['ci_high']:+.4f}). "
        + ("The whole interval is below zero." if significant else
           "The interval includes or sits above zero - no demonstrated edge over the closing price.") + "\n",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_MD}, {COMPARISON_CSV}, figures/rd_round4_market_anchored.png")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

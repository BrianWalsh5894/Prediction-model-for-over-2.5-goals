"""
v3-eu-upper - step 3: backtest the v3 recipe on La Liga, Serie A,
Bundesliga and Ligue 1, and decide pooled vs per-country.

Same evaluation as the English R&D: 6-fold walk-forward (train on every
season before, predict 2020/21 ... 2025/26), log loss, compared with the
bookmaker average (all 6 folds) and the Betfair closing price
(2024/25 + 2025/26), with 95% bootstrap intervals.

Candidates (Poisson alpha 1.3335 and Dixon-Coles settings exactly as v3):
  A. POOLED   - one Poisson on all four leagues (league indicators included)
                + Dixon-Coles fitted separately per league, 50/50 blend
  B. PER-COUNTRY - one Poisson per league + Dixon-Coles per league, 50/50
  C. Pooled Poisson alone
  D. Dixon-Coles alone (per league)
Dixon-Coles is always per league: teams from different countries never
play each other in the league, so a joint rating would mean nothing.

Run from inside src/ (a couple of minutes):
    python rd_eu_upper.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from rd_experiments3 import rolling_dc, poisson_fit_predict, NON_FEATURE_COLS  # noqa: E402

FEATURES_CSV = os.path.join("..", "data", "clean", "features_eu_upper.csv")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean_eu_upper.csv")
REPORT_MD = os.path.join("..", "reports", "rd_eu_upper.md")
DETAIL_CSV = os.path.join("..", "reports", "rd_eu_upper_detail.csv")
SUMMARY_CSV = os.path.join("..", "reports", "rd_eu_upper_summary.csv")
FIG_DIR = os.path.join("..", "reports", "figures")

LEAGUES = ["La Liga", "Serie A", "Bundesliga", "Ligue 1"]
ALL_SEASONS = ["2017/18", "2018/19", "2019/20", "2020/21", "2021/22",
               "2022/23", "2023/24", "2024/25", "2025/26"]
FOLD_START = 3
BETFAIR_SEASONS = ["2024/25", "2025/26"]
SEED = 424410
N_BOOT = 2000

A, B, C, D = ("A. Pooled Poisson + DC (v3 recipe)", "B. Per-country Poisson + DC",
              "C. Pooled Poisson alone", "D. Dixon-Coles alone")
MODELS = [A, B, C, D]


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def pm_ll(y, p):
    p = clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def demargin(o, u):
    return (1 / o) / (1 / o + 1 / u)


def md_table(df, float_fmt="{:.4f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: float_fmt.format(v) if pd.notna(v) else "")
        else:
            df[c] = df[c].astype(str)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(r) + " |" for r in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep] + rows)


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(FIG_DIR, exist_ok=True)

    feats = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])
    feats = feats[feats["season"].isin(ALL_SEASONS)].reset_index(drop=True).copy()
    clean = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    clean = clean[clean["season"].isin(ALL_SEASONS)]
    goals_df = clean[["match_id", "home_goals", "away_goals"]]
    base_cols = [c for c in feats.columns if c not in NON_FEATURE_COLS]
    print(f"{len(feats)} matches, {len(base_cols)} features")
    print(feats.groupby(["season", "league_name"]).size().unstack().to_string())

    print("\nRolling Dixon-Coles per league (xi=0)...")
    dc = pd.concat([rolling_dc(clean[clean["league_name"] == lg], 0.0) for lg in LEAGUES])
    dfx = feats.merge(dc[["match_id", "dc_p_over"]], on="match_id", how="left")

    preds = []
    for i in range(FOLD_START, len(ALL_SEASONS)):
        season = ALL_SEASONS[i]
        train = dfx[dfx["season"].isin(ALL_SEASONS[:i])]
        test = dfx[dfx["season"] == season].copy()
        p_pool = pd.Series(poisson_fit_predict(train, test, base_cols, goals_df), index=test.index)
        p_country = pd.Series(np.nan, index=test.index)
        for lg in LEAGUES:
            tr, te = train[train["league_name"] == lg], test[test["league_name"] == lg]
            if len(te) and len(tr):
                p_country.loc[te.index] = poisson_fit_predict(tr, te, base_cols, goals_df)
        p_dc = test["dc_p_over"].fillna(p_pool)
        out = test[["match_id", "season", "league_name", "over_2_5"]].copy()
        out[A] = 0.5 * p_pool + 0.5 * p_dc
        out[B] = 0.5 * p_country + 0.5 * p_dc
        out[C] = p_pool
        out[D] = p_dc
        out["Bookmaker average"] = demargin(test["avg_over_2_5"], test["avg_under_2_5"])
        out["Betfair close"] = demargin(test["bfe_close_over_2_5"], test["bfe_close_under_2_5"])
        preds.append(out)
        print(f"  predicted {season}")
    P = pd.concat(preds, ignore_index=True)

    # --- detail: log loss per league x season ---------------------------------
    detail = []
    for (lg, season), g in P.groupby(["league_name", "season"]):
        y = g["over_2_5"].values
        row = {"league": lg, "season": season, "n": len(g)}
        for col in MODELS + ["Bookmaker average", "Betfair close"]:
            m = g[col].notna().values
            row[col] = pm_ll(y[m], g[col].values[m]).mean() if m.any() else np.nan
        detail.append(row)
    detail = pd.DataFrame(detail)
    detail.to_csv(DETAIL_CSV, index=False)

    # --- summary vs market, per league and overall ----------------------------
    summary = []
    for league in ["ALL"] + LEAGUES:
        g0 = P if league == "ALL" else P[P["league_name"] == league]
        for ref, seasons in [("Bookmaker average", None), ("Betfair close", BETFAIR_SEASONS)]:
            g = g0 if seasons is None else g0[g0["season"].isin(seasons)]
            g = g[g[ref].notna()]
            if g.empty:
                continue
            y = g["over_2_5"].values
            ll_ref = pm_ll(y, g[ref].values)
            for model in MODELS:
                d = pm_ll(y, g[model].values) - ll_ref
                boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
                better = sum(pm_ll(s["over_2_5"].values, s[model].values).mean()
                             < pm_ll(s["over_2_5"].values, s[ref].values).mean() for _, s in g.groupby("season"))
                summary.append({"league": league, "vs": ref, "model": model,
                                "model_ll": pm_ll(y, g[model].values).mean(), "market_ll": ll_ref.mean(),
                                "diff": d.mean(), "ci_low": np.percentile(boots, 2.5),
                                "ci_high": np.percentile(boots, 97.5),
                                "seasons_beating_market": f"{better}/{g['season'].nunique()}", "n": len(g)})
    summary = pd.DataFrame(summary)
    summary.to_csv(SUMMARY_CSV, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    for ref in ["Bookmaker average", "Betfair close"]:
        span = "6 seasons 2020/21-2025/26" if ref == "Bookmaker average" else "2024/25 + 2025/26"
        print(f"\n=== vs {ref} ({span}) - diff < 0 means model better ===")
        print(summary[summary["vs"] == ref].drop(columns="vs").round(4).to_string(index=False))

    allb = summary[(summary["league"] == "ALL") & (summary["vs"] == "Bookmaker average")].set_index("model")
    winner = allb["model_ll"].idxmin()
    print(f"\nPooled (A) vs per-country (B), all leagues: "
          f"{allb.loc[A, 'model_ll']:.4f} vs {allb.loc[B, 'model_ll']:.4f}")
    print(f"Best overall: {winner} ({allb.loc[winner, 'model_ll']:.4f}) vs bookmaker average "
          f"{allb.loc[winner, 'market_ll']:.4f}")

    # --- figure -----------------------------------------------------------------
    fig, axes = plt.subplots(1, len(LEAGUES), figsize=(16, 4), sharey=True)
    for ax, lg in zip(axes, LEAGUES):
        sub = detail[detail["league"] == lg]
        for col in [A, B, "Bookmaker average", "Betfair close"]:
            ax.plot(sub["season"], sub[col], marker="o", markersize=3, label=col,
                    linestyle="--" if col in ("Bookmaker average", "Betfair close") else "-")
        ax.set_title(lg)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    axes[0].set_ylabel("Log Loss")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rd_eu_upper.png"), dpi=120)
    plt.close(fig)

    lines = ["# v3-eu-upper backtest - La Liga, Serie A, Bundesliga, Ligue 1\n",
             "v3 recipe (Poisson alpha 1.3335 + per-league Dixon-Coles, 50/50), pooled vs per-country. "
             "6-fold walk-forward. `diff` < 0 = model better; 95% bootstrap interval.\n"]
    for ref in ["Bookmaker average", "Betfair close"]:
        lines += [f"\n## vs {ref}\n", md_table(summary[summary["vs"] == ref].drop(columns="vs")) + "\n"]
    lines += ["\n## Detail - log loss per league and season\n", md_table(detail) + "\n",
              f"\n**Best overall: {winner}** ({allb.loc[winner, 'model_ll']:.4f} vs bookmaker "
              f"{allb.loc[winner, 'market_ll']:.4f}).\n", "\n![EU upper](figures/rd_eu_upper.png)\n"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_MD}, {SUMMARY_CSV}, {DETAIL_CSV}, figures/rd_eu_upper.png")


if __name__ == "__main__":
    main()

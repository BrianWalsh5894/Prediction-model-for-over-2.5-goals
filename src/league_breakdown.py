"""
Step 1 - Per-league breakdown of the existing models vs the market.

Every R&D round so far compared all four divisions pooled together. A
small edge in one league (most plausibly League One / League Two, where
the market gets less attention) could be hidden by the Premier League.
This script re-runs round 3's walk-forward (same 6 folds, same models)
and splits every result by league.

Two market references, kept separate:
  - Bookmaker average (Football-Data Avg>2.5, margin removed), all 6
    folds 2020/21 - 2025/26.
  - Betfair Exchange CLOSING price, 2024/25 + 2025/26 only - the honest
    benchmark for betting on Betfair.

For each league the per-match log loss difference (model - market) is
pooled across folds, with a 95% bootstrap interval. Negative = model
better. "Real edge" would need the WHOLE interval below zero.

Nothing is tuned here - it's a read-out of the round 3 models.

Run from inside src/ (about 1 minute):
    python league_breakdown.py
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
from rd_experiments3 import (  # noqa: E402
    rolling_dc, poisson_fit_predict, DC_FEATURES, NON_FEATURE_COLS,
    ALL_SEASONS_CHRONOLOGICAL, WALK_FORWARD_MIN_TRAIN_SEASONS,
)

FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
REPORT_MD = os.path.join("..", "reports", "league_breakdown.md")
DETAIL_CSV = os.path.join("..", "reports", "league_breakdown_detail.csv")
SUMMARY_CSV = os.path.join("..", "reports", "league_breakdown_summary.csv")
FIG_DIR = os.path.join("..", "reports", "figures")

LEAGUES = ["Premier League", "Championship", "League One", "League Two"]
DC_XI = 0.0
SEED = 424410
N_BOOT = 2000

MODELS = [
    "50/50 blend: Poisson + Dixon-Coles",
    "Poisson + rolling Dixon-Coles features",
    "Poisson baseline",
]


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def per_match_ll(y, p):
    p = clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def demargin(o, u):
    io, iu = 1 / o, 1 / u
    return io / (io + iu)


def boot_ci(d, rng):
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


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


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])
    df = df[df["season"].isin(ALL_SEASONS_CHRONOLOGICAL)].reset_index(drop=True).copy()
    clean = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    clean = clean[clean["season"].isin(ALL_SEASONS_CHRONOLOGICAL)]
    goals_df = clean[["match_id", "home_goals", "away_goals"]]
    base_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    print("Rolling Dixon-Coles (xi=0)...")
    dfx = df.merge(rolling_dc(clean, DC_XI), on="match_id", how="left")

    # --- Walk-forward predictions, one row per test match -------------------
    preds = []
    for i in range(WALK_FORWARD_MIN_TRAIN_SEASONS, len(ALL_SEASONS_CHRONOLOGICAL)):
        season = ALL_SEASONS_CHRONOLOGICAL[i]
        train = dfx[dfx["season"].isin(ALL_SEASONS_CHRONOLOGICAL[:i])]
        test = dfx[dfx["season"] == season].copy()
        p_base = poisson_fit_predict(train, test, base_cols, goals_df)
        p_dc = test["dc_p_over"].fillna(pd.Series(p_base, index=test.index)).values
        out = test[["match_id", "season", "league_name", "over_2_5"]].copy()
        out["Poisson baseline"] = p_base
        out["Poisson + rolling Dixon-Coles features"] = poisson_fit_predict(
            train, test, base_cols + DC_FEATURES, goals_df)
        out["50/50 blend: Poisson + Dixon-Coles"] = 0.5 * p_base + 0.5 * p_dc
        out["Bookmaker average"] = demargin(test["avg_over_2_5"], test["avg_under_2_5"]).values
        out["Betfair close"] = demargin(test["bfe_close_over_2_5"], test["bfe_close_under_2_5"]).values
        preds.append(out)
        print(f"  predicted {season}")
    P = pd.concat(preds, ignore_index=True)
    y_all = P["over_2_5"].values

    # --- Per league x season detail ----------------------------------------
    detail_rows = []
    for (league, season), g in P.groupby(["league_name", "season"]):
        y = g["over_2_5"].values
        row = {"league": league, "season": season, "n": len(g), "over_rate": y.mean()}
        for col in MODELS + ["Bookmaker average", "Betfair close"]:
            m = g[col].notna().values
            row[col] = per_match_ll(y[m], g[col].values[m]).mean() if m.any() else np.nan
        detail_rows.append(row)
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(DETAIL_CSV, index=False)

    # --- Per league summary with bootstrap CIs ------------------------------
    summary_rows = []
    for league in LEAGUES + ["ALL LEAGUES"]:
        g = P if league == "ALL LEAGUES" else P[P["league_name"] == league]
        y = g["over_2_5"].values
        for ref, seasons in [("Bookmaker average", None), ("Betfair close", ["2024/25", "2025/26"])]:
            gm = g if seasons is None else g[g["season"].isin(seasons)]
            gm = gm[gm[ref].notna()]
            ym = gm["over_2_5"].values
            ll_ref = per_match_ll(ym, gm[ref].values)
            for model in MODELS:
                d = per_match_ll(ym, gm[model].values) - ll_ref
                lo, hi = boot_ci(d, rng)
                seasons_better = sum(
                    per_match_ll(s["over_2_5"].values, s[model].values).mean()
                    < per_match_ll(s["over_2_5"].values, s[ref].values).mean()
                    for _, s in gm.groupby("season")
                )
                summary_rows.append({
                    "league": league, "vs": ref, "model": model,
                    "model_ll": per_match_ll(ym, gm[model].values).mean(),
                    "market_ll": ll_ref.mean(), "diff": d.mean(),
                    "ci_low": lo, "ci_high": hi,
                    "seasons_beating_market": f"{int(seasons_better)}/{gm['season'].nunique()}",
                    "n": len(gm),
                })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    for ref in ["Bookmaker average", "Betfair close"]:
        span = "6 seasons 2020/21-2025/26" if ref == "Bookmaker average" else "2024/25 + 2025/26"
        print(f"\n=== vs {ref} ({span}) - diff < 0 means model better ===")
        print(summary[summary["vs"] == ref].drop(columns="vs").round(4).to_string(index=False))

    # --- Figure: best model's diff vs bookmaker, per league, with CI --------
    best = MODELS[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, ref in zip(axes, ["Bookmaker average", "Betfair close"]):
        s = summary[(summary["vs"] == ref) & (summary["model"] == best)]
        x = np.arange(len(s))
        ax.errorbar(x, s["diff"], yerr=[s["diff"] - s["ci_low"], s["ci_high"] - s["diff"]],
                    fmt="o", capsize=4)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(s["league"], rotation=20, fontsize=8)
        ax.set_title(f"{best}\nvs {ref} (below 0 = model better)", fontsize=9)
        ax.set_ylabel("Log loss difference per match")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "league_breakdown.png"), dpi=120)
    plt.close(fig)

    # --- Report ---------------------------------------------------------------
    lines = [
        "# Per-league breakdown vs the market\n",
        "Round 3 models (same 6-fold walk-forward), split by league. `diff` = mean per-match "
        "log loss of model minus market; negative = model better. 95% bootstrap interval in "
        "`ci_low`/`ci_high`. A real edge needs the whole interval below zero.\n",
    ]
    for ref in ["Bookmaker average", "Betfair close"]:
        span = "6 seasons, 2020/21 - 2025/26" if ref == "Bookmaker average" else "2024/25 + 2025/26"
        lines.append(f"\n## vs {ref} ({span})\n")
        lines.append(md_table(summary[summary["vs"] == ref].drop(columns="vs")) + "\n")
    lines.append("\n## Detail - log loss per league and season\n")
    lines.append(md_table(detail) + "\n")
    lines.append("\n![League breakdown](figures/league_breakdown.png)\n")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_MD}, {SUMMARY_CSV}, {DETAIL_CSV}, figures/league_breakdown.png")


if __name__ == "__main__":
    main()

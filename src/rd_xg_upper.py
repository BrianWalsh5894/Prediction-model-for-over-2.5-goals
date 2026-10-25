"""
Step 4 R&D - does Understat xG help predict Over 2.5 in the Premier League?

Premier League only (Understat has no Championship). Needs match_xg in the
database - run download_understat.py first.

xG features (all leak-free: shifted per team so a match never sees its own
xG, same rule as Phase 5; a team's history = its Premier League matches):
  - xg_for / xg_against: rolling last 5 and last 10, season-to-date average
  - home-only / away-only rolling 5 (venue form)
  - finishing luck: goals_for_roll10 - xg_for_roll10 (and the same against)
  - match-level: combined expected xG for each side, and their total
Promoted teams have no PL xG history at first: missing values are filled
with the training medians (same as every other feature).

Same 6-fold walk-forward as rounds 1-3 (predicting 2020/21 - 2025/26),
scored on Premier League matches only. Candidates:
  A. v3 blend as it is (trained on all four leagues, base features)
  B. Poisson, base features, trained on PL only
  C. Poisson, base + xG features, PL only
  D. Poisson, xG features + Elo only (small model), PL only
  E. 50/50 blend of C with Dixon-Coles
Benchmarks: bookmaker average (6 folds) and Betfair closing price
(2024/25 + 2025/26), with 95% bootstrap intervals on the difference.

Run from inside src/ (about 1 minute):
    python rd_xg_upper.py
"""

import os
import sys
import sqlite3
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

FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
DB_PATH = os.path.join("..", "data", "over25.db")
REPORT_MD = os.path.join("..", "reports", "rd_xg_upper.md")
DETAIL_CSV = os.path.join("..", "reports", "rd_xg_upper_comparison.csv")
FIG_DIR = os.path.join("..", "reports", "figures")

ALL_SEASONS = ["2017/18", "2018/19", "2019/20", "2020/21", "2021/22",
               "2022/23", "2023/24", "2024/25", "2025/26"]
FOLD_START = 3
LEAGUE = "Premier League"
SEED = 424410
N_BOOT = 2000


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


def build_xg_features(pl):
    """pl: PL matches with goals and xG, sorted by date. Returns one row per
    match_id with leak-free xG features."""
    home = pd.DataFrame({"match_id": pl["match_id"], "match_date": pl["match_date"], "season": pl["season"],
                         "team": pl["home_team"], "is_home": 1,
                         "xg_for": pl["home_xg"], "xg_against": pl["away_xg"],
                         "gf": pl["home_goals"], "ga": pl["away_goals"]})
    away = pd.DataFrame({"match_id": pl["match_id"], "match_date": pl["match_date"], "season": pl["season"],
                         "team": pl["away_team"], "is_home": 0,
                         "xg_for": pl["away_xg"], "xg_against": pl["home_xg"],
                         "gf": pl["away_goals"], "ga": pl["home_goals"]})
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "match_date", "match_id"])
    g = long.groupby("team")
    feats = {}
    for col in ["xg_for", "xg_against", "gf", "ga"]:
        prev = g[col].shift(1)                                  # shift BEFORE rolling - no leakage
        for w in (5, 10):
            long[f"{col}_r{w}"] = prev.groupby(long["team"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        long[f"{col}_season"] = prev.groupby([long["team"], long["season"]]).transform(lambda s: s.expanding().mean())
    for col in ["xg_for", "xg_against"]:
        prev_v = long.groupby(["team", "is_home"])[col].shift(1)
        long[f"{col}_venue_r5"] = prev_v.groupby([long["team"], long["is_home"]]).transform(
            lambda s: s.rolling(5, min_periods=1).mean())
    long["luck_for_r10"] = long["gf_r10"] - long["xg_for_r10"]
    long["luck_against_r10"] = long["ga_r10"] - long["xg_against_r10"]

    keep = ["xg_for_r5", "xg_for_r10", "xg_against_r5", "xg_against_r10", "xg_for_season",
            "xg_against_season", "xg_for_venue_r5", "xg_against_venue_r5", "luck_for_r10", "luck_against_r10"]
    h = long[long["is_home"] == 1].set_index("match_id")[keep].add_prefix("xg_home_")
    a = long[long["is_home"] == 0].set_index("match_id")[keep].add_prefix("xg_away_")
    out = h.join(a)
    out["xg_exp_home"] = (out["xg_home_xg_for_r10"] + out["xg_away_xg_against_r10"]) / 2
    out["xg_exp_away"] = (out["xg_away_xg_for_r10"] + out["xg_home_xg_against_r10"]) / 2
    out["xg_exp_total"] = out["xg_exp_home"] + out["xg_exp_away"]
    return out.reset_index()


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(FIG_DIR, exist_ok=True)

    feats = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])
    feats = feats[feats["season"].isin(ALL_SEASONS)].reset_index(drop=True).copy()
    clean = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    clean = clean[clean["season"].isin(ALL_SEASONS)]
    goals_df = clean[["match_id", "home_goals", "away_goals"]]
    base_cols = [c for c in feats.columns if c not in NON_FEATURE_COLS]

    with sqlite3.connect(DB_PATH) as conn:
        xg = pd.read_sql_query("SELECT match_id, home_xg, away_xg FROM match_xg", conn)
    pl = clean[clean["league_name"] == LEAGUE].merge(xg, on="match_id", how="left")
    cover = pl.groupby("season")["home_xg"].apply(lambda s: s.notna().mean())
    print("xG coverage of our PL matches by season:")
    print(cover.round(3).to_string())
    if cover.min() < 0.95:
        print("WARNING: xG missing for >5% of matches in some season - check download_understat.py output.")
    pl = pl.sort_values(["match_date", "match_id"])
    xg_feats = build_xg_features(pl)
    xg_cols = [c for c in xg_feats.columns if c != "match_id"]
    print(f"{len(xg_cols)} xG features built")

    print("Rolling Dixon-Coles (xi=0, all leagues)...")
    dc = rolling_dc(clean, 0.0)[["match_id", "dc_p_over"]]
    dfx = feats.merge(dc, on="match_id", how="left").merge(xg_feats, on="match_id", how="left")
    small_cols = xg_cols + ["home_elo_pre", "away_elo_pre"]

    preds = []
    for i in range(FOLD_START, len(ALL_SEASONS)):
        season = ALL_SEASONS[i]
        train_all = dfx[dfx["season"].isin(ALL_SEASONS[:i])]
        train_pl = train_all[train_all["league_name"] == LEAGUE]
        test = dfx[(dfx["season"] == season) & (dfx["league_name"] == LEAGUE)].copy()

        p_all = poisson_fit_predict(train_all, test, base_cols, goals_df)
        p_dc = test["dc_p_over"].fillna(pd.Series(p_all, index=test.index)).values
        out = test[["match_id", "season", "over_2_5"]].copy()
        out["A. v3 blend (all leagues, base)"] = 0.5 * p_all + 0.5 * p_dc
        out["B. Poisson base, PL only"] = poisson_fit_predict(train_pl, test, base_cols, goals_df)
        p_c = poisson_fit_predict(train_pl, test, base_cols + xg_cols, goals_df)
        out["C. Poisson base + xG, PL only"] = p_c
        out["D. Poisson xG + Elo only, PL only"] = poisson_fit_predict(train_pl, test, small_cols, goals_df)
        out["E. 50/50 blend: C + Dixon-Coles"] = 0.5 * p_c + 0.5 * p_dc
        out["Bookmaker average"] = demargin(test["avg_over_2_5"], test["avg_under_2_5"]).values
        out["Betfair close"] = demargin(test["bfe_close_over_2_5"], test["bfe_close_under_2_5"]).values
        preds.append(out)
        print(f"  predicted {season}")
    P = pd.concat(preds, ignore_index=True)
    models = [c for c in P.columns if c[:2] in ("A.", "B.", "C.", "D.", "E.")]

    # per-season detail
    detail = []
    for season, g in P.groupby("season"):
        y = g["over_2_5"].values
        row = {"season": season, "n": len(g)}
        for col in models + ["Bookmaker average", "Betfair close"]:
            m = g[col].notna().values
            row[col] = pm_ll(y[m], g[col].values[m]).mean() if m.any() else np.nan
        detail.append(row)
    detail = pd.DataFrame(detail)
    detail.to_csv(DETAIL_CSV, index=False)

    summary = []
    for ref, seasons in [("Bookmaker average", None), ("Betfair close", ["2024/25", "2025/26"])]:
        g = P if seasons is None else P[P["season"].isin(seasons)]
        g = g[g[ref].notna()]
        y = g["over_2_5"].values
        ll_ref = pm_ll(y, g[ref].values)
        for model in models:
            d = pm_ll(y, g[model].values) - ll_ref
            boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
            better = sum(pm_ll(s["over_2_5"].values, s[model].values).mean()
                         < pm_ll(s["over_2_5"].values, s[ref].values).mean() for _, s in g.groupby("season"))
            summary.append({"vs": ref, "model": model, "model_ll": pm_ll(y, g[model].values).mean(),
                            "market_ll": ll_ref.mean(), "diff": d.mean(),
                            "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                            "seasons_beating_market": f"{better}/{g['season'].nunique()}", "n": len(g)})
    summary = pd.DataFrame(summary)

    pd.set_option("display.width", 220)
    print("\nPer-season log loss (Premier League):")
    print(detail.round(4).to_string(index=False))
    for ref in ["Bookmaker average", "Betfair close"]:
        print(f"\n=== vs {ref} - diff < 0 means model better ===")
        print(summary[summary["vs"] == ref].drop(columns="vs").round(4).to_string(index=False))

    b = summary[(summary["vs"] == "Bookmaker average")].set_index("model")
    xg_gain = b.loc["C. Poisson base + xG, PL only", "model_ll"] - b.loc["B. Poisson base, PL only", "model_ll"]
    print(f"\nEffect of adding xG (C minus B): {xg_gain:+.4f} log loss (negative = xG helps)")

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in models + ["Bookmaker average", "Betfair close"]:
        ax.plot(detail["season"], detail[col], marker="o", label=col,
                linestyle="--" if col in ("Bookmaker average", "Betfair close") else "-", markersize=4)
    ax.set_title("Premier League - xG R&D vs market")
    ax.set_ylabel("Log Loss")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rd_xg_upper.png"), dpi=120)
    plt.close(fig)

    lines = ["# Step 4 R&D - Understat xG, Premier League\n",
             f"{len(xg_cols)} leak-free xG features. 6-fold walk-forward (2020/21 - 2025/26), PL matches only.\n",
             "## Per-season log loss\n", md_table(detail) + "\n"]
    for ref in ["Bookmaker average", "Betfair close"]:
        lines += [f"\n## vs {ref} (diff < 0 = model better; 95% bootstrap interval)\n",
                  md_table(summary[summary["vs"] == ref].drop(columns="vs")) + "\n"]
    lines += [f"\n**Effect of adding xG (C minus B): {xg_gain:+.4f}** log loss (negative = xG helps).\n",
              "\n![xG R&D](figures/rd_xg_upper.png)\n"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_MD}, {DETAIL_CSV}, figures/rd_xg_upper.png")


if __name__ == "__main__":
    main()

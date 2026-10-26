"""
v3-eu-upper - improvement round 1: second divisions + Understat xG.

Round 1 (rd_eu_upper.py) lost to the market by +0.0055 (bookmaker average)
and +0.0086 (Betfair close). Two likely weaknesses:
  1. Promoted teams arrived with no history: no form features and a
     league-average Dixon-Coles rating. England avoids this because all
     four divisions are in the data and linked by promotion.
  2. No xG.

Fixes tested here (predictions are for TOP-division matches only):
  A. Round-1 baseline, rebuilt here as a check (should reproduce ~0.6768)
  B. + second divisions: features built on top + second divisions (so a
     promoted team carries its form), Poisson trained on both divisions,
     Dixon-Coles fitted per COUNTRY on both divisions (linked by promotion,
     like England)
  C. Same as B, but Poisson trained on top divisions only
  D. B features + Understat xG features, Poisson on top divisions
     + per-country Dixon-Coles
  E. Compact: xG features + Elo only + per-country Dixon-Coles

Success bar (set BEFORE running): freeze as v3-eu-upper only if the best
candidate is within +0.002 of the Betfair close (v3's gap in England).

Needs: download_eu_upper.py (with second divisions), clean_data.py,
build_features_eu_upper.py and download_understat.py run first.

Run from inside src/ (a few minutes):
    python rd_eu_upper2.py
"""

import os
import sys
import sqlite3
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from rd_experiments3 import rolling_dc, poisson_fit_predict, NON_FEATURE_COLS  # noqa: E402
from rd_xg_upper import build_xg_features  # noqa: E402
from build_features import build_feature_table  # noqa: E402

FEATURES_CSV = os.path.join("..", "data", "clean", "features_eu_upper.csv")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean_eu_upper.csv")
DB_PATH = os.path.join("..", "data", "over25.db")
REPORT_MD = os.path.join("..", "reports", "rd_eu_upper2.md")
SUMMARY_CSV = os.path.join("..", "reports", "rd_eu_upper2_summary.csv")

COUNTRIES = {"La Liga": "La Liga 2", "Serie A": "Serie B", "Bundesliga": "2. Bundesliga", "Ligue 1": "Ligue 2"}
TOP = list(COUNTRIES)
SECOND = list(COUNTRIES.values())
ALL_SEASONS = ["2017/18", "2018/19", "2019/20", "2020/21", "2021/22",
               "2022/23", "2023/24", "2024/25", "2025/26"]
FOLD_START = 3
BETFAIR_SEASONS = ["2024/25", "2025/26"]
ENGLAND_GAP_TO_CLOSE = 0.002
SEED = 424410
N_BOOT = 2000

A = "A. Round-1 baseline (top divisions only)"
B = "B. + 2nd divisions (Poisson on both)"
C = "C. + 2nd divisions (Poisson on top only)"
D = "D. + 2nd divisions + xG"
E = "E. Compact: xG + Elo + DC"
MODELS = [A, B, C, D, E]


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
    return "\n".join([header, sep] + ["| " + " | ".join(r) + " |" for r in df.itertuples(index=False, name=None)])


def build_predictions():
    """Walk-forward predictions for candidates A-E (one row per top-division
    test match), plus market prices. Also used by rd_eu_upper3.py."""
    clean = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    clean = clean[clean["season"].isin(ALL_SEASONS)]
    goals_df = clean[["match_id", "home_goals", "away_goals"]]
    have_second = clean["league_name"].isin(SECOND).any()
    print("Matches per league:")
    print(clean.groupby("league_name").size().to_string())
    if not have_second:
        print("\nWARNING: no second-division matches found - run download_eu_upper.py, clean_data.py and "
              "build_features_eu_upper.py again first. B-E will equal the baseline.")

    # Features with second-division history (from build_features_eu_upper.py)
    feats_all = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])
    feats_all = feats_all[feats_all["season"].isin(ALL_SEASONS)].reset_index(drop=True).copy()
    # Round-1 features: top divisions only, rebuilt the same way
    feats_top = build_feature_table(clean[clean["league_name"].isin(TOP)].copy())
    feats_top = feats_top[feats_top["season"].isin(ALL_SEASONS)].reset_index(drop=True).copy()
    cols_top = [c for c in feats_top.columns if c not in NON_FEATURE_COLS]
    cols_all = [c for c in feats_all.columns if c not in NON_FEATURE_COLS]

    # xG features (top divisions only - Understat has no second divisions)
    with sqlite3.connect(DB_PATH) as conn:
        xg = pd.read_sql_query("SELECT match_id, home_xg, away_xg FROM match_xg", conn)
    top_matches = clean[clean["league_name"].isin(TOP)].merge(xg, on="match_id", how="left")
    cover = top_matches.groupby("league_name")["home_xg"].apply(lambda s: s.notna().mean())
    print("\nxG coverage of top-division matches:")
    print(cover.round(3).to_string())
    xg_feats = build_xg_features(top_matches.sort_values(["match_date", "match_id"]))
    xg_cols = [c for c in xg_feats.columns if c != "match_id"]

    print("\nDixon-Coles: per league (round 1) and per country incl. second division...")
    dc_top = pd.concat([rolling_dc(clean[clean["league_name"] == lg], 0.0) for lg in TOP])
    dc_ctry = pd.concat([rolling_dc(clean[clean["league_name"].isin([lg, COUNTRIES[lg]])], 0.0) for lg in TOP])
    dc_top = dc_top[["match_id", "dc_p_over"]].rename(columns={"dc_p_over": "dc_top"})
    dc_ctry = dc_ctry[["match_id", "dc_p_over"]].rename(columns={"dc_p_over": "dc_ctry"})

    ft = feats_top.merge(dc_top, on="match_id", how="left")
    fa = feats_all.merge(dc_ctry, on="match_id", how="left").merge(xg_feats, on="match_id", how="left")
    small_cols = xg_cols + ["home_elo_pre", "away_elo_pre"]

    preds = []
    for i in range(FOLD_START, len(ALL_SEASONS)):
        season = ALL_SEASONS[i]
        prev = ALL_SEASONS[:i]
        # A - round 1
        trA, teA = ft[ft["season"].isin(prev)], ft[ft["season"] == season]
        pA_pois = pd.Series(poisson_fit_predict(trA, teA, cols_top, goals_df), index=teA["match_id"])
        pA = 0.5 * pA_pois + 0.5 * teA.set_index("match_id")["dc_top"].fillna(pA_pois)
        # B-E - with second divisions
        tr_both = fa[fa["season"].isin(prev)]
        tr_top = tr_both[tr_both["league_name"].isin(TOP)]
        te = fa[(fa["season"] == season) & fa["league_name"].isin(TOP)]
        idx = te["match_id"]
        pB_pois = pd.Series(poisson_fit_predict(tr_both, te, cols_all, goals_df), index=idx)
        pC_pois = pd.Series(poisson_fit_predict(tr_top, te, cols_all, goals_df), index=idx)
        pD_pois = pd.Series(poisson_fit_predict(tr_top, te, cols_all + xg_cols, goals_df), index=idx)
        pE_pois = pd.Series(poisson_fit_predict(tr_top, te, small_cols, goals_df), index=idx)
        dc = te.set_index("match_id")["dc_ctry"]
        out = te[["match_id", "season", "league_name", "over_2_5"]].set_index("match_id").copy()
        out[A] = pA.reindex(out.index)
        out[B] = 0.5 * pB_pois + 0.5 * dc.fillna(pB_pois)
        out[C] = 0.5 * pC_pois + 0.5 * dc.fillna(pC_pois)
        out[D] = 0.5 * pD_pois + 0.5 * dc.fillna(pD_pois)
        out[E] = 0.5 * pE_pois + 0.5 * dc.fillna(pE_pois)
        out["Bookmaker average"] = demargin(te["avg_over_2_5"], te["avg_under_2_5"]).values
        out["Betfair close"] = demargin(te["bfe_close_over_2_5"], te["bfe_close_under_2_5"]).values
        for col in ["bfe_over_2_5", "bfe_under_2_5", "bfe_close_over_2_5", "bfe_close_under_2_5"]:
            out[col] = te[col].values
        preds.append(out.reset_index())
        print(f"  predicted {season}")
    return pd.concat(preds, ignore_index=True)


def main():
    rng = np.random.default_rng(SEED)
    P = build_predictions()

    summary = []
    for league in ["ALL"] + TOP:
        g0 = P if league == "ALL" else P[P["league_name"] == league]
        for ref, seasons in [("Bookmaker average", None), ("Betfair close", BETFAIR_SEASONS)]:
            g = g0 if seasons is None else g0[g0["season"].isin(seasons)]
            g = g[g[ref].notna() & g[MODELS].notna().all(axis=1)]
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
        print(f"\n=== vs {ref} - diff < 0 means model better ===")
        print(summary[summary["vs"] == ref].drop(columns="vs").round(4).to_string(index=False))

    close_all = summary[(summary["league"] == "ALL") & (summary["vs"] == "Betfair close")].set_index("model")
    best = close_all["diff"].idxmin()
    gap = close_all.loc[best, "diff"]
    passed = gap <= ENGLAND_GAP_TO_CLOSE
    print(f"\nBest vs Betfair close: {best}  gap {gap:+.4f}  (bar: +{ENGLAND_GAP_TO_CLOSE:.4f}) -> "
          f"{'PASSES - candidate to freeze as v3-eu-upper' if passed else 'does NOT pass the bar'}")

    lines = ["# v3-eu-upper improvement round 1 - second divisions + xG\n",
             "Top-division matches only. 6-fold walk-forward. `diff` < 0 = model better; 95% bootstrap interval.\n",
             f"Success bar set before running: within +{ENGLAND_GAP_TO_CLOSE} of the Betfair close.\n"]
    for ref in ["Bookmaker average", "Betfair close"]:
        lines += [f"\n## vs {ref}\n", md_table(summary[summary["vs"] == ref].drop(columns="vs")) + "\n"]
    lines.append(f"\n**Best vs Betfair close: {best}, gap {gap:+.4f} - "
                 f"{'passes' if passed else 'does not pass'} the bar.**\n")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_MD}, {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

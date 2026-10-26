"""
v3-eu-upper-second - R&D round 1: predict the SECOND divisions
(La Liga 2, Serie B, 2. Bundesliga, Ligue 2).

v3-eu-upper predicts the top four European leagues and uses the second
divisions only as history for promoted teams. Those top leagues are some of
the sharpest betting markets in the world. In England the model came
closest to the market in League Two, so this tests the same v3 recipe on
the second divisions as TARGET leagues.

Candidates (all: v3 recipe = 50/50 Poisson regression + Dixon-Coles, no
time decay; features from features_eu_upper.csv, which already carry
top-division history for relegated teams; no xG - Understat has no second
divisions):
  A. Poisson trained on second divisions only + Dixon-Coles per country
     (top + second division, linked by promotion/relegation)
  B. Poisson trained on both divisions + Dixon-Coles per country
  C. Poisson trained on second divisions only + Dixon-Coles per second
     division only

Benchmarks: bookmaker average (all folds) and Betfair close (2024/25 +
2025/26). Also simulates the betting rule (flat 1 unit, EV >= 3% after 2%
commission) at the Betfair early and closing prices, per league.

Success bar (set BEFORE running, same as v3-eu-upper): freeze as
v3-eu-upper-second only if the best candidate is within +0.002 of the
Betfair close over 2024/25 + 2025/26.

Needs the same data as rd_eu_upper2.py (matches_clean_eu_upper.csv and
features_eu_upper.csv with second divisions). Run from inside src/
(a few minutes):
    python rd_eu_upper_second.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from rd_experiments3 import rolling_dc, poisson_fit_predict, NON_FEATURE_COLS  # noqa: E402
import rd_eu_upper2 as r2  # noqa: E402
from rd_eu_upper3 import bet_sim  # noqa: E402

REPORT_MD = os.path.join("..", "reports", "rd_eu_upper_second.md")
SUMMARY_CSV = os.path.join("..", "reports", "rd_eu_upper_second_summary.csv")
BETS_CSV = os.path.join("..", "reports", "rd_eu_upper_second_bets.csv")
PREDS_CSV = os.path.join("..", "reports", "rd_eu_upper_second_predictions.csv")

COUNTRIES = r2.COUNTRIES                 # top -> second
SECOND = r2.SECOND
TOP_OF = {v: k for k, v in COUNTRIES.items()}
ALL_SEASONS = r2.ALL_SEASONS
FOLD_START = r2.FOLD_START
BETFAIR_SEASONS = r2.BETFAIR_SEASONS
BAR = 0.002
SEED = 424410
N_BOOT = 2000

A = "A. Poisson 2nd div + DC per country"
B = "B. Poisson both divs + DC per country"
C = "C. Poisson 2nd div + DC 2nd div only"
MODELS = [A, B, C]


def build_predictions():
    clean = pd.read_csv(r2.CLEAN_CSV, parse_dates=["match_date"])
    clean = clean[clean["season"].isin(ALL_SEASONS)]
    goals_df = clean[["match_id", "home_goals", "away_goals"]]
    print("Matches per league:")
    print(clean.groupby("league_name").size().to_string())

    feats = pd.read_csv(r2.FEATURES_CSV, parse_dates=["match_date"])
    feats = feats[feats["season"].isin(ALL_SEASONS)].reset_index(drop=True)
    cols = [c for c in feats.columns if c not in NON_FEATURE_COLS]
    print(f"\n{len(cols)} features")

    print("\nDixon-Coles: per country (top + second) and per second division only...")
    dc_ctry = pd.concat([rolling_dc(clean[clean["league_name"].isin([TOP_OF[lg], lg])], 0.0) for lg in SECOND])
    dc_sec = pd.concat([rolling_dc(clean[clean["league_name"] == lg], 0.0) for lg in SECOND])
    dc_ctry = dc_ctry[["match_id", "dc_p_over"]].rename(columns={"dc_p_over": "dc_ctry"})
    dc_sec = dc_sec[["match_id", "dc_p_over"]].rename(columns={"dc_p_over": "dc_sec"})
    fa = feats.merge(dc_ctry, on="match_id", how="left").merge(dc_sec, on="match_id", how="left")

    preds = []
    for i in range(FOLD_START, len(ALL_SEASONS)):
        season, prev = ALL_SEASONS[i], ALL_SEASONS[:i]
        tr_both = fa[fa["season"].isin(prev)]
        tr_sec = tr_both[tr_both["league_name"].isin(SECOND)]
        te = fa[(fa["season"] == season) & fa["league_name"].isin(SECOND)]
        idx = te["match_id"]
        p_sec = pd.Series(poisson_fit_predict(tr_sec, te, cols, goals_df), index=idx)
        p_both = pd.Series(poisson_fit_predict(tr_both, te, cols, goals_df), index=idx)
        dcc = te.set_index("match_id")["dc_ctry"]
        dcs = te.set_index("match_id")["dc_sec"]
        out = te[["match_id", "season", "league_name", "over_2_5"]].set_index("match_id").copy()
        out[A] = 0.5 * p_sec + 0.5 * dcc.fillna(p_sec)
        out[B] = 0.5 * p_both + 0.5 * dcc.fillna(p_both)
        out[C] = 0.5 * p_sec + 0.5 * dcs.fillna(p_sec)
        out["Bookmaker average"] = r2.demargin(te["avg_over_2_5"], te["avg_under_2_5"]).values
        out["Betfair close"] = r2.demargin(te["bfe_close_over_2_5"], te["bfe_close_under_2_5"]).values
        for col in ["bfe_over_2_5", "bfe_under_2_5", "bfe_close_over_2_5", "bfe_close_under_2_5"]:
            out[col] = te[col].values
        preds.append(out.reset_index())
        print(f"  predicted {season}")
    return pd.concat(preds, ignore_index=True)


def main():
    rng = np.random.default_rng(SEED)
    P = build_predictions()
    P.to_csv(PREDS_CSV, index=False)

    # ---- log loss vs market, overall and per league ----
    summary = []
    for league in ["ALL"] + SECOND:
        g0 = P if league == "ALL" else P[P["league_name"] == league]
        for ref, seasons in [("Bookmaker average", None), ("Betfair close", BETFAIR_SEASONS)]:
            g = g0 if seasons is None else g0[g0["season"].isin(seasons)]
            g = g[g[ref].notna() & g[MODELS].notna().all(axis=1)]
            y = g["over_2_5"].values
            ll_ref = r2.pm_ll(y, g[ref].values)
            for model in MODELS:
                ll_m = r2.pm_ll(y, g[model].values)
                d = ll_m - ll_ref
                boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
                better = sum(r2.pm_ll(s["over_2_5"].values, s[model].values).mean()
                             < r2.pm_ll(s["over_2_5"].values, s[ref].values).mean() for _, s in g.groupby("season"))
                summary.append({"league": league, "vs": ref, "model": model,
                                "model_ll": ll_m.mean(), "market_ll": ll_ref.mean(), "diff": d.mean(),
                                "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                                "seasons_beating_market": f"{better}/{g['season'].nunique()}", "n": len(g)})
    summary = pd.DataFrame(summary)
    summary.to_csv(SUMMARY_CSV, index=False)

    # ---- betting simulation (Betfair seasons) ----
    Bm = P[P["season"].isin(BETFAIR_SEASONS) & P["Betfair close"].notna()]
    bets = []
    for model in MODELS:
        for league in ["ALL"] + SECOND:
            g = Bm if league == "ALL" else Bm[Bm["league_name"] == league]
            for label, oc, uc in [("Betfair early", "bfe_over_2_5", "bfe_under_2_5"),
                                  ("Betfair close", "bfe_close_over_2_5", "bfe_close_under_2_5")]:
                bets.append({"model": model, "league": league, "price": label, **bet_sim(g, model, oc, uc, rng)})
    bets = pd.DataFrame(bets)
    bets.to_csv(BETS_CSV, index=False)

    gap_pp = {m: (Bm[m] - Bm["Betfair close"]).abs().mean() * 100 for m in MODELS}
    corr = {m: np.corrcoef(Bm[m], Bm["Betfair close"])[0, 1] for m in MODELS}

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    for ref in ["Bookmaker average", "Betfair close"]:
        print(f"\n=== vs {ref} - diff < 0 means model better ===")
        print(summary[summary["vs"] == ref].drop(columns="vs").round(4).to_string(index=False))
    print(f"\nHow far off (2024/25 + 2025/26, {len(Bm)} matches):")
    for m in MODELS:
        print(f"  {m}: average gap to Betfair close {gap_pp[m]:.1f} pp, correlation {corr[m]:.2f}")
    print("\nBetting simulation - flat 1 unit, EV >= 3% after 2% commission (2024/25 + 2025/26):")
    print(bets.round(4).to_string(index=False))

    close_all = summary[(summary["league"] == "ALL") & (summary["vs"] == "Betfair close")].set_index("model")
    best = close_all["diff"].idxmin()
    gap = close_all.loc[best, "diff"]
    passed = gap <= BAR
    verdict = "PASSES - candidate to freeze as v3-eu-upper-second" if passed else "does NOT pass the bar"
    print(f"\nBest vs Betfair close: {best}  gap {gap:+.4f}  (bar: +{BAR:.4f}) -> {verdict}")

    lines = ["# v3-eu-upper-second - R&D round 1 (second divisions as targets)\n",
             "La Liga 2, Serie B, 2. Bundesliga, Ligue 2. 6-fold walk-forward (2020/21-2025/26). "
             "`diff` < 0 = model better; 95% bootstrap interval.\n",
             f"Success bar set before running: within +{BAR} of the Betfair close.\n"]
    for ref in ["Bookmaker average", "Betfair close"]:
        lines += [f"\n## vs {ref}\n", r2.md_table(summary[summary["vs"] == ref].drop(columns="vs")) + "\n"]
    lines.append(f"\n## How far off (2024/25 + 2025/26, {len(Bm)} matches)\n")
    lines += [f"- {m}: average gap {gap_pp[m]:.1f} pp, correlation {corr[m]:.2f}" for m in MODELS]
    lines += ["\n## Betting simulation (flat 1 unit, EV >= 3% after 2% commission)\n", r2.md_table(bets) + "\n",
              f"\n**Best vs Betfair close: {best}, gap {gap:+.4f} - {'passes' if passed else 'does not pass'} the bar.**\n"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_MD}, {SUMMARY_CSV}, {BETS_CSV}, {PREDS_CSV}")


if __name__ == "__main__":
    main()

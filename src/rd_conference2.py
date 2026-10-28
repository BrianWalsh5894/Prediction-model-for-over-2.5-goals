"""
National League (Conference) - R&D round 2: tune the goals-only model and
fix its two weak spots.

Round 1 (rd_conference.py), best = C (Poisson + Dixon-Coles both on
League Two + NL): +0.0030 vs bookmaker average, +0.0061 vs Betfair close
(bar +0.002). Matches between established teams were almost level with the
bookmakers (+0.0008); the gap came from
  - teams relegated from League Two (+0.0115) and promoted from
    North/South (+0.0042): their goal form comes from a different level
  - rising scoring: the model kept predicting too few Overs (2025/26:
    actual 57%, bookmakers 55%, model 51%)

NEW FEATURES (all leak-free)
  level: for each team, the change of division since last season
         (+1 = came down a level, -1 = came up, e.g. from North/South), plus
         the same value faded out over the first 15 games of the season
  trend: the league's average goals and Over 2.5 rate over its previous
         300 matches (only matches on EARLIER dates), so the model sees the
         current scoring level, not the 2017-onwards average

HOW THIS ROUND STAYS HONEST
  - TUNING uses results only (log loss vs what happened) on the National
    League folds 2020/21 - 2023/24. No odds are used to tune.
  - The single best setting is LOCKED and run ONCE on 2024/25 - 2025/26,
    the only seasons with Betfair prices. Same bar, set before running:
    within +0.002 of the Betfair close.

TUNED (Poisson and Dixon-Coles both on League Two + NL, as round-1 C)
  feature set: base / +level / +trend / +level+trend
  Poisson alpha: 0.3, 0.6, 1.3335 (v3), 2.5, 5
  Dixon-Coles window 1/2/3 years, time decay 0 / 0.002, newcomer rating
     league average / weakest quarter
  blend weight on Poisson: 0.3 - 0.7
  calibration (fitted on earlier seasons only): none, Platt, shift
     (chosen on 2021/22 - 2023/24 - the first fold has nothing earlier)

Run from inside src/ (about 5 minutes):
    python rd_conference2.py
"""

import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import rd_conference as r1  # noqa: E402
from rd_experiments3 import NON_FEATURE_COLS  # noqa: E402
from rd_eu_upper_second2 import rolling_dc_v, poisson_preds  # noqa: E402
from build_features import build_feature_table  # noqa: E402
from rd_eu_upper3 import bet_sim  # noqa: E402

REPORT_MD = os.path.join("..", "reports", "rd_conference2.md")
GRID_CSV = os.path.join("..", "reports", "rd_conference2_grid.csv")
FINAL_CSV = os.path.join("..", "reports", "rd_conference2_final.csv")
BETS_CSV = os.path.join("..", "reports", "rd_conference2_bets.csv")
PREDS_CSV = os.path.join("..", "reports", "rd_conference2_predictions.csv")

NL, L2 = r1.NL, r1.L2
ALL_SEASONS = r1.ALL_SEASONS
FOLDS = r1.FOLDS                              # 2020/21 .. 2025/26
TUNE = ["2020/21", "2021/22", "2022/23", "2023/24"]
TEST = ["2024/25", "2025/26"]
BAR = 0.002
SEED = 424410
N_BOOT = 2000

LEVEL = {"Premier League": 1, "Championship": 2, "League One": 3, "League Two": 4, NL: 5}
FADE_GAMES = 15
TREND_WINDOW = 300

FEATURE_SETS = ["base", "+level", "+trend", "+level+trend"]
ALPHAS = [0.3, 0.6, 1.3335, 2.5, 5.0]
WINDOWS = [1, 2, 3]
XIS = [0.0, 0.002]
NEWCOMERS = ["avg", "weak"]
WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]
CALIBS = ["none", "Platt", "shift"]
ROUND1 = dict(fset="base", alpha=1.3335, window=3, xi=0.0, newcomer="avg", w=0.5)

ll, demargin = r1.ll, r1.demargin


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def level_features(combined, feats):
    """Change of division since last season, per team (leak-free: last
    season's division is known before this season starts)."""
    league_of = {}
    for (team, season), lg in pd.concat([
            combined[["home_team", "season", "league_name"]].rename(columns={"home_team": "team"}),
            combined[["away_team", "season", "league_name"]].rename(columns={"away_team": "team"})
    ]).groupby(["team", "season"])["league_name"].first().items():
        league_of[(team, season)] = lg
    prev = {s: ALL_SEASONS[i - 1] for i, s in enumerate(ALL_SEASONS) if i > 0}

    def change(team, season, league):
        p = prev.get(season)
        if p is None:
            return 0.0
        before = league_of.get((team, p))
        lvl_before = LEVEL[before] if before else LEVEL[league] + 1   # unseen = came up from below
        return float(LEVEL[league] - lvl_before)

    out = pd.DataFrame({"match_id": feats["match_id"]})
    for side in ("home", "away"):
        ch = [change(t, s, lg) for t, s, lg in zip(feats[f"{side}_team"], feats["season"], feats["league_name"])]
        out[f"{side}_level_change"] = ch
        fade = np.clip(1 - feats[f"{side}_season_match_num"].values / FADE_GAMES, 0, 1)
        out[f"{side}_level_change_faded"] = np.array(ch) * fade
    return out


def trend_features(combined):
    """League's average goals / Over 2.5 rate over its previous TREND_WINDOW
    matches played on EARLIER dates (same-day matches never see each other)."""
    rows = []
    for lg, g in combined.sort_values(["match_date", "match_id"]).groupby("league_name"):
        dates = g["match_date"].values
        goals = g["total_goals"].values.astype(float)
        over = g["over_2_5"].values.astype(float)
        cg, co = np.concatenate([[0], np.cumsum(goals)]), np.concatenate([[0], np.cumsum(over)])
        first_same_day = np.searchsorted(dates, dates, side="left")
        for mid, end in zip(g["match_id"].values, first_same_day):
            start = max(0, end - TREND_WINDOW)
            n = end - start
            if n < 50:
                rows.append((mid, np.nan, np.nan))
            else:
                rows.append((mid, (cg[end] - cg[start]) / n, (co[end] - co[start]) / n))
    return pd.DataFrame(rows, columns=["match_id", "league_goals_recent", "league_over25_recent"])


def calibrate(P, col, method):
    out = P[col].copy()
    if method == "none":
        return out
    for k, season in enumerate(FOLDS):
        cur = (P["season"] == season).values
        past = P[P["season"].isin(FOLDS[:k])]
        if k == 0 or len(past) < 300:
            continue
        x, y = logit(past[col].values), past["over_2_5"].values
        if method == "Platt":
            m = LogisticRegression(C=1e6, max_iter=1000).fit(x.reshape(-1, 1), y)
            out[cur] = m.predict_proba(logit(P.loc[cur, col].values).reshape(-1, 1))[:, 1]
        else:
            a = minimize_scalar(lambda a: ll(y, sigmoid(x + a)).sum(), bounds=(-1, 1), method="bounded").x
            out[cur] = sigmoid(logit(P.loc[cur, col].values) + a)
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    eng = pd.read_csv(r1.CLEAN_CSV, parse_dates=["match_date"])
    eng = eng[eng["season"].isin(ALL_SEASONS)]
    nl = r1.load_national_league()
    combined = pd.concat([eng[[c for c in nl.columns if c in eng.columns]], nl], ignore_index=True)
    goals_df = combined[["match_id", "home_goals", "away_goals"]]

    print(f"Building features ({len(combined)} matches)...")
    feats = build_feature_table(combined)
    feats = feats[feats["season"].isin(ALL_SEASONS)].reset_index(drop=True)
    base_cols = [c for c in feats.columns if c not in NON_FEATURE_COLS
                 and "shots" not in c and "sot_" not in c and not c.startswith("league_")]
    base_cols += ["league_National League"]
    lv = level_features(combined, feats)
    tr = trend_features(combined)
    feats = feats.merge(lv, on="match_id", how="left").merge(tr, on="match_id", how="left")
    level_cols = [c for c in lv.columns if c != "match_id"]
    trend_cols = ["league_goals_recent", "league_over25_recent"]
    fsets = {"base": base_cols, "+level": base_cols + level_cols, "+trend": base_cols + trend_cols,
             "+level+trend": base_cols + level_cols + trend_cols}

    base = feats[feats["season"].isin(FOLDS) & (feats["league_name"] == NL)][
        ["match_id", "season", "league_name", "over_2_5", "avg_over_2_5", "avg_under_2_5",
         "bfe_over_2_5", "bfe_under_2_5", "bfe_close_over_2_5", "bfe_close_under_2_5"]].copy()
    base["Bookmaker average"] = demargin(base["avg_over_2_5"], base["avg_under_2_5"])
    base["Betfair close"] = demargin(base["bfe_close_over_2_5"], base["bfe_close_under_2_5"])
    base["team_type"] = base["match_id"].map(r1.team_types(nl, eng))
    base = base.set_index("match_id")
    trainable = feats[feats["league_name"].isin([L2, NL])]

    print(f"Poisson: {len(FEATURE_SETS)} feature sets x {len(ALPHAS)} alphas x {len(FOLDS)} folds...")
    pois = {}
    for fs, alpha in itertools.product(FEATURE_SETS, ALPHAS):
        s = []
        for season in FOLDS:
            prev = ALL_SEASONS[:ALL_SEASONS.index(season)]
            trn = trainable[trainable["season"].isin(prev)]
            te = feats[(feats["season"] == season) & (feats["league_name"] == NL)]
            s.append(pd.Series(poisson_preds(trn, te, fsets[fs], goals_df, alpha), index=te["match_id"]))
        pois[(fs, alpha)] = pd.concat(s).reindex(base.index)
    print(f"  done ({(time.time() - t0) / 60:.1f} min)")

    print(f"Dixon-Coles on League Two + NL: {len(WINDOWS) * len(XIS) * len(NEWCOMERS)} settings...")
    l2nl = combined[combined["league_name"].isin([L2, NL])]
    dcs = {}
    for window, xi, nc in itertools.product(WINDOWS, XIS, NEWCOMERS):
        d = rolling_dc_v(l2nl, xi, window * 365, nc).set_index("match_id")["dc"]
        dcs[(window, xi, nc)] = d.reindex(base.index)
        print(f"  window {window}y, xi {xi}, newcomer {nc}  ({(time.time() - t0) / 60:.1f} min)")

    tune_mask = base["season"].isin(TUNE).values
    y_tune = base.loc[tune_mask, "over_2_5"].values
    grid = []
    for (fs, alpha), (window, xi, nc), w in itertools.product(pois, dcs, WEIGHTS):
        p_p, p_d = pois[(fs, alpha)], dcs[(window, xi, nc)]
        p = w * p_p + (1 - w) * p_d.fillna(p_p)
        grid.append({"fset": fs, "alpha": alpha, "window": window, "xi": xi, "newcomer": nc, "w": w,
                     "tune_ll": ll(y_tune, p.values[tune_mask]).mean()})
    grid = pd.DataFrame(grid).sort_values("tune_ll").reset_index(drop=True)
    grid.to_csv(GRID_CSV, index=False)
    best = grid.iloc[0].to_dict()
    q = " & ".join(f"(grid['{k}'] == {v!r})" for k, v in ROUND1.items())
    r1row = grid[eval(q)].iloc[0]
    keys = ["fset", "alpha", "window", "xi", "newcomer", "w"]
    print(f"\nGrid: {len(grid)} settings scored on {TUNE[0]} - {TUNE[-1]} (log loss vs results)")
    print(f"  round-1 C setting: {r1row['tune_ll']:.4f}")
    print(f"  best setting:      {best['tune_ll']:.4f}  {({k: best[k] for k in keys})}")
    print("\nTop 10 settings:")
    print(grid.head(10).round(4).to_string(index=False))
    print("\nBest tuning log loss for each option:")
    for k in keys:
        print(f"  {k:9}", "  ".join(f"{v}: {g['tune_ll'].min():.4f}" for v, g in grid.groupby(k)))

    def blend(cfg):
        p_p, p_d = pois[(cfg["fset"], cfg["alpha"])], dcs[(cfg["window"], cfg["xi"], cfg["newcomer"])]
        return cfg["w"] * p_p + (1 - cfg["w"]) * p_d.fillna(p_p)

    P = base.copy()
    P["blend_best"] = blend(best).values
    P["round1"] = blend(ROUND1).values
    Pr = P.reset_index()
    cal_mask = base["season"].isin(TUNE[1:]).values
    y_cal = base.loc[cal_mask, "over_2_5"].values
    cal_rows = []
    for c in CALIBS:
        P[f"cal_{c}"] = calibrate(Pr, "blend_best", c).values
        cal_rows.append({"calibration": c, "tune_ll_2021-24": ll(y_cal, P.loc[cal_mask, f"cal_{c}"].values).mean()})
    cal_rows = pd.DataFrame(cal_rows).sort_values("tune_ll_2021-24")
    best_cal = cal_rows.iloc[0]["calibration"]
    print("\nCalibration on top of the best blend (2021/22 - 2023/24):")
    print(cal_rows.round(4).to_string(index=False))
    P["FINAL"] = P[f"cal_{best_cal}"]
    locked = {**{k: best[k] for k in keys}, "calib": best_cal}
    print(f"\nLOCKED SETTING: {locked}")

    # ---- final test, run ONCE ----
    T = P[P["season"].isin(TEST)].reset_index()
    rows = []
    for ref in ["Bookmaker average", "Betfair close"]:
        g = T[T[ref].notna()]
        y = g["over_2_5"].values
        ll_ref = ll(y, g[ref].values)
        for name, label in [("FINAL", "round 2 (locked)"), ("round1", "round 1 C")]:
            d = ll(y, g[name].values) - ll_ref
            boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
            rows.append({"vs": ref, "model": label, "model_ll": ll(y, g[name].values).mean(),
                         "market_ll": ll_ref.mean(), "diff": d.mean(), "ci_low": np.percentile(boots, 2.5),
                         "ci_high": np.percentile(boots, 97.5), "n": len(g)})
    final = pd.DataFrame(rows)
    final.to_csv(FINAL_CSV, index=False)

    gb = T[T["Bookmaker average"].notna()]
    by_type = gb.groupby("team_type").apply(lambda s: pd.Series({
        "n": len(s),
        "round2_vs_book": (ll(s["over_2_5"].values, s["FINAL"].values) - ll(s["over_2_5"].values, s["Bookmaker average"].values)).mean(),
        "round1_vs_book": (ll(s["over_2_5"].values, s["round1"].values) - ll(s["over_2_5"].values, s["Bookmaker average"].values)).mean()}))
    per_season = gb.groupby("season").apply(lambda s: pd.Series({
        "over_rate": s["over_2_5"].mean(), "round2_avg_p": s["FINAL"].mean(), "round1_avg_p": s["round1"].mean(),
        "book_avg_p": s["Bookmaker average"].mean()}))

    Bm = T[T["Betfair close"].notna()]
    bets = []
    for name, label in [("FINAL", "round 2 (locked)"), ("round1", "round 1 C")]:
        for price, oc, uc in [("Betfair early", "bfe_over_2_5", "bfe_under_2_5"),
                              ("Betfair close", "bfe_close_over_2_5", "bfe_close_under_2_5")]:
            bets.append({"model": label, "price": price, **bet_sim(Bm, name, oc, uc, rng)})
    bets = pd.DataFrame(bets)
    bets.to_csv(BETS_CSV, index=False)
    P.reset_index().to_csv(PREDS_CSV, index=False)
    gap_pp = (Bm["FINAL"] - Bm["Betfair close"]).abs().mean() * 100
    corr = np.corrcoef(Bm["FINAL"], Bm["Betfair close"])[0, 1]

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    print(f"\n=== FINAL TEST {TEST[0]} - {TEST[-1]} (never used for tuning) - diff < 0 means model better ===")
    for ref in ["Bookmaker average", "Betfair close"]:
        print(f"\nvs {ref}:")
        print(final[final["vs"] == ref].drop(columns="vs").round(4).to_string(index=False))
    print("\nBy team type vs bookmaker average (final test seasons):")
    print(by_type.round(4).to_string())
    print("\nMean predicted vs actual Over rate (final test seasons):")
    print(per_season.round(3).to_string())
    print(f"\nRound 2 vs Betfair close: average gap {gap_pp:.1f} pp, correlation {corr:.2f}")
    print("\nBetting simulation - flat 1 unit, EV >= 3% after 2% commission:")
    print(bets.round(4).to_string(index=False))

    fc = final[final["vs"] == "Betfair close"].set_index("model")
    gap = fc.loc["round 2 (locked)", "diff"]
    passed = gap <= BAR
    verdict = "PASSES - candidate to freeze as v3-conf" if passed else "does NOT pass the bar"
    print(f"\nRound 2 (locked) vs Betfair close: gap {gap:+.4f}  (round 1: {fc.loc['round 1 C', 'diff']:+.4f}, "
          f"bar: +{BAR:.4f}) -> {verdict}")
    print(f"Total time {(time.time() - t0) / 60:.1f} min")

    lines = ["# National League - R&D round 2 (tuning + level/trend features)\n",
             f"Tuned on results only, {TUNE[0]} - {TUNE[-1]} ({len(grid)} settings + {len(CALIBS)} calibrations). "
             f"Locked setting tested once on {TEST[0]} - {TEST[-1]}. Bar: within +{BAR} of the Betfair close.\n",
             f"\n**Locked setting:** `{locked}`\n",
             f"\nTuning log loss: round-1 C {r1row['tune_ll']:.4f}, best {best['tune_ll']:.4f}.\n",
             "\n## Top 10 settings (tuning seasons)\n", r1.md_table(grid.head(10)) + "\n",
             "\n## Calibration (2021/22 - 2023/24)\n", r1.md_table(cal_rows) + "\n",
             "\n## Final test\n", r1.md_table(final) + "\n",
             "\n## By team type (final test)\n", r1.md_table(by_type.reset_index()) + "\n",
             "\n## Predicted vs actual Over rate (final test)\n", r1.md_table(per_season.reset_index()) + "\n",
             f"\nGap to Betfair close {gap_pp:.1f} pp, correlation {corr:.2f}.\n",
             "\n## Betting simulation\n", r1.md_table(bets) + "\n",
             f"\n**Round 2 vs Betfair close: {gap:+.4f} - {'passes' if passed else 'does not pass'} the bar.**\n"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_MD}, {GRID_CSV}, {FINAL_CSV}, {BETS_CSV}, {PREDS_CSV}")


if __name__ == "__main__":
    main()

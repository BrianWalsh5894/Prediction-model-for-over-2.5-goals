"""
v3-eu-upper-second - R&D round 2: tune the model for the second divisions
(La Liga 2, Serie B, 2. Bundesliga, Ligue 2).

Round 1 (rd_eu_upper_second.py) used v3's England settings unchanged. Best
was B (Poisson on both divisions + Dixon-Coles per country): +0.0093 vs
the Betfair close, bar +0.002. Diagnostics showed the model predicting too
low in 2. Bundesliga / Ligue 2, too spread out (overconfident), and much
worse on teams promoted from the third tiers (no history).

HOW THIS ROUND STAYS HONEST
  - TUNING uses results only (log loss against what actually happened),
    on the walk-forward folds 2020/21 - 2023/24. No odds are used to tune.
  - The single best setup is then LOCKED and run ONCE on the final test
    seasons 2024/25 - 2025/26, which were never used for tuning. Only
    there is it compared with the bookmaker average and the Betfair close.
  - Same bar as round 1, set before running: within +0.002 of the
    Betfair close on 2024/25 + 2025/26.

WHAT IS TUNED
  1. Poisson strength alpha: 0.3, 0.6, 1.3335 (v3), 2.5, 5, 10
  2. Poisson training data: second divisions only, or both divisions
  3. Dixon-Coles history window: 1, 2 or 3 years (v3 = 3)
  4. Dixon-Coles time decay xi: 0 (v3) or 0.002 per day (half-life ~1 yr)
  5. Starting rating for a team with no history (promoted from the third
     tier): league average (v3) or the average of the weakest quarter
  6. Blend weight on Poisson: 0.3, 0.4, 0.5 (v3), 0.6, 0.7
  7. Calibration on top of the best blend, each fitted only on earlier
     seasons' out-of-sample predictions: none, Platt (all leagues),
     per-league shift, per-league Platt. Chosen on 2021/22 - 2023/24
     (the first fold has nothing earlier to fit on).

Run from inside src/ (about 5-10 minutes - Dixon-Coles is re-fitted weekly
for 12 settings x 4 countries):
    python rd_eu_upper_second2.py
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
import rd_experiments3 as r3  # noqa: E402
import rd_eu_upper2 as r2  # noqa: E402
from rd_eu_upper3 import bet_sim  # noqa: E402
from sklearn.linear_model import PoissonRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPORT_MD = os.path.join("..", "reports", "rd_eu_upper_second2.md")
GRID_CSV = os.path.join("..", "reports", "rd_eu_upper_second2_grid.csv")
FINAL_CSV = os.path.join("..", "reports", "rd_eu_upper_second2_final.csv")
BETS_CSV = os.path.join("..", "reports", "rd_eu_upper_second2_bets.csv")
PREDS_CSV = os.path.join("..", "reports", "rd_eu_upper_second2_predictions.csv")

COUNTRIES = r2.COUNTRIES
SECOND = r2.SECOND
TOP_OF = {v: k for k, v in COUNTRIES.items()}
ALL_SEASONS = r2.ALL_SEASONS
FOLDS = ALL_SEASONS[r2.FOLD_START:]                 # 2020/21 .. 2025/26
TUNE = ["2020/21", "2021/22", "2022/23", "2023/24"]
TEST = ["2024/25", "2025/26"]
BAR = 0.002
SEED = 424410
N_BOOT = 2000

ALPHAS = [0.3, 0.6, 1.3335, 2.5, 5.0, 10.0]
TRAIN_SETS = ["second", "both"]
WINDOWS = [1, 2, 3]
XIS = [0.0, 0.002]
NEWCOMERS = ["avg", "weak"]
WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]
CALIBS = ["none", "Platt (all leagues)", "shift (per league)", "Platt (per league)"]

ROUND1 = dict(alpha=1.3335, train="both", window=3, xi=0.0, newcomer="avg", w=0.5, calib="none")


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# Rolling Dixon-Coles with a configurable window and newcomer rating
# (same method as rd_experiments3.rolling_dc; only these two options added)
# ---------------------------------------------------------------------------
def rolling_dc_v(matches, xi, window_days, newcomer):
    m = matches.sort_values("match_date").reset_index(drop=True)
    m["week_start"] = (m["match_date"] - pd.to_timedelta(m["match_date"].dt.weekday, unit="D")).dt.normalize()
    out, prev_params, prev_home = [], {}, 0.0
    for week in sorted(m["week_start"].unique()):
        week = pd.Timestamp(week)
        hist = m[(m["match_date"] < week) & (m["match_date"] >= week - pd.Timedelta(days=window_days))]
        target = m[m["week_start"] == week]
        if len(hist) < r3.MIN_MATCHES_TO_FIT:
            continue
        teams = sorted(set(hist["home_team"]) | set(hist["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        w = np.exp(-xi * (week - hist["match_date"]).dt.days.values.astype(float))
        x0 = np.zeros(2 * n + 1)
        for t, i in idx.items():
            if t in prev_params:
                x0[i], x0[n + i] = prev_params[t]
        x0[-1] = prev_home
        params, rho = r3.fit_dc_weighted(hist["home_team"].map(idx).values, hist["away_team"].map(idx).values,
                                         hist["home_goals"].values.astype(float),
                                         hist["away_goals"].values.astype(float), w, n, x0)
        att, dfn, home_adv = params[:n], params[n:2 * n], params[-1]
        prev_params = {t: (att[i], dfn[i]) for t, i in idx.items()}
        prev_home = home_adv
        last_league = (pd.concat([
            hist[["match_date", "home_team", "league_name"]].rename(columns={"home_team": "team"}),
            hist[["match_date", "away_team", "league_name"]].rename(columns={"away_team": "team"})])
            .sort_values("match_date").groupby("team")["league_name"].last())
        new_rating = {}
        for lg, grp in last_league.groupby(last_league):
            ii = np.array([idx[t] for t in grp.index])
            if newcomer == "weak":
                strength = att[ii] - dfn[ii]                      # higher = stronger
                k = max(1, len(ii) // 4)
                ii = ii[np.argsort(strength)[:k]]
            new_rating[lg] = (att[ii].mean(), dfn[ii].mean())
        for r in target.itertuples(index=False):
            ha, hd = (att[idx[r.home_team]], dfn[idx[r.home_team]]) if r.home_team in idx \
                else new_rating.get(r.league_name, (0.0, 0.0))
            aa, ad = (att[idx[r.away_team]], dfn[idx[r.away_team]]) if r.away_team in idx \
                else new_rating.get(r.league_name, (0.0, 0.0))
            lam_h, lam_a = float(np.exp(ha + ad + home_adv)), float(np.exp(aa + hd))
            out.append({"match_id": r.match_id, "dc": float(r3.dc_p_over(lam_h, lam_a, rho))})
    return pd.DataFrame(out)


def poisson_preds(train, test, cols, goals_df, alpha):
    med = train[cols].median(numeric_only=True)
    sc = StandardScaler()
    Xtr = sc.fit_transform(train[cols].fillna(med))
    Xte = sc.transform(test[cols].fillna(med))
    tg = train[["match_id"]].merge(goals_df, on="match_id", how="left")
    lam = (PoissonRegressor(alpha=alpha, max_iter=1000).fit(Xtr, tg["home_goals"]).predict(Xte)
           + PoissonRegressor(alpha=alpha, max_iter=1000).fit(Xtr, tg["away_goals"]).predict(Xte))
    return 1 - r3.poisson_dist.cdf(2, lam)


def calibrate(P, col, method):
    """Calibrate column `col` fold by fold, each fold fitted on earlier folds only."""
    out = P[col].copy()
    if method == "none":
        return out
    for k, season in enumerate(FOLDS):
        cur = P["season"] == season
        past = P[P["season"].isin(FOLDS[:k])]
        if k == 0 or len(past) < 300:
            continue                                   # nothing earlier to fit on - left as is
        groups = [(None, cur, past)] if method == "Platt (all leagues)" else \
            [(lg, cur & (P["league_name"] == lg), past[past["league_name"] == lg]) for lg in SECOND]
        for _, mask, pst in groups:
            if not mask.any() or len(pst) < 100:
                continue
            x, y = logit(pst[col].values), pst["over_2_5"].values
            if method.startswith("Platt"):
                m = LogisticRegression(C=1e6, max_iter=1000).fit(x.reshape(-1, 1), y)
                out[mask] = m.predict_proba(logit(P.loc[mask, col].values).reshape(-1, 1))[:, 1]
            else:
                a = minimize_scalar(lambda a: ll(y, sigmoid(x + a)).sum(), bounds=(-1, 1), method="bounded").x
                out[mask] = sigmoid(logit(P.loc[mask, col].values) + a)
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    clean = pd.read_csv(r2.CLEAN_CSV, parse_dates=["match_date"])
    clean = clean[clean["season"].isin(ALL_SEASONS)]
    goals_df = clean[["match_id", "home_goals", "away_goals"]]
    feats = pd.read_csv(r2.FEATURES_CSV, parse_dates=["match_date"])
    feats = feats[feats["season"].isin(ALL_SEASONS)].reset_index(drop=True)
    cols = [c for c in feats.columns if c not in r3.NON_FEATURE_COLS]

    # ---- base table: every second-division fold match + market prices ----
    base = feats[feats["season"].isin(FOLDS) & feats["league_name"].isin(SECOND)][
        ["match_id", "season", "league_name", "over_2_5", "avg_over_2_5", "avg_under_2_5",
         "bfe_over_2_5", "bfe_under_2_5", "bfe_close_over_2_5", "bfe_close_under_2_5"]].copy()
    base["Bookmaker average"] = r2.demargin(base["avg_over_2_5"], base["avg_under_2_5"])
    base["Betfair close"] = r2.demargin(base["bfe_close_over_2_5"], base["bfe_close_under_2_5"])
    base = base.set_index("match_id")

    # ---- Poisson predictions for every alpha x training set ----
    print(f"Poisson: {len(ALPHAS)} alphas x {len(TRAIN_SETS)} training sets x {len(FOLDS)} folds...")
    pois = {}
    for alpha, ts in itertools.product(ALPHAS, TRAIN_SETS):
        s = []
        for i, season in enumerate(FOLDS):
            prev = ALL_SEASONS[:ALL_SEASONS.index(season)]
            tr = feats[feats["season"].isin(prev)]
            if ts == "second":
                tr = tr[tr["league_name"].isin(SECOND)]
            te = feats[(feats["season"] == season) & feats["league_name"].isin(SECOND)]
            s.append(pd.Series(poisson_preds(tr, te, cols, goals_df, alpha), index=te["match_id"]))
        pois[(alpha, ts)] = pd.concat(s).reindex(base.index)
    print(f"  done ({(time.time() - t0) / 60:.1f} min)")

    # ---- Dixon-Coles for every window x decay x newcomer (per country) ----
    print(f"Dixon-Coles: {len(WINDOWS) * len(XIS) * len(NEWCOMERS)} settings x 4 countries (slow part)...")
    dcs = {}
    for window, xi, nc in itertools.product(WINDOWS, XIS, NEWCOMERS):
        d = pd.concat([rolling_dc_v(clean[clean["league_name"].isin([TOP_OF[lg], lg])], xi, window * 365, nc)
                       for lg in SECOND]).set_index("match_id")["dc"]
        dcs[(window, xi, nc)] = d.reindex(base.index)
        print(f"  window {window}y, xi {xi}, newcomer {nc}  ({(time.time() - t0) / 60:.1f} min)")

    # ---- grid on the TUNING seasons (results only) ----
    tune_mask = base["season"].isin(TUNE).values
    y_tune = base.loc[tune_mask, "over_2_5"].values
    grid = []
    for (alpha, ts), (window, xi, nc), w in itertools.product(pois, dcs, WEIGHTS):
        p_pois, p_dc = pois[(alpha, ts)], dcs[(window, xi, nc)]
        p = w * p_pois + (1 - w) * p_dc.fillna(p_pois)
        grid.append({"alpha": alpha, "train": ts, "window": window, "xi": xi, "newcomer": nc, "w": w,
                     "tune_ll": ll(y_tune, p.values[tune_mask]).mean()})
    grid = pd.DataFrame(grid).sort_values("tune_ll").reset_index(drop=True)
    grid.to_csv(GRID_CSV, index=False)
    best = grid.iloc[0].to_dict()
    r1 = grid[(grid["alpha"] == 1.3335) & (grid["train"] == "both") & (grid["window"] == 3) & (grid["xi"] == 0.0)
              & (grid["newcomer"] == "avg") & (grid["w"] == 0.5)].iloc[0]
    print(f"\nGrid: {len(grid)} settings scored on tuning seasons {TUNE[0]} - {TUNE[-1]} (log loss vs results)")
    print(f"  round-1 setting: {r1['tune_ll']:.4f}")
    print(f"  best setting:    {best['tune_ll']:.4f}  {({k: best[k] for k in ['alpha', 'train', 'window', 'xi', 'newcomer', 'w']})}")
    print("\nTop 10 settings:")
    print(grid.head(10).round(4).to_string(index=False))

    # one-at-a-time view: best tuning log loss for each value of each setting
    print("\nBest tuning log loss for each option (shows which settings matter):")
    for k in ["alpha", "train", "window", "xi", "newcomer", "w"]:
        print(f"  {k:9}", "  ".join(f"{v}: {g['tune_ll'].min():.4f}" for v, g in grid.groupby(k)))

    # ---- calibration on top of the best blend (choose on 2021/22 - 2023/24) ----
    def blend(cfg):
        p_pois, p_dc = pois[(cfg["alpha"], cfg["train"])], dcs[(cfg["window"], cfg["xi"], cfg["newcomer"])]
        return cfg["w"] * p_pois + (1 - cfg["w"]) * p_dc.fillna(p_pois)

    P = base.copy()
    P["blend_best"] = blend(best).values
    P["round1"] = blend(ROUND1).values
    cal_mask = base["season"].isin(TUNE[1:]).values
    y_cal = base.loc[cal_mask, "over_2_5"].values
    cal_rows = []
    for c in CALIBS:
        P[f"cal_{c}"] = calibrate(P.reset_index(), "blend_best", c).values
        cal_rows.append({"calibration": c, "tune_ll_2021-24": ll(y_cal, P.loc[cal_mask, f"cal_{c}"].values).mean()})
    cal_rows = pd.DataFrame(cal_rows).sort_values("tune_ll_2021-24")
    best_cal = cal_rows.iloc[0]["calibration"]
    print("\nCalibration on top of the best blend (tuning seasons 2021/22 - 2023/24):")
    print(cal_rows.round(4).to_string(index=False))
    P["FINAL"] = P[f"cal_{best_cal}"]
    locked = {**{k: best[k] for k in ["alpha", "train", "window", "xi", "newcomer", "w"]}, "calib": best_cal}
    print(f"\nLOCKED SETTING: {locked}")

    # ---- final test: run ONCE on 2024/25 - 2025/26 ----
    T = P[P["season"].isin(TEST)].reset_index()
    rows = []
    for league in ["ALL"] + SECOND:
        g0 = T if league == "ALL" else T[T["league_name"] == league]
        for ref in ["Bookmaker average", "Betfair close"]:
            g = g0[g0[ref].notna()]
            y = g["over_2_5"].values
            ll_ref = ll(y, g[ref].values)
            for name in ["FINAL", "round1"]:
                d = ll(y, g[name].values) - ll_ref
                boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
                rows.append({"league": league, "vs": ref, "model": "round 2 (locked)" if name == "FINAL" else "round 1 B",
                             "model_ll": ll(y, g[name].values).mean(), "market_ll": ll_ref.mean(), "diff": d.mean(),
                             "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5), "n": len(g)})
    final = pd.DataFrame(rows)
    final.to_csv(FINAL_CSV, index=False)

    Bm = T[T["Betfair close"].notna()]
    bets = []
    for name, label in [("FINAL", "round 2 (locked)"), ("round1", "round 1 B")]:
        for league in ["ALL"] + SECOND:
            g = Bm if league == "ALL" else Bm[Bm["league_name"] == league]
            for price, oc, uc in [("Betfair early", "bfe_over_2_5", "bfe_under_2_5"),
                                  ("Betfair close", "bfe_close_over_2_5", "bfe_close_under_2_5")]:
                bets.append({"model": label, "league": league, "price": price, **bet_sim(g, name, oc, uc, rng)})
    bets = pd.DataFrame(bets)
    bets.to_csv(BETS_CSV, index=False)
    P.reset_index().to_csv(PREDS_CSV, index=False)

    bias = T.groupby("league_name").agg(actual=("over_2_5", "mean"), round2=("FINAL", "mean"),
                                        round1=("round1", "mean"), market=("Betfair close", "mean"))
    spread = {"round 2": T["FINAL"].std(), "round 1": T["round1"].std(), "Betfair close": Bm["Betfair close"].std()}
    gap_pp = (Bm["FINAL"] - Bm["Betfair close"]).abs().mean() * 100
    corr = np.corrcoef(Bm["FINAL"], Bm["Betfair close"])[0, 1]

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    print(f"\n=== FINAL TEST {TEST[0]} - {TEST[-1]} (never used for tuning) - diff < 0 means model better ===")
    for ref in ["Bookmaker average", "Betfair close"]:
        print(f"\nvs {ref}:")
        print(final[final["vs"] == ref].drop(columns="vs").round(4).to_string(index=False))
    print("\nBias check (mean predicted vs actual Over rate, final test seasons):")
    print(bias.round(3).to_string())
    print("Spread (std of probabilities): " + ", ".join(f"{k} {v:.3f}" for k, v in spread.items()))
    print(f"Round 2 vs Betfair close: average gap {gap_pp:.1f} pp, correlation {corr:.2f}")
    print("\nBetting simulation - flat 1 unit, EV >= 3% after 2% commission:")
    print(bets.round(4).to_string(index=False))

    fc = final[(final["league"] == "ALL") & (final["vs"] == "Betfair close")].set_index("model")
    gap = fc.loc["round 2 (locked)", "diff"]
    passed = gap <= BAR
    verdict = "PASSES - candidate to freeze as v3-eu-upper-second" if passed else "does NOT pass the bar"
    print(f"\nRound 2 (locked) vs Betfair close: gap {gap:+.4f}  (round 1: {fc.loc['round 1 B', 'diff']:+.4f}, "
          f"bar: +{BAR:.4f}) -> {verdict}")
    print(f"Total time {(time.time() - t0) / 60:.1f} min")

    lines = ["# v3-eu-upper-second - R&D round 2 (tuning)\n",
             f"Tuned on results only, {TUNE[0]} - {TUNE[-1]} ({len(grid)} settings + {len(CALIBS)} calibrations). "
             f"Locked setting tested once on {TEST[0]} - {TEST[-1]}. Bar: within +{BAR} of the Betfair close.\n",
             f"\n**Locked setting:** `{locked}`\n",
             f"\nTuning log loss: round-1 setting {r1['tune_ll']:.4f}, best {best['tune_ll']:.4f}.\n",
             "\n## Top 10 settings (tuning seasons)\n", r2.md_table(grid.head(10)) + "\n",
             "\n## Calibration (2021/22 - 2023/24)\n", r2.md_table(cal_rows) + "\n",
             "\n## Final test\n", r2.md_table(final) + "\n",
             "\n## Bias check\n", r2.md_table(bias.reset_index()) + "\n",
             f"\nSpread: " + ", ".join(f"{k} {v:.3f}" for k, v in spread.items()) +
             f". Gap to Betfair close {gap_pp:.1f} pp, correlation {corr:.2f}.\n",
             "\n## Betting simulation\n", r2.md_table(bets) + "\n",
             f"\n**Round 2 vs Betfair close: {gap:+.4f} - {'passes' if passed else 'does not pass'} the bar.**\n"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_MD}, {GRID_CSV}, {FINAL_CSV}, {BETS_CSV}, {PREDS_CSV}")


if __name__ == "__main__":
    main()

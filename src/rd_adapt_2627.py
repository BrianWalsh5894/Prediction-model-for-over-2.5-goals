"""
Does a faster-adapting model beat the frozen live models on 2026/27?

Why: in the first 33 live predictions the models rated Over below the
market 82% of the time (avg -4 points). Scoring is up this season in most
leagues (Championship, Serie A, Bundesliga, Ligue 1, La Liga, National
League) and the frozen models adapt slowly: the Poisson half is frozen on
2017/18-2025/26 and Dixon-Coles looks back 3 years with no time decay.

2026/27 has never been used to tune anything, so it is a clean test.
Every variant below is fixed BEFORE looking at 2026/27 - nothing is tuned:

  BASE  the frozen model exactly as tracked live (Poisson re-trained from
        the frozen code - its fingerprint must match the manifest - plus
        Dixon-Coles re-fitted each week, 3-year window, no decay)
  A     BASE but Dixon-Coles with time decay xi = 0.002/day (half-life
        about a year - the value tested in earlier R&D rounds)
  B     BASE but Dixon-Coles on the last 1 year only
  C     BASE + a per-league level shift: before each week, the logit shift
        that best fits BASE's own predictions vs results on that league's
        previous 200 matches (out-of-sample predictions only) is added

Groups: lower (League One + League Two, v3), upper (Premier League +
Championship, v3), eu_upper (top 4 European leagues, v3-eu-upper).
Judged on every finished 2026/27 match: log loss vs results, vs the
bookmaker average, vs Betfair close where available, and the average
predicted Over rate vs the actual rate. Also checks BASE against the
probabilities actually logged live (should be very close).

A winner would become a NEW version with a new record; the frozen models
keep running. Run from inside src/ (about 5-10 minutes):
    python rd_adapt_2627.py
"""

import hashlib
import json
import os
import sqlite3
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import model_v3 as mv  # noqa: E402
import model_v3_eu_upper as me  # noqa: E402
from build_features import build_feature_table  # noqa: E402

DB_PATH = os.path.join("..", "data", "over25.db")
REPORT_MD = os.path.join("..", "reports", "rd_adapt_2627.md")
SUMMARY_CSV = os.path.join("..", "reports", "rd_adapt_2627_summary.csv")
PREDS_CSV = os.path.join("..", "reports", "rd_adapt_2627_predictions.csv")

TEST_SEASON = "2026/27"
PREV_SEASON = "2025/26"
SHIFT_WINDOW = 200
VARIANTS = {"BASE": (0.0, 3 * 365), "A": (0.002, 3 * 365), "B": (0.0, 365)}   # Dixon-Coles (xi, window days)
MODELS = ["BASE", "A", "B", "C"]
SEED = 424410
N_BOOT = 2000
GROUPS = {"lower": ["League One", "League Two"], "upper": ["Premier League", "Championship"],
          "eu_upper": me.TOP}


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def demargin(o, u):
    return (1 / o) / (1 / o + 1 / u)


def fingerprint(bundle):
    parts = [bundle["m_home"].coef_, bundle["m_away"].coef_, [bundle["m_home"].intercept_, bundle["m_away"].intercept_]]
    return hashlib.sha256(np.concatenate([np.ravel(p) for p in parts]).round(10).tobytes()).hexdigest()[:16]


def weekly_dc(matches, seasons, dc_groups, xi, window):
    """Dixon-Coles re-fitted at the start of each week (model_v3's own code),
    predicting that week's matches. dc_groups: list of league lists fitted
    separately (England: one group; Europe: one per country)."""
    mv.DC_XI, mv.DC_WINDOW_DAYS = xi, window
    m = matches.sort_values("match_date").copy()
    m["week"] = (m["match_date"] - pd.to_timedelta(m["match_date"].dt.weekday, unit="D")).dt.normalize()
    out = []
    for leagues in dc_groups:
        sub = m[m["league_name"].isin(leagues)]
        prev = None
        for week in sorted(sub.loc[sub["season"].isin(seasons), "week"].unique()):
            fit = mv.fit_dc_asof(sub, week, prev)
            if fit is None:
                continue
            prev = fit
            tgt = sub[(sub["week"] == week) & sub["season"].isin(seasons)]
            _, _, p = mv.predict_dc(fit, tgt)
            out.append(pd.DataFrame({"match_id": tgt["match_id"].values, "p_dc": p}))
    mv.DC_XI, mv.DC_WINDOW_DAYS = 0.0, 3 * 365
    return pd.concat(out).set_index("match_id")["p_dc"]


def rolling_shift(P):
    """Variant C: per league, shift fitted on that league's previous
    SHIFT_WINDOW matches (earlier weeks only) of BASE vs results."""
    P = P.sort_values(["match_date", "match_id"]).copy()
    P["week"] = (P["match_date"] - pd.to_timedelta(P["match_date"].dt.weekday, unit="D")).dt.normalize()
    P["C"] = np.nan
    for lg, g in P.groupby("league_name"):
        for week in sorted(g.loc[g["season"] == TEST_SEASON, "week"].unique()):
            hist = g[g["week"] < week].tail(SHIFT_WINDOW)
            cur = g.index[(g["week"] == week) & (g["season"] == TEST_SEASON)]
            if len(hist) < 50:
                P.loc[cur, "C"] = P.loc[cur, "BASE"]
                continue
            x, y = logit(hist["BASE"].values), hist["over_2_5"].values
            a = minimize_scalar(lambda a: ll(y, 1 / (1 + np.exp(-(x + a)))).sum(), bounds=(-1, 1), method="bounded").x
            P.loc[cur, "C"] = 1 / (1 + np.exp(-(logit(P.loc[cur, "BASE"].values) + a)))
    return P


def build_group_predictions(kind):
    if kind == "england":
        clean = pd.read_csv(os.path.join("..", "data", "clean", "matches_clean.csv"), parse_dates=["match_date"])
        feats = build_feature_table(clean)
        train_rows = feats["season"].isin(mv.TRAIN_SEASONS)
        manifest = os.path.join("..", "models", "manifest_v3.json")
        dc_groups = [GROUPS["lower"] + GROUPS["upper"]]
        cols = mv.feature_columns(feats[train_rows])
        top_mask = pd.Series(True, index=feats.index)
    else:
        clean = pd.read_csv(me.CLEAN_CSV, parse_dates=["match_date"])
        feats = me.prepare_features(clean)
        train_rows = feats["season"].isin(me.TRAIN_SEASONS)
        manifest = me.MANIFEST_PATH
        dc_groups = [[t, s] for t, s in me.COUNTRIES.items()]
        cols = me.feature_columns(feats[train_rows])
        top_mask = feats["league_name"].isin(me.TOP)
    goals_df = clean[["match_id", "home_goals", "away_goals"]]

    frozen = mv.train_poisson(feats[train_rows & top_mask], goals_df, cols)
    fp = fingerprint(frozen)
    with open(manifest, encoding="utf-8") as f:
        expected = json.load(f)["poisson_fingerprint"]
    print(f"  re-trained Poisson fingerprint {fp} (frozen: {expected}) -> {'MATCH' if fp == expected else 'DIFFERENT'}")
    if fp != expected:
        sys.exit("STOP: could not reproduce the frozen model exactly.")
    prev_bundle = mv.train_poisson(feats[feats["season"].isin(mv.TRAIN_SEASONS[:-1]) & top_mask], goals_df, cols)

    test = feats[feats["season"].isin([PREV_SEASON, TEST_SEASON]) & top_mask & feats["over_2_5"].notna()].copy()
    p_pois = np.empty(len(test))
    is_test = (test["season"] == TEST_SEASON).values
    p_pois[is_test] = mv.predict_poisson(frozen, test[is_test])[2]
    p_pois[~is_test] = mv.predict_poisson(prev_bundle, test[~is_test])[2]      # out-of-sample for 2025/26
    test["p_poisson"] = p_pois

    for name, (xi, window) in VARIANTS.items():
        dc = weekly_dc(clean, [PREV_SEASON, TEST_SEASON], dc_groups, xi, window)
        d = test["match_id"].map(dc)
        test[name] = 0.5 * test["p_poisson"] + 0.5 * d.fillna(test["p_poisson"])
        print(f"  Dixon-Coles {name} (xi {xi}, window {window // 365}y) done")
    return test


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    parts = []
    for kind in ["england", "europe"]:
        print(f"\n{kind}: rebuilding the frozen model and variants...")
        parts.append(build_group_predictions(kind))
        print(f"  ({(time.time() - t0) / 60:.1f} min)")
    P = pd.concat(parts, ignore_index=True)
    P = rolling_shift(P.reset_index(drop=True))
    P["group"] = P["league_name"].map({lg: g for g, lgs in GROUPS.items() for lg in lgs})
    P["Bookmaker average"] = demargin(P["avg_over_2_5"], P["avg_under_2_5"])
    P["Betfair close"] = demargin(P["bfe_close_over_2_5"], P["bfe_close_under_2_5"])
    T = P[P["season"] == TEST_SEASON].copy()
    T.to_csv(PREDS_CSV, index=False)

    # ---- BASE vs what was logged live ----
    with sqlite3.connect(DB_PATH) as conn:
        live = pd.read_sql_query("SELECT match_id, predicted_prob_over_2_5 AS live_p FROM predictions "
                                 "WHERE model_version IN ('v3', 'v3-eu-upper')", conn)
    chk = T.merge(live, on="match_id")
    if len(chk):
        print(f"\nBASE vs live-logged predictions: {len(chk)} matches, mean |difference| "
              f"{(chk['BASE'] - chk['live_p']).abs().mean() * 100:.2f} points (Dixon-Coles here is fitted at the "
              f"start of the week, live on the prediction day)")

    # ---- results ----
    rows = []
    for grp in ["ALL"] + list(GROUPS):
        g0 = T if grp == "ALL" else T[T["group"] == grp]
        y = g0["over_2_5"].values
        row = {"group": grp, "matches": len(g0), "actual_over": g0["over_2_5"].mean()}
        for m in MODELS:
            row[f"{m}_avg_p"] = g0[m].mean()
            row[f"{m}_ll"] = ll(y, g0[m].values).mean()
        for ref in ["Bookmaker average", "Betfair close"]:
            g = g0[g0[ref].notna()]
            row[f"n_{ref}"] = len(g)
            row[f"{ref}_avg_p"] = g[ref].mean() if len(g) else np.nan
            row[f"{ref}_ll"] = ll(g["over_2_5"].values, g[ref].values).mean() if len(g) else np.nan
            for m in MODELS:
                row[f"{m}_vs_{ref}"] = (ll(g["over_2_5"].values, g[m].values) - ll(g["over_2_5"].values, g[ref].values)).mean() if len(g) else np.nan
        for m in ["A", "B", "C"]:
            d = ll(y, g0[m].values) - ll(y, g0["BASE"].values)
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
            row[f"{m}_minus_BASE"] = d.mean()
            row[f"{m}_minus_BASE_ci"] = f"{np.percentile(boots, 2.5):+.4f} to {np.percentile(boots, 97.5):+.4f}"
        rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(SUMMARY_CSV, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== 2026/27 so far - average predicted P(Over) vs actual Over rate ===")
    print(S[["group", "matches", "actual_over"] + [f"{m}_avg_p" for m in MODELS] + ["Bookmaker average_avg_p"]].round(3).to_string(index=False))
    print("\n=== Log loss vs results (lower = better) ===")
    print(S[["group", "matches"] + [f"{m}_ll" for m in MODELS] + ["Bookmaker average_ll"]].round(4).to_string(index=False))
    print("\n=== Change vs the frozen model (BASE), same matches; < 0 = better ===")
    for m in ["A", "B", "C"]:
        print(f"  {m}: " + " | ".join(f"{r['group']} {r[f'{m}_minus_BASE']:+.4f} ({r[f'{m}_minus_BASE_ci']})" for _, r in S.iterrows()))
    print("\n=== vs bookmaker average (diff < 0 = model better) ===")
    print(S[["group", "n_Bookmaker average"] + [f"{m}_vs_Bookmaker average" for m in MODELS]].round(4).to_string(index=False))
    print("\n=== vs Betfair close ===")
    print(S[["group", "n_Betfair close"] + [f"{m}_vs_Betfair close" for m in MODELS]].round(4).to_string(index=False))
    best = min(["A", "B", "C"], key=lambda m: S.loc[S["group"] == "ALL", f"{m}_minus_BASE"].iloc[0])
    print(f"\nBest variant overall: {best} ({S.loc[S['group'] == 'ALL', f'{best}_minus_BASE'].iloc[0]:+.4f} vs frozen, "
          f"interval {S.loc[S['group'] == 'ALL', f'{best}_minus_BASE_ci'].iloc[0]})")
    print(f"Total time {(time.time() - t0) / 60:.1f} min")

    def md(df):
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        df = df.astype(str)
        return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
                         + ["| " + " | ".join(r) + " |" for r in df.itertuples(index=False, name=None)])
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Faster-adapting variants vs the frozen live models - 2026/27 so far\n\n"
                "Variants fixed before testing: A = Dixon-Coles time decay 0.002/day; B = Dixon-Coles 1-year window; "
                "C = frozen + per-league shift from the previous 200 matches. 2026/27 never used for tuning.\n\n"
                + md(S) + f"\n\n**Best variant overall: {best}.**\n")
    print(f"\nWrote {REPORT_MD}, {SUMMARY_CSV}, {PREDS_CSV}")


if __name__ == "__main__":
    main()

"""
v3-eu-upper - improvement round 2: calibration, plus "how far off are we"
in betting terms.

Starts from round 1's best (D: second divisions + xG, rd_eu_upper2.py) and
tests whether its probabilities are systematically off and can be corrected:
  D          as is
  D + Platt (all leagues)     one logistic correction a + b*logit(p)
  D + Platt (per league)      a separate correction for each league
  D + shift (per league)      only moves each league up/down (slope fixed)
Every correction for season N is fitted ONLY on the model's own
out-of-sample predictions for seasons before N (the earlier walk-forward
folds) - so the first fold (2020/21) can't be corrected and all methods are
compared on 2021/22 - 2025/26.

"How far off": for 2024/25 + 2025/26 (Betfair prices available) it also
simulates the actual betting rule - flat 1 unit when EV >= 3% after 2%
commission - at the Betfair early price and at the Betfair closing price,
with ROI and a 95% bootstrap interval.

Run from inside src/ (a few minutes - it rebuilds round 1's predictions):
    python rd_eu_upper3.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import rd_eu_upper2 as r2  # noqa: E402

REPORT_MD = os.path.join("..", "reports", "rd_eu_upper3.md")
SUMMARY_CSV = os.path.join("..", "reports", "rd_eu_upper3_summary.csv")
COMMISSION, MIN_EDGE = 0.02, 0.03
SEED = 424410
N_BOOT = 2000

BASE = r2.D
RAW, PLATT, PLATT_LG, SHIFT_LG = "D as is", "D + Platt (all leagues)", "D + Platt (per league)", "D + shift (per league)"
METHODS = [RAW, PLATT, PLATT_LG, SHIFT_LG]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_platt(p, y):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return lambda q: m.predict_proba(logit(q).reshape(-1, 1))[:, 1]


def fit_shift(p, y):
    x = logit(p)
    a = minimize_scalar(lambda a: r2.pm_ll(y, sigmoid(x + a)).sum(), bounds=(-1, 1), method="bounded").x
    return lambda q: sigmoid(logit(q) + a)


def bet_sim(df, pcol, over_col, under_col, rng):
    """Flat 1-unit bets when EV >= MIN_EDGE at the given prices."""
    d = df.dropna(subset=[over_col, under_col])
    p = d[pcol].values
    o, u = d[over_col].values, d[under_col].values
    ev_o = p * (o - 1) * (1 - COMMISSION) - (1 - p)
    ev_u = (1 - p) * (u - 1) * (1 - COMMISSION) - p
    over_side = ev_o >= ev_u
    ev = np.where(over_side, ev_o, ev_u)
    odds = np.where(over_side, o, u)
    won = np.where(over_side, d["over_2_5"].values == 1, d["over_2_5"].values == 0)
    take = ev >= MIN_EDGE
    pl = np.where(won, (odds - 1) * (1 - COMMISSION), -1.0)[take]
    n = int(take.sum())
    if n == 0:
        return {"bets": 0}
    boots = [pl[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    return {"bets": n, "share_of_matches": n / len(d), "avg_model_EV": float(ev[take].mean()),
            "profit_units": float(pl.sum()), "ROI": float(pl.mean()),
            "ROI_ci_low": float(np.percentile(boots, 2.5)), "ROI_ci_high": float(np.percentile(boots, 97.5))}


def main():
    rng = np.random.default_rng(SEED)
    P = r2.build_predictions()
    P = P[P["season"].isin(r2.ALL_SEASONS)].copy()
    folds = sorted(P["season"].unique())

    # --- calibrate each fold on earlier folds only ----------------------------
    for m in METHODS:
        P[m] = np.nan
    P[RAW] = P[BASE]
    for k, season in enumerate(folds):
        cur = P["season"] == season
        if k == 0:
            for m in (PLATT, PLATT_LG, SHIFT_LG):
                P.loc[cur, m] = P.loc[cur, BASE]
            continue
        past = P[P["season"].isin(folds[:k])]
        f = fit_platt(past[BASE].values, past["over_2_5"].values)
        P.loc[cur, PLATT] = f(P.loc[cur, BASE].values)
        for lg in r2.TOP:
            pl_, cl_ = past[past["league_name"] == lg], cur & (P["league_name"] == lg)
            if len(pl_) < 100 or not cl_.any():
                P.loc[cl_, PLATT_LG] = P.loc[cl_, BASE]
                P.loc[cl_, SHIFT_LG] = P.loc[cl_, BASE]
                continue
            P.loc[cl_, PLATT_LG] = fit_platt(pl_[BASE].values, pl_["over_2_5"].values)(P.loc[cl_, BASE].values)
            P.loc[cl_, SHIFT_LG] = fit_shift(pl_[BASE].values, pl_["over_2_5"].values)(P.loc[cl_, BASE].values)

    # --- is D biased? mean predicted vs actual, per league (all folds) --------
    bias = P.groupby("league_name").agg(matches=("over_2_5", "size"), actual_over_rate=("over_2_5", "mean"),
                                        model_avg=(BASE, "mean"), bookmaker_avg=("Bookmaker average", "mean"))
    bias["model_minus_actual"] = bias["model_avg"] - bias["actual_over_rate"]

    # --- log loss: methods vs market on folds with calibration -----------------
    rows = []
    E = P[P["season"].isin(folds[1:])]
    for ref, seasons in [("Bookmaker average", None), ("Betfair close", r2.BETFAIR_SEASONS)]:
        g = E if seasons is None else E[E["season"].isin(seasons)]
        g = g[g[ref].notna()]
        y = g["over_2_5"].values
        ll_ref = r2.pm_ll(y, g[ref].values)
        for m in METHODS:
            d = r2.pm_ll(y, g[m].values) - ll_ref
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
            rows.append({"vs": ref, "method": m, "model_ll": r2.pm_ll(y, g[m].values).mean(),
                         "market_ll": ll_ref.mean(), "diff": d.mean(),
                         "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5), "n": len(g)})
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    # --- how far off, in plain terms -------------------------------------------
    B = P[P["season"].isin(r2.BETFAIR_SEASONS) & P["Betfair close"].notna()]
    gap_pp = (B[RAW] - B["Betfair close"]).abs().mean() * 100
    corr = np.corrcoef(B[RAW], B["Betfair close"])[0, 1]
    bets = []
    for m in METHODS:
        for label, oc, uc in [("Betfair early price", "bfe_over_2_5", "bfe_under_2_5"),
                              ("Betfair closing price", "bfe_close_over_2_5", "bfe_close_under_2_5")]:
            bets.append({"method": m, "price": label, **bet_sim(B, m, oc, uc, rng)})
    bets = pd.DataFrame(bets)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\nIs D biased? (all 6 folds)")
    print(bias.round(4).to_string())
    print("\nLog loss, 2021/22-2025/26 (calibration needs an earlier fold) - diff < 0 means model better:")
    print(summary.round(4).to_string(index=False))
    print(f"\nHOW FAR OFF (2024/25 + 2025/26, {len(B)} matches):")
    print(f"  average gap between D's probability and Betfair close: {gap_pp:.1f} percentage points")
    print(f"  correlation with Betfair close: {corr:.2f}")
    print("\nBetting simulation - flat 1 unit, EV >= 3% after 2% commission:")
    print(bets.round(4).to_string(index=False))

    best = summary[summary["vs"] == "Betfair close"].sort_values("diff").iloc[0]
    print(f"\nBest vs Betfair close: {best['method']}  gap {best['diff']:+.4f}  (bar +0.0020) -> "
          f"{'PASSES' if best['diff'] <= 0.002 else 'does NOT pass the bar'}")

    def md(df):
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        df = df.astype(str)
        return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
                         + ["| " + " | ".join(r) + " |" for r in df.itertuples(index=False, name=None)])

    lines = ["# v3-eu-upper improvement round 2 - calibration + how far off\n",
             "## Bias check (all folds)\n", md(bias.reset_index()) + "\n",
             "\n## Log loss (2021/22 - 2025/26)\n", md(summary) + "\n",
             f"\n## How far off (2024/25 + 2025/26, {len(B)} matches)\n",
             f"Average gap to Betfair close: {gap_pp:.1f} percentage points; correlation {corr:.2f}.\n",
             md(bets) + "\n",
             f"\n**Best vs Betfair close: {best['method']}, gap {best['diff']:+.4f}.**\n"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_MD}, {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

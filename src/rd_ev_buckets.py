"""
EV bucket test - is a bigger claimed edge a better bet, or model error?

Plan agreed 28 Oct 2026 (claude/over25_ev_bucket_test_2026-10-28.txt).
DIAGNOSTIC ONLY: the 2024/25 + 2025/26 test seasons are already spent, and
nothing here changes the live 3% rule or any tracked model.

For every group it replays the paper-betting rule on the backtest
(flat 1 unit on the better side when EV >= 3% after 2% commission), at the
Betfair EARLY price (how Brian mostly bets) and at the Betfair CLOSE, and
splits the bets by:
  1. claimed EV bucket      3-6%, 6-10%, 10-15%, 15%+
  2. odds band              under 2.0, 2.0-3.0, 3.0+
  3. side                   Over / Under   (+ side x EV bucket)
  4. cumulative thresholds  EV >= 3, 5, 8, 10, 15%
  5. caps                   3-6%, 3-8%, 3-10%

Per cell: bets, wins, avg odds, avg claimed EV, ROI with 95% bootstrap CI,
and two lower-noise checks against the Betfair close:
  value_at_close  EV of the bet if the de-margined Betfair close is the true
                  probability (positive = the bet beat the closing line)
  model_gap_pp    model prob of the side minus Betfair close prob (points)
  market_move_pp  (early-price bets) close prob minus early prob of the side;
                  positive = the market moved towards the model
Cells with fewer than 50 bets are flagged "small".

Trend test per group: slope of per-bet profit on claimed EV (and of
value_at_close on claimed EV), with bootstrap CIs. Negative slope = the
bigger the claimed edge, the worse the bet.

Groups (the same predictions as the R&D rounds that produced them):
  England lower / upper  v3 recipe walk-forward (50/50 Poisson + Dixon-Coles,
                         league_breakdown.py loop), 2024/25-2025/26
  eu_upper               v3-eu-upper recipe, variant D (rd_eu_upper2.py)
  eu_second              rd_eu_upper_second2.py locked model (FINAL)
  conference             rd_conference3.py C+SOT+xG
Plus, separately: the FROZEN live models on 2026/27 so far
(rd_adapt_2627_predictions.csv, column BASE) - small, but never used to
pick anything.

England and eu_upper predictions are rebuilt (a few minutes) and cached in
reports/rd_ev_buckets_predictions.csv; add --rebuild to force a rebuild.

Run from inside src/:
    python rd_ev_buckets.py
Writes reports/rd_ev_buckets.md and reports/rd_ev_buckets_cells.csv
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

REPORTS = os.path.join("..", "reports")
CACHE_CSV = os.path.join(REPORTS, "rd_ev_buckets_predictions.csv")
CELLS_CSV = os.path.join(REPORTS, "rd_ev_buckets_cells.csv")
REPORT_MD = os.path.join(REPORTS, "rd_ev_buckets.md")
CONF_CSV = os.path.join(REPORTS, "rd_conference3_predictions.csv")
SECOND_CSV = os.path.join(REPORTS, "rd_eu_upper_second2_predictions.csv")
LIVE_CSV = os.path.join(REPORTS, "rd_adapt_2627_predictions.csv")

COMMISSION, MIN_EDGE = 0.02, 0.03
TEST = ["2024/25", "2025/26"]
SEED = 424410
N_BOOT = 2000
SMALL = 50

EV_BUCKETS = [(0.03, 0.06, "3-6%"), (0.06, 0.10, "6-10%"), (0.10, 0.15, "10-15%"), (0.15, np.inf, "15%+")]
ODDS_BANDS = [(0, 2.0, "under 2.0"), (2.0, 3.0, "2.0-3.0"), (3.0, np.inf, "3.0+")]
THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.15]
CAPS = [(0.03, 0.06), (0.03, 0.08), (0.03, 0.10)]
PRICES = [("early", "bfe_over_2_5", "bfe_under_2_5"), ("close", "bfe_close_over_2_5", "bfe_close_under_2_5")]
KEEP = ["match_id", "season", "league_name", "over_2_5", "p",
        "bfe_over_2_5", "bfe_under_2_5", "bfe_close_over_2_5", "bfe_close_under_2_5"]


# ---------------------------------------------------------------------------
# Predictions per group
# ---------------------------------------------------------------------------

def england_predictions():
    """v3 recipe walk-forward - the league_breakdown.py loop, 50/50 blend only."""
    from rd_experiments3 import (rolling_dc, poisson_fit_predict, NON_FEATURE_COLS,
                                 ALL_SEASONS_CHRONOLOGICAL as SEASONS, WALK_FORWARD_MIN_TRAIN_SEASONS)
    df = pd.read_csv(os.path.join("..", "data", "clean", "features.csv"), parse_dates=["match_date"])
    df = df[df["season"].isin(SEASONS)].reset_index(drop=True)
    clean = pd.read_csv(os.path.join("..", "data", "clean", "matches_clean.csv"), parse_dates=["match_date"])
    clean = clean[clean["season"].isin(SEASONS)]
    goals = clean[["match_id", "home_goals", "away_goals"]]
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print("England: rolling Dixon-Coles (xi=0)...")
    dfx = df.merge(rolling_dc(clean, 0.0), on="match_id", how="left")
    out = []
    for i in range(WALK_FORWARD_MIN_TRAIN_SEASONS, len(SEASONS)):
        season = SEASONS[i]
        if season not in TEST:
            continue
        train = dfx[dfx["season"].isin(SEASONS[:i])]
        test = dfx[dfx["season"] == season].copy()
        p_pois = poisson_fit_predict(train, test, cols, goals)
        p_dc = test["dc_p_over"].fillna(pd.Series(p_pois, index=test.index)).values
        test["p"] = 0.5 * p_pois + 0.5 * p_dc
        out.append(test)
        print(f"  England predicted {season}")
    P = pd.concat(out, ignore_index=True)[KEEP]
    P["group"] = np.where(P["league_name"].isin(["League One", "League Two"]), "England lower (L1+L2)",
                          "England upper (PL+Champ)")
    return P


def eu_upper_predictions():
    import rd_eu_upper2 as r2
    P = r2.build_predictions()
    P = P[P["season"].isin(TEST)].rename(columns={r2.D: "p"})[KEEP].copy()
    P["group"] = "eu_upper (top 4 Europe)"
    return P


def saved_predictions():
    parts = []
    if os.path.exists(SECOND_CSV):
        s = pd.read_csv(SECOND_CSV).rename(columns={"FINAL": "p"})
        s = s[s["season"].isin(TEST)][KEEP].copy()
        s["group"] = "eu_second (Europe 2nd div)"
        parts.append(s)
    else:
        print(f"  (skipping eu_second - {SECOND_CSV} not found)")
    if os.path.exists(CONF_CSV):
        c = pd.read_csv(CONF_CSV).rename(columns={"C+SOT+xG": "p"})
        c["league_name"] = "National League"
        c = c[c["season"].isin(TEST)][KEEP].copy()
        c["group"] = "conference (National League)"
        parts.append(c)
    else:
        print(f"  (skipping conference - {CONF_CSV} not found)")
    return parts


def load_backtest(rebuild):
    if os.path.exists(CACHE_CSV) and not rebuild:
        print(f"Using cached England + eu_upper predictions ({CACHE_CSV}); --rebuild to redo them.")
        built = pd.read_csv(CACHE_CSV)
    else:
        built = pd.concat([england_predictions(), eu_upper_predictions()], ignore_index=True)
        built.to_csv(CACHE_CSV, index=False)
    return pd.concat([built] + saved_predictions(), ignore_index=True)


def load_live():
    if not os.path.exists(LIVE_CSV):
        return None
    L = pd.read_csv(LIVE_CSV).rename(columns={"BASE": "p"})
    L = L[KEEP + ["group"]].copy()
    names = {"lower": "LIVE 2026/27 lower", "upper": "LIVE 2026/27 upper", "eu_upper": "LIVE 2026/27 eu_upper"}
    L["group"] = L["group"].map(names)
    return L


# ---------------------------------------------------------------------------
# Betting replay
# ---------------------------------------------------------------------------

def demargin(o, u):
    io, iu = 1 / np.asarray(o, float), 1 / np.asarray(u, float)
    return io / (io + iu)


def place_bets(df, oc, uc):
    """One row per bet (EV >= MIN_EDGE on the better side) at the given prices."""
    d = df.dropna(subset=[oc, uc, "p"]).copy()
    p, o, u = d["p"].values, d[oc].values, d[uc].values
    ev_o = p * (o - 1) * (1 - COMMISSION) - (1 - p)
    ev_u = (1 - p) * (u - 1) * (1 - COMMISSION) - p
    over = ev_o >= ev_u
    d["side"] = np.where(over, "Over", "Under")
    d["ev"] = np.where(over, ev_o, ev_u)
    d["odds"] = np.where(over, o, u)
    d["p_side"] = np.where(over, p, 1 - p)
    d = d[d["ev"] >= MIN_EDGE].copy()
    is_over = d["side"] == "Over"
    won = np.where(is_over, d["over_2_5"] == 1, d["over_2_5"] == 0)
    d["won"] = won.astype(int)
    d["profit"] = np.where(won, (d["odds"] - 1) * (1 - COMMISSION), -1.0)
    # Betfair close as the reference "true" probability
    pc_over = demargin(d["bfe_close_over_2_5"], d["bfe_close_under_2_5"])
    d["p_close_side"] = np.where(is_over, pc_over, 1 - pc_over)
    d["value_at_close"] = d["p_close_side"] * (d["odds"] - 1) * (1 - COMMISSION) - (1 - d["p_close_side"])
    d["model_gap_pp"] = (d["p_side"] - d["p_close_side"]) * 100
    pe_over = demargin(d["bfe_over_2_5"], d["bfe_under_2_5"])
    d["market_move_pp"] = (d["p_close_side"] - np.where(is_over, pe_over, 1 - pe_over)) * 100
    return d


def cell(b, rng, **labels):
    n = len(b)
    row = dict(labels, bets=n)
    if n == 0:
        return row
    pl = b["profit"].values
    boots = np.array([pl[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    vac = b["value_at_close"].dropna().values
    row.update(wins=int(b["won"].sum()), strike=b["won"].mean(), avg_odds=b["odds"].mean(),
               avg_claimed_ev=b["ev"].mean(), roi=pl.mean(),
               roi_ci_low=np.percentile(boots, 2.5), roi_ci_high=np.percentile(boots, 97.5),
               profit=pl.sum(),
               value_at_close=vac.mean() if len(vac) else np.nan,
               model_gap_pp=b["model_gap_pp"].mean(),
               market_move_pp=b["market_move_pp"].mean(),
               small=n < SMALL)
    return row


def trend(b, rng):
    """Slope of profit (and of value_at_close) per +1 point of claimed EV, bootstrap CIs."""
    out = {}
    for col in ["profit", "value_at_close"]:
        d = b[["ev", col]].dropna()
        if len(d) < 30:
            out[col] = (np.nan, np.nan, np.nan)
            continue
        x, y = d["ev"].values * 100, d[col].values
        slope = np.polyfit(x, y, 1)[0]
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(x), len(x))
            bs.append(np.polyfit(x[i], y[i], 1)[0])
        out[col] = (slope, np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    return out


def analyse(data, rng):
    rows, trends = [], []
    for grp, g in data.groupby("group", sort=False):
        for price, oc, uc in PRICES:
            b = place_bets(g, oc, uc)
            base = dict(group=grp, price=price)
            rows.append(cell(b, rng, **base, split="all", cell="EV >= 3% (live rule)"))
            for lo, hi, name in EV_BUCKETS:
                rows.append(cell(b[(b["ev"] >= lo) & (b["ev"] < hi)], rng, **base, split="EV bucket", cell=name))
            for lo, hi, name in ODDS_BANDS:
                rows.append(cell(b[(b["odds"] >= lo) & (b["odds"] < hi)], rng, **base, split="odds band", cell=name))
            for side in ["Over", "Under"]:
                s = b[b["side"] == side]
                rows.append(cell(s, rng, **base, split="side", cell=side))
                for lo, hi, name in EV_BUCKETS:
                    rows.append(cell(s[(s["ev"] >= lo) & (s["ev"] < hi)], rng, **base,
                                     split="side x EV", cell=f"{side} {name}"))
            for t in THRESHOLDS:
                rows.append(cell(b[b["ev"] >= t], rng, **base, split="threshold", cell=f"EV >= {t:.0%}"))
            for lo, hi in CAPS:
                rows.append(cell(b[(b["ev"] >= lo) & (b["ev"] < hi)], rng, **base,
                                 split="cap", cell=f"cap {lo:.0%}-{hi:.0%}"))
            t = trend(b, rng)
            trends.append(dict(base, bets=len(b),
                               profit_slope=t["profit"][0], profit_slope_lo=t["profit"][1],
                               profit_slope_hi=t["profit"][2],
                               vac_slope=t["value_at_close"][0], vac_slope_lo=t["value_at_close"][1],
                               vac_slope_hi=t["value_at_close"][2]))
    return pd.DataFrame(rows), pd.DataFrame(trends)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(x, signed=True):
    if pd.isna(x):
        return ""
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def fmt_cells(df):
    t = pd.DataFrame({
        "cell": df["cell"], "bets": df["bets"].astype(int),
        "wins": df["wins"].fillna(0).astype(int),
        "avg_odds": df["avg_odds"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
        "claimed_EV": df["avg_claimed_ev"].map(pct),
        "ROI": df["roi"].map(pct),
        "ROI_95%": [("" if pd.isna(a) else f"{a:+.0%} to {b:+.0%}") for a, b in zip(df["roi_ci_low"], df["roi_ci_high"])],
        "value_at_close": df["value_at_close"].map(pct),
        "model_gap_pp": df["model_gap_pp"].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}"),
        "mkt_move_pp": df["market_move_pp"].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}"),
        "": df["small"].map(lambda s: "small" if s is True else ""),
    })
    head = "| " + " | ".join(t.columns) + " |"
    sep = "|" + "---|" * len(t.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in t.values])


def slope_txt(s, lo, hi):
    if pd.isna(s):
        return "n/a"
    verdict = "falls" if hi < 0 else ("rises" if lo > 0 else "no clear trend")
    return f"{s * 100:+.2f} pts per +1pt EV ({lo * 100:+.2f} to {hi * 100:+.2f}) -> {verdict}"


def write_report(cells, trends, live_cells, live_trends, n_matches):
    L = ["# EV bucket test: is a bigger claimed edge real, or model error?", "",
         "DIAGNOSTIC ONLY. Backtest seasons 2024/25 + 2025/26 (already spent). Flat 1 unit, better side, "
         "EV >= 3% after 2% commission, at the Betfair early price and at the Betfair close. "
         "Nothing here changes the live 3% rule.", "",
         "- `value_at_close`: EV of the bet if the de-margined Betfair close is the true probability "
         "(much less noisy than ROI; positive = beat the closing line).",
         "- `model_gap_pp`: model probability of the side minus Betfair close probability.",
         "- `mkt_move_pp`: close minus early probability of the side; positive = market moved towards the model "
         "(only meaningful at the early price).",
         f"- `small` = fewer than {SMALL} bets - too small to read.", "",
         "Matches per group: " + ", ".join(f"{k} {v}" for k, v in n_matches.items()), "",
         "## 1. Trend: does the bet get worse as the claimed edge gets bigger?", "",
         "Slope of per-bet result on claimed EV (in points), 95% bootstrap interval.", "",
         "| group | price | bets | profit vs claimed EV | value at close vs claimed EV |", "|---|---|---|---|---|"]
    for r in trends.itertuples(index=False):
        vac = "n/a at the close" if r.price == "close" else slope_txt(r.vac_slope, r.vac_slope_lo, r.vac_slope_hi)
        L.append(f"| {r.group} | {r.price} | {r.bets} | {slope_txt(r.profit_slope, r.profit_slope_lo, r.profit_slope_hi)} "
                 f"| {vac} |")
    L.append("")
    # Decision rules agreed 28 Oct, before running: a change only counts if it holds in more than one
    # of England / Europe / National League.
    fam = {"England": "England", "eu_": "Europe", "conference": "National League"}
    t = trends[~trends["group"].str.startswith("ALL")].copy()
    t["family"] = t["group"].map(lambda g: next(v for k, v in fam.items() if g.startswith(k)))
    falls = sorted(t.loc[t["profit_slope_hi"] < 0, "family"].unique())
    rises = sorted(t.loc[t["profit_slope_lo"] > 0, "family"].unique())
    if len(falls) >= 2:
        verdict = f"ROI FALLS as claimed EV rises in {', '.join(falls)} -> a cap is justified (report it alongside 3%)."
    elif len(rises) >= 2:
        verdict = f"ROI RISES with claimed EV in {', '.join(rises)} -> consider a higher threshold (alongside 3%)."
    else:
        verdict = ("No pattern that holds in more than one of England / Europe / National League "
                   f"(falls: {', '.join(falls) or 'none'}; rises: {', '.join(rises) or 'none'}) -> keep 3%, no cap.")
    L += ["**Verdict under the rules agreed before running:** " + verdict, ""]
    L.append("## 2. Summary: ROI by claimed EV bucket (early price)")
    L.append("")
    piv = cells[(cells["price"] == "early") & (cells["split"].isin(["all", "EV bucket"]))]
    t = piv.pivot_table(index="group", columns="cell", values="roi", sort=False)
    n = piv.pivot_table(index="group", columns="cell", values="bets", sort=False)
    v = piv.pivot_table(index="group", columns="cell", values="value_at_close", sort=False)
    order = ["EV >= 3% (live rule)"] + [b[2] for b in EV_BUCKETS]
    L.append("| group | " + " | ".join(order) + " |")
    L.append("|---|" + "---|" * len(order))
    for g in t.index:
        L.append(f"| {g} | " + " | ".join(
            f"{pct(t.loc[g, c])} (n={int(n.loc[g, c])}, vac {pct(v.loc[g, c])})" if c in t.columns and not pd.isna(t.loc[g, c])
            else "-" for c in order) + " |")
    L.append("")
    for grp in cells["group"].unique():
        L += [f"## {grp}", ""]
        for price in ["early", "close"]:
            L += [f"### {price} price", ""]
            L.append(fmt_cells(cells[(cells["group"] == grp) & (cells["price"] == price)]))
            L.append("")
    if live_cells is not None:
        L += ["## FROZEN live models, 2026/27 so far (out of sample, small)", "",
              "| group | price | bets | profit vs claimed EV | value at close vs claimed EV |", "|---|---|---|---|---|"]
        for r in live_trends.itertuples(index=False):
            vac = "n/a at the close" if r.price == "close" else slope_txt(r.vac_slope, r.vac_slope_lo, r.vac_slope_hi)
            L.append(f"| {r.group} | {r.price} | {r.bets} | {slope_txt(r.profit_slope, r.profit_slope_lo, r.profit_slope_hi)} "
                     f"| {vac} |")
        L.append("")
        for grp in live_cells["group"].unique():
            L += [f"### {grp} (early price)", ""]
            sub = live_cells[(live_cells["group"] == grp) & (live_cells["price"] == "early")
                             & live_cells["split"].isin(["all", "EV bucket", "side", "cap"])]
            L.append(fmt_cells(sub))
            L.append("")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild England + eu_upper predictions")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    data = load_backtest(args.rebuild)
    n_matches = data.groupby("group", sort=False).size().to_dict()
    cells, trends = analyse(data, rng)
    pool = data.copy()
    pool["group"] = "ALL backtest groups pooled"
    pc, pt = analyse(pool, rng)
    cells, trends = pd.concat([cells, pc], ignore_index=True), pd.concat([trends, pt], ignore_index=True)

    live = load_live()
    live_cells = live_trends = None
    if live is not None:
        live_cells, live_trends = analyse(live, rng)
        cells = pd.concat([cells, live_cells], ignore_index=True)
    cells.to_csv(CELLS_CSV, index=False)
    write_report(cells[~cells["group"].str.startswith("LIVE")], trends, live_cells, live_trends, n_matches)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    show = ["group", "price", "cell", "bets", "avg_odds", "avg_claimed_ev", "roi", "roi_ci_low", "roi_ci_high",
            "value_at_close", "model_gap_pp", "market_move_pp"]
    print("\nROI and value at close by claimed EV bucket:")
    s = cells[cells["split"].isin(["all", "EV bucket"])][show]
    print(s.round(3).to_string(index=False))
    print("\nOver vs Under:")
    print(cells[cells["split"] == "side"][show].round(3).to_string(index=False))
    print("\nCaps:")
    print(cells[cells["split"] == "cap"][show].round(3).to_string(index=False))
    print("\nTrend (slope per +1 point of claimed EV):")
    allt = trends if live_trends is None else pd.concat([trends, live_trends], ignore_index=True)
    print(allt.round(4).to_string(index=False))
    print(f"\nWrote {REPORT_MD} and {CELLS_CSV}")


if __name__ == "__main__":
    main()

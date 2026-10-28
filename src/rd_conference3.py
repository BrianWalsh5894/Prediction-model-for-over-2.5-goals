"""
National League - R&D round 3: round 1 C + MORE DATA (no tuning).

Brian's decision (28 Oct): keep round 1 C (goals-only v3 recipe, Poisson +
Dixon-Coles both on League Two + NL, v3 settings) and add data instead of
tuning. New data from Bzzoiro (data/raw/conference/bzzoiro_stats_nl_l2.csv):
  - National League shots on target, counted from Bzzoiro shotmaps, from
    2023/24 (matches football-data's shots on target closely; Bzzoiro's
    TOTAL shots run ~3 higher than football-data's, so total shots are not
    used)
  - xG: National League (provider xG) from 2023/24, League Two (Bzzoiro's
    estimate) from Nov 2022. Similar scales (2.87 vs 2.74 per match).
League Two shots on target come from football-data for every season, so
the model learns what shots on target are worth mostly from League Two and
applies it to the National League.

Three variants, fixed before running (nothing tuned):
  C          round 1 C exactly (check: +0.0061 vs Betfair close)
  C+SOT      + shots-on-target form features (same build_feature_table code)
  C+SOT+xG   + rolling xG for/against (last 5 / 10, shifted - leak-free)
Missing values (e.g. NL before 2023/24) are filled with the training median,
as everywhere else.

Judged on the final test seasons 2024/25 + 2025/26 against the bookmaker
average and the Betfair close, same +0.002 bar, plus the betting simulation.
NOTE: third look at these test seasons for the National League - read a
narrow pass with caution.

Run from inside src/ (about a minute):
    python rd_conference3.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import rd_conference as r1  # noqa: E402
from rd_experiments3 import rolling_dc, poisson_fit_predict, NON_FEATURE_COLS  # noqa: E402
from build_features import build_feature_table  # noqa: E402
from rd_eu_upper3 import bet_sim  # noqa: E402

BZ_CSV = os.path.join("..", "data", "raw", "conference", "bzzoiro_stats_nl_l2.csv")
REPORT_MD = os.path.join("..", "reports", "rd_conference3.md")
SUMMARY_CSV = os.path.join("..", "reports", "rd_conference3_summary.csv")
BETS_CSV = os.path.join("..", "reports", "rd_conference3_bets.csv")
PREDS_CSV = os.path.join("..", "reports", "rd_conference3_predictions.csv")

NL, L2 = r1.NL, r1.L2
ALL_SEASONS, FOLDS = r1.ALL_SEASONS, r1.FOLDS
TEST = ["2024/25", "2025/26"]
BAR = 0.002
SEED = 424410
N_BOOT = 2000
V_C, V_SOT, V_XG = "C (round 1)", "C+SOT", "C+SOT+xG"
MODELS = [V_C, V_SOT, V_XG]
ll, demargin = r1.ll, r1.demargin


def xg_features(combined):
    """Rolling xG for/against per team (last 5 and 10 matches with xG),
    shifted so a match never sees its own xG."""
    home = pd.DataFrame({"match_id": combined["match_id"], "match_date": combined["match_date"],
                         "team": combined["home_team"], "is_home": 1,
                         "xg_for": combined["home_xg"], "xg_against": combined["away_xg"]})
    away = pd.DataFrame({"match_id": combined["match_id"], "match_date": combined["match_date"],
                         "team": combined["away_team"], "is_home": 0,
                         "xg_for": combined["away_xg"], "xg_against": combined["home_xg"]})
    long = pd.concat([home, away]).sort_values(["team", "match_date", "match_id"]).reset_index(drop=True)
    for col in ["xg_for", "xg_against"]:
        sh = long.groupby("team")[col].shift(1)
        for w in (5, 10):
            long[f"{col}_roll{w}"] = sh.groupby(long["team"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
    feat_cols = [c for c in long.columns if "_roll" in c]
    h = long[long["is_home"] == 1].set_index("match_id")[feat_cols].add_prefix("home_")
    a = long[long["is_home"] == 0].set_index("match_id")[feat_cols].add_prefix("away_")
    out = h.join(a)
    out["xg_match_expected"] = (out["home_xg_for_roll10"] + out["away_xg_against_roll10"]
                                + out["away_xg_for_roll10"] + out["home_xg_against_roll10"]) / 2
    return out.reset_index()


def main():
    rng = np.random.default_rng(SEED)
    eng = pd.read_csv(r1.CLEAN_CSV, parse_dates=["match_date"])
    eng = eng[eng["season"].isin(ALL_SEASONS)]
    nl = r1.load_national_league()
    combined = pd.concat([eng[[c for c in nl.columns if c in eng.columns]], nl], ignore_index=True)

    bz = pd.read_csv(BZ_CSV)
    combined = combined.merge(bz[["match_id", "home_xg", "away_xg", "home_sot", "away_sot"]], on="match_id", how="left")
    is_nl = combined["league_name"] == NL
    combined.loc[is_nl, "home_shots_on_target"] = combined.loc[is_nl, "home_sot"]
    combined.loc[is_nl, "away_shots_on_target"] = combined.loc[is_nl, "away_sot"]
    cover = combined[is_nl].groupby("season")["home_shots_on_target"].apply(lambda s: s.notna().mean())
    print("National League shots-on-target coverage by season (from Bzzoiro):")
    print(cover.round(3).to_string())
    goals_df = combined[["match_id", "home_goals", "away_goals"]]

    print(f"\nBuilding features ({len(combined)} matches)...")
    feats = build_feature_table(combined.drop(columns=["home_xg", "away_xg", "home_sot", "away_sot"]))
    feats = feats[feats["season"].isin(ALL_SEASONS)].reset_index(drop=True)
    xgf = xg_features(combined)
    feats = feats.merge(xgf, on="match_id", how="left")

    base = [c for c in feats.columns if c not in NON_FEATURE_COLS and "shots" not in c and "sot_" not in c
            and not c.startswith("league_") and "xg" not in c] + ["league_National League"]
    sot_cols = [c for c in feats.columns if "sot_" in c]
    xg_cols = [c for c in xgf.columns if c != "match_id"]
    fsets = {V_C: base, V_SOT: base + sot_cols, V_XG: base + sot_cols + xg_cols}
    print(f"features: C {len(base)}, +SOT {len(sot_cols)}, +xG {len(xg_cols)}")

    print("\nDixon-Coles (League Two + NL linked, as round 1 C)...")
    dc = rolling_dc(combined[combined["league_name"].isin([L2, NL])], 0.0)[["match_id", "dc_p_over"]]
    fa = feats.merge(dc, on="match_id", how="left")

    preds = []
    for season in FOLDS:
        prev = ALL_SEASONS[:ALL_SEASONS.index(season)]
        tr = fa[fa["season"].isin(prev) & fa["league_name"].isin([L2, NL])]
        te = fa[(fa["season"] == season) & (fa["league_name"] == NL)]
        out = te[["match_id", "season", "over_2_5", "avg_over_2_5", "avg_under_2_5", "bfe_over_2_5",
                  "bfe_under_2_5", "bfe_close_over_2_5", "bfe_close_under_2_5"]].set_index("match_id").copy()
        d = te.set_index("match_id")["dc_p_over"]
        for v, cols in fsets.items():
            usable = [c for c in cols if tr[c].notna().any()]   # e.g. no xG at all before 2022/23
            p = pd.Series(poisson_fit_predict(tr, te, usable, goals_df), index=te["match_id"])
            out[v] = 0.5 * p + 0.5 * d.fillna(p)
        preds.append(out.reset_index())
        print(f"  predicted {season}")
    P = pd.concat(preds, ignore_index=True)
    P["Bookmaker average"] = demargin(P["avg_over_2_5"], P["avg_under_2_5"])
    P["Betfair close"] = demargin(P["bfe_close_over_2_5"], P["bfe_close_under_2_5"])
    P["team_type"] = P["match_id"].map(r1.team_types(nl, eng))
    P.to_csv(PREDS_CSV, index=False)

    rows = []
    for ref in ["Bookmaker average", "Betfair close"]:
        g = P[P["season"].isin(TEST) & P[ref].notna()]
        y = g["over_2_5"].values
        ll_ref = ll(y, g[ref].values)
        for m in MODELS:
            d = ll(y, g[m].values) - ll_ref
            boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
            rows.append({"vs": ref, "model": m, "model_ll": ll(y, g[m].values).mean(), "market_ll": ll_ref.mean(),
                         "diff": d.mean(), "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                         "n": len(g)})
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    gb = P[P["season"].isin(TEST) & P["Bookmaker average"].notna()]
    by_type = gb.groupby("team_type").apply(lambda s: pd.Series(
        {"n": len(s), **{m: (ll(s["over_2_5"].values, s[m].values)
                             - ll(s["over_2_5"].values, s["Bookmaker average"].values)).mean() for m in MODELS}}))
    paired = {m: (ll(gb["over_2_5"].values, gb[m].values) - ll(gb["over_2_5"].values, gb[V_C].values))
              for m in [V_SOT, V_XG]}
    paired_ci = {m: np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)], [2.5, 97.5])
                 for m, v in paired.items()}

    Bm = P[P["season"].isin(TEST) & P["Betfair close"].notna()]
    bets = []
    for m in MODELS:
        for price, oc, uc in [("Betfair early", "bfe_over_2_5", "bfe_under_2_5"),
                              ("Betfair close", "bfe_close_over_2_5", "bfe_close_under_2_5")]:
            bets.append({"model": m, "price": price, **bet_sim(Bm, m, oc, uc, rng)})
    bets = pd.DataFrame(bets)
    bets.to_csv(BETS_CSV, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    print(f"\n=== TEST {TEST[0]} - {TEST[-1]} - diff < 0 means model better ===")
    for ref in ["Bookmaker average", "Betfair close"]:
        print(f"\nvs {ref}:")
        print(summary[summary["vs"] == ref].drop(columns="vs").round(4).to_string(index=False))
    print("\nDoes the new data help? (log loss vs round 1 C on the same matches; < 0 = better)")
    for m, v in paired.items():
        print(f"  {m}: {v.mean():+.4f}  (95% interval {paired_ci[m][0]:+.4f} to {paired_ci[m][1]:+.4f})")
    print("\nBy team type vs bookmaker average (test seasons):")
    print(by_type.round(4).to_string())
    print("\nBetting simulation - flat 1 unit, EV >= 3% after 2% commission:")
    print(bets.round(4).to_string(index=False))

    close = summary[summary["vs"] == "Betfair close"].set_index("model")
    best = close["diff"].idxmin()
    gap = close.loc[best, "diff"]
    passed = gap <= BAR
    verdict = "PASSES - candidate to freeze as v3-conf" if passed else "does NOT pass the bar"
    print(f"\nBest vs Betfair close: {best}  gap {gap:+.4f}  (round 1 C: {close.loc[V_C, 'diff']:+.4f}, "
          f"bar: +{BAR:.4f}) -> {verdict}")

    lines = ["# National League - R&D round 3 (round 1 C + Bzzoiro shots on target and xG)\n",
             "No tuning. Variants fixed before running. Test 2024/25 - 2025/26 (third look at these seasons).\n",
             f"Bar: within +{BAR} of the Betfair close.\n"]
    for ref in ["Bookmaker average", "Betfair close"]:
        lines += [f"\n## vs {ref}\n", r1.md_table(summary[summary["vs"] == ref].drop(columns="vs")) + "\n"]
    lines += ["\n## Gain from new data vs round 1 C\n"]
    lines += [f"- {m}: {v.mean():+.4f} (95% interval {paired_ci[m][0]:+.4f} to {paired_ci[m][1]:+.4f})" for m, v in paired.items()]
    lines += ["\n## By team type\n", r1.md_table(by_type.reset_index()) + "\n",
              "\n## Betting simulation\n", r1.md_table(bets) + "\n",
              f"\n**Best vs Betfair close: {best}, gap {gap:+.4f} - {'passes' if passed else 'does not pass'} the bar.**\n"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_MD}, {SUMMARY_CSV}, {BETS_CSV}, {PREDS_CSV}")


if __name__ == "__main__":
    main()

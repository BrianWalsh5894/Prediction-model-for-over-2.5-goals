"""
National League (Conference) - R&D round 1: GOALS-ONLY v3 recipe.

football-data's National League file (EC) has no shots, shots on target or
corners before 2026/27, so the full v3 feature set is impossible. This
round uses the same v3 recipe with only the goals-based features (goal
form, Over 2.5 form, points, Elo, rest days, early-season blends) plus
Dixon-Coles ratings. Settings are v3's, unchanged - nothing is tuned, so
the test seasons stay clean.

Features are built on ALL English divisions + the National League together
(same build_feature_table code as v3), so teams relegated from League Two
carry their history and Elo into the National League. Teams promoted from
National League North/South have no history (league-average start).

Candidates (each 50/50 Poisson + Dixon-Coles, as v3):
  A. National League only: Poisson trained on NL; Dixon-Coles on NL
  B. Poisson trained on League Two + NL (league indicator feature);
     Dixon-Coles on NL only
  C. Poisson on League Two + NL; Dixon-Coles on League Two + NL linked by
     promotion / relegation (Brian's preferred setup, both parts pooled)

Benchmarks: bookmaker average (folds 2020/21 - 2025/26; odds start 2019/20)
and Betfair close (2024/25 + 2025/26). Betting simulation (flat 1 unit,
EV >= 3% after 2% commission) at the Betfair early and closing prices.
Also breaks the gap down by team type: promoted from North/South,
relegated from League Two, established.

Success bar (set BEFORE running, same as every model): within +0.002 of
the Betfair close over 2024/25 + 2025/26.

Needs data/raw/conference/EC_*.csv (check_conference_data.py) and
matches_clean.csv. Read-only: touches no live files or database tables.

Run from inside src/ (a few minutes):
    python rd_conference.py
"""

import glob
import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from rd_experiments3 import rolling_dc, poisson_fit_predict, NON_FEATURE_COLS  # noqa: E402
from build_features import build_feature_table  # noqa: E402
from import_helpers import build_match_id, season_code_to_display  # noqa: E402
from rd_eu_upper3 import bet_sim  # noqa: E402

CONF_DIR = os.path.join("..", "data", "raw", "conference")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
DB_PATH = os.path.join("..", "data", "over25.db")
REPORT_MD = os.path.join("..", "reports", "rd_conference.md")
SUMMARY_CSV = os.path.join("..", "reports", "rd_conference_summary.csv")
BETS_CSV = os.path.join("..", "reports", "rd_conference_bets.csv")
PREDS_CSV = os.path.join("..", "reports", "rd_conference_predictions.csv")

NL, L2 = "National League", "League Two"
ALL_SEASONS = ["2017/18", "2018/19", "2019/20", "2020/21", "2021/22",
               "2022/23", "2023/24", "2024/25", "2025/26"]
FOLDS = ALL_SEASONS[3:]                       # 2020/21 .. 2025/26
BETFAIR_SEASONS = ["2024/25", "2025/26"]
BAR = 0.002
SEED = 424410
N_BOOT = 2000

A = "A. NL only (Poisson NL, DC NL)"
B = "B. Poisson L2+NL, DC NL"
C = "C. Poisson L2+NL, DC L2+NL"
MODELS = [A, B, C]


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def demargin(o, u):
    return (1 / o) / (1 / o + 1 / u)


def md_table(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
    df = df.astype(str)
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in df.itertuples(index=False, name=None)])


def load_national_league():
    with sqlite3.connect(DB_PATH) as conn:
        name_map = dict(conn.execute("SELECT raw_name, standard_name FROM team_name_map").fetchall())
    frames = []
    for path in sorted(glob.glob(os.path.join(CONF_DIR, "EC_*.csv"))):
        code = os.path.basename(path)[3:7]
        season = season_code_to_display(code)
        if season not in ALL_SEASONS:
            continue
        raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        raw.columns = [c.replace("﻿", "").replace("ï»¿", "").strip() for c in raw.columns]
        raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        rows = []
        for r in raw.to_dict("records"):
            match_id, iso = build_match_id(r["Date"], "EC", r["HomeTeam"], r["AwayTeam"])
            hg, ag = int(r["FTHG"]), int(r["FTAG"])
            rows.append({
                "match_id": match_id, "season": season, "league_name": NL, "match_date": iso,
                "home_team": name_map.get(r["HomeTeam"], r["HomeTeam"]),
                "away_team": name_map.get(r["AwayTeam"], r["AwayTeam"]),
                "home_goals": hg, "away_goals": ag, "total_goals": hg + ag, "over_2_5": int(hg + ag > 2),
                "result": "H" if hg > ag else "A" if ag > hg else "D",
                "behind_closed_doors": int("2020-03-14" <= iso <= "2021-05-31"),
                "home_shots": np.nan, "away_shots": np.nan,
                "home_shots_on_target": np.nan, "away_shots_on_target": np.nan,
                "avg_over_2_5": r.get("Avg>2.5", np.nan), "avg_under_2_5": r.get("Avg<2.5", np.nan),
                "bfe_over_2_5": r.get("BFE>2.5", np.nan), "bfe_under_2_5": r.get("BFE<2.5", np.nan),
                "bfe_close_over_2_5": r.get("BFEC>2.5", np.nan), "bfe_close_under_2_5": r.get("BFEC<2.5", np.nan),
            })
        frames.append(pd.DataFrame(rows))
    nl = pd.concat(frames, ignore_index=True)
    nl["match_date"] = pd.to_datetime(nl["match_date"])
    return nl


def team_types(nl, eng):
    """For each NL match: is either team promoted from North/South (not in NL
    or any English league the previous season), relegated from League Two,
    or established?"""
    allm = pd.concat([eng, nl])
    in_season = {}
    for (lg, s), g in allm.groupby(["league_name", "season"]):
        in_season[(lg, s)] = set(g["home_team"]) | set(g["away_team"])
    prev = {s: ALL_SEASONS[i - 1] for i, s in enumerate(ALL_SEASONS) if i > 0}

    def kind(team, season):
        p = prev.get(season)
        if p is None:
            return "established"
        if team in in_season.get((NL, p), set()):
            return "established"
        if team in in_season.get((L2, p), set()):
            return "relegated_from_L2"
        return "promoted_from_N/S"

    t = []
    for r in nl.itertuples(index=False):
        k = {kind(r.home_team, r.season), kind(r.away_team, r.season)}
        t.append("promoted_from_N/S" if "promoted_from_N/S" in k else
                 "relegated_from_L2" if "relegated_from_L2" in k else "established")
    return pd.Series(t, index=nl["match_id"].values)


def main():
    rng = np.random.default_rng(SEED)
    eng = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    eng = eng[eng["season"].isin(ALL_SEASONS)]
    nl = load_national_league()
    print("National League matches per season:")
    print(nl.groupby("season").size().to_string())

    cols_needed = list(nl.columns)
    combined = pd.concat([eng[[c for c in cols_needed if c in eng.columns]], nl], ignore_index=True)
    print(f"\nBuilding goals-only features on all English divisions + NL ({len(combined)} matches)...")
    feats = build_feature_table(combined)
    feats = feats[feats["season"].isin(ALL_SEASONS)].reset_index(drop=True)
    cols = [c for c in feats.columns if c not in NON_FEATURE_COLS
            and "shots" not in c and "sot_" not in c and not c.startswith("league_")]
    cols += ["league_National League"]                  # league indicator for pooled training
    print(f"{len(cols)} goals-based features (no shots)")
    goals_df = combined[["match_id", "home_goals", "away_goals"]]

    print("\nDixon-Coles (weekly re-fits): NL only, and League Two + NL linked...")
    dc_nl = rolling_dc(combined[combined["league_name"] == NL], 0.0)[["match_id", "dc_p_over"]] \
        .rename(columns={"dc_p_over": "dc_nl"})
    dc_both = rolling_dc(combined[combined["league_name"].isin([L2, NL])], 0.0)[["match_id", "dc_p_over"]] \
        .rename(columns={"dc_p_over": "dc_both"})
    fa = feats.merge(dc_nl, on="match_id", how="left").merge(dc_both, on="match_id", how="left")

    preds = []
    for season in FOLDS:
        prev = ALL_SEASONS[:ALL_SEASONS.index(season)]
        tr = fa[fa["season"].isin(prev)]
        tr_nl = tr[tr["league_name"] == NL]
        tr_both = tr[tr["league_name"].isin([L2, NL])]
        te = fa[(fa["season"] == season) & (fa["league_name"] == NL)]
        idx = te["match_id"]
        p_nl = pd.Series(poisson_fit_predict(tr_nl, te, cols, goals_df), index=idx)
        p_both = pd.Series(poisson_fit_predict(tr_both, te, cols, goals_df), index=idx)
        dcn = te.set_index("match_id")["dc_nl"]
        dcb = te.set_index("match_id")["dc_both"]
        out = te[["match_id", "season", "league_name", "over_2_5"]].set_index("match_id").copy()
        out[A] = 0.5 * p_nl + 0.5 * dcn.fillna(p_nl)
        out[B] = 0.5 * p_both + 0.5 * dcn.fillna(p_both)
        out[C] = 0.5 * p_both + 0.5 * dcb.fillna(p_both)
        out["Bookmaker average"] = demargin(te["avg_over_2_5"], te["avg_under_2_5"]).values
        out["Betfair close"] = demargin(te["bfe_close_over_2_5"], te["bfe_close_under_2_5"]).values
        for col in ["bfe_over_2_5", "bfe_under_2_5", "bfe_close_over_2_5", "bfe_close_under_2_5"]:
            out[col] = te[col].values
        preds.append(out.reset_index())
        print(f"  predicted {season} ({len(te)} matches)")
    P = pd.concat(preds, ignore_index=True)
    P["team_type"] = P["match_id"].map(team_types(nl, eng))
    P.to_csv(PREDS_CSV, index=False)

    # ---- log loss vs market ----
    rows = []
    for ref, seasons in [("Bookmaker average", FOLDS), ("Betfair close", BETFAIR_SEASONS)]:
        g = P[P["season"].isin(seasons) & P[ref].notna()]
        y = g["over_2_5"].values
        ll_ref = ll(y, g[ref].values)
        for m in MODELS:
            d = ll(y, g[m].values) - ll_ref
            boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
            better = sum(ll(s["over_2_5"].values, s[m].values).mean() < ll(s["over_2_5"].values, s[ref].values).mean()
                         for _, s in g.groupby("season"))
            rows.append({"vs": ref, "model": m, "model_ll": ll(y, g[m].values).mean(), "market_ll": ll_ref.mean(),
                         "diff": d.mean(), "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                         "seasons_beating_market": f"{better}/{g['season'].nunique()}", "n": len(g)})
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    # per season vs bookmaker average, best model
    close = summary[summary["vs"] == "Betfair close"].set_index("model")
    best = close["diff"].idxmin()
    g = P[P["Bookmaker average"].notna()]
    per_season = g.groupby("season").apply(lambda s: pd.Series({
        "n": len(s), "over_rate": s["over_2_5"].mean(), "model_avg_p": s[best].mean(),
        "book_avg_p": s["Bookmaker average"].mean(),
        "diff_vs_book": (ll(s["over_2_5"].values, s[best].values) - ll(s["over_2_5"].values, s["Bookmaker average"].values)).mean()}))
    by_type = g.groupby("team_type").apply(lambda s: pd.Series({
        "n": len(s),
        "diff_vs_book": (ll(s["over_2_5"].values, s[best].values) - ll(s["over_2_5"].values, s["Bookmaker average"].values)).mean()}))

    # ---- betting simulation ----
    Bm = P[P["season"].isin(BETFAIR_SEASONS) & P["Betfair close"].notna()]
    bets = []
    for m in MODELS:
        for label, oc, uc in [("Betfair early", "bfe_over_2_5", "bfe_under_2_5"),
                              ("Betfair close", "bfe_close_over_2_5", "bfe_close_under_2_5")]:
            bets.append({"model": m, "price": label, **bet_sim(Bm, m, oc, uc, rng)})
    bets = pd.DataFrame(bets)
    bets.to_csv(BETS_CSV, index=False)
    gap_pp = (Bm[best] - Bm["Betfair close"]).abs().mean() * 100
    corr = np.corrcoef(Bm[best], Bm["Betfair close"])[0, 1]

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    for ref in ["Bookmaker average", "Betfair close"]:
        print(f"\n=== vs {ref} - diff < 0 means model better ===")
        print(summary[summary["vs"] == ref].drop(columns="vs").round(4).to_string(index=False))
    print(f"\nBest vs Betfair close: {best}")
    print("\nPer season vs bookmaker average (best model):")
    print(per_season.round(4).to_string())
    print("\nBy team type vs bookmaker average (best model, all folds):")
    print(by_type.round(4).to_string())
    print(f"\nHow far off (2024/25 + 2025/26, {len(Bm)} matches): average gap to Betfair close {gap_pp:.1f} pp, "
          f"correlation {corr:.2f}")
    print("\nBetting simulation - flat 1 unit, EV >= 3% after 2% commission (2024/25 + 2025/26):")
    print(bets.round(4).to_string(index=False))

    gap = close.loc[best, "diff"]
    passed = gap <= BAR
    verdict = "PASSES - candidate to freeze as v3-conf" if passed else "does NOT pass the bar"
    print(f"\nBest vs Betfair close: {best}  gap {gap:+.4f}  (bar: +{BAR:.4f}) -> {verdict}")

    lines = ["# National League - R&D round 1 (goals-only v3 recipe)\n",
             "No shots before 2026/27, so goals-based features only. v3 settings, untuned. "
             "Walk-forward 2020/21 - 2025/26. `diff` < 0 = model better; 95% bootstrap interval.\n",
             f"Success bar set before running: within +{BAR} of the Betfair close.\n"]
    for ref in ["Bookmaker average", "Betfair close"]:
        lines += [f"\n## vs {ref}\n", md_table(summary[summary["vs"] == ref].drop(columns="vs")) + "\n"]
    lines += ["\n## Per season vs bookmaker average (best model)\n", md_table(per_season.reset_index()) + "\n",
              "\n## By team type (best model)\n", md_table(by_type.reset_index()) + "\n",
              f"\nGap to Betfair close {gap_pp:.1f} pp, correlation {corr:.2f}.\n",
              "\n## Betting simulation\n", md_table(bets) + "\n",
              f"\n**Best vs Betfair close: {best}, gap {gap:+.4f} - {'passes' if passed else 'does not pass'} the bar.**\n"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_MD}, {SUMMARY_CSV}, {BETS_CSV}, {PREDS_CSV}")


if __name__ == "__main__":
    main()

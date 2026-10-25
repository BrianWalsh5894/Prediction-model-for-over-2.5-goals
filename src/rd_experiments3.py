"""
R&D round 3 - Dixon-Coles WITH time decay, done as a proper rolling rating.

Why round 2's Dixon-Coles barely helped (0.6866 -> 0.6865)
---------------------------------------------------------
Round 2 fitted attack/defence ratings ONCE per fold on all training
seasons, then:
  1. used them as features for the SAME training matches they were fitted
     on (in-sample ratings - they "know" those results), while
  2. using them frozen and stale for the whole test season (no update
     after a single match, promoted teams get a flat average).
So the Poisson regression learned to trust ratings that were far better
in training than they could ever be at test time. That mismatch, plus no
time decay, is the most likely reason the gain was tiny.

What this round does instead
----------------------------
A ROLLING rating, re-fitted every week, using ONLY matches played before
that week. Every match - training or test - gets a rating that was
genuinely out-of-sample for it. This is exactly what the Phase 11 live
pipeline would do each week, so the historical test mirrors live use.

  - Model: one Dixon-Coles model across all four divisions (so promoted /
    relegated teams carry their rating between leagues).
      log(lambda_home) = attack[home] + defence[away] + home_adv
      log(lambda_away) = attack[away] + defence[home]
    plus the Dixon-Coles rho correction for 0-0, 1-0, 0-1 and 1-1, which
    matters for Over/Under 2.5 because those are all "Under" scorelines.
  - Time decay: each past match weighted by exp(-xi * days_ago), using
    the last 3 years of matches.
  - Team never seen before (e.g. promoted from the National League): gets
    the average rating of the teams in the league it is playing in.

How xi (the decay rate) is chosen - honestly
--------------------------------------------
xi is picked on 2018/19 + 2019/20, the two seasons BEFORE the six
walk-forward folds. The folds (2020/21 - 2025/26) are then used only to
judge the result, never to pick xi. The table of every xi on the folds
is shown for transparency but is not used to choose.

Candidates judged on the same 6-fold walk-forward mean as rounds 1-2:
  1. Poisson baseline (Phase 8 alpha 1.3335, existing 103 features)
  2. Dixon-Coles time-decay on its own (no ML at all)
  3. Poisson + rolling Dixon-Coles features
  4. 50/50 average of (1) and (2) - fixed weight, not tuned
  Market reference (same as round 2)

Run from inside src/ (budget ~5 minutes - the weekly re-fits are the
slow part):
    python rd_experiments3.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402,F401

FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
REPORT_MD = os.path.join("..", "reports", "rd_round3.md")
COMPARISON_CSV = os.path.join("..", "reports", "rd_round3_comparison.csv")
XI_CSV = os.path.join("..", "reports", "rd_round3_xi_search.csv")
FIG_DIR = os.path.join("..", "reports", "figures")

ALL_SEASONS_CHRONOLOGICAL = [
    "2017/18", "2018/19", "2019/20", "2020/21", "2021/22",
    "2022/23", "2023/24", "2024/25", "2025/26",
]
WALK_FORWARD_MIN_TRAIN_SEASONS = 3
XI_SELECTION_SEASONS = ["2018/19", "2019/20"]   # before the folds - used ONLY to pick xi
FOLD_SEASONS = ALL_SEASONS_CHRONOLOGICAL[WALK_FORWARD_MIN_TRAIN_SEASONS:]
POISSON_ALPHA = 1.3335                           # Phase 8 / round 2 pick

# Dixon-Coles settings
XI_GRID = [0.0, 0.001, 0.002, 0.0035, 0.005, 0.0075]   # per day; half-life = ln2 / xi
WINDOW_DAYS = 3 * 365          # only matches from the last 3 years are used in each fit
MIN_MATCHES_TO_FIT = 500       # don't produce ratings off a tiny history (start of 2017/18)
RIDGE = 1.0                    # light shrinkage so teams with few matches stay near average
RHO_BOUNDS = (-0.2, 0.2)

DC_FEATURES = [
    "dc_lam_home", "dc_lam_away", "dc_lam_total", "dc_p_over",
    "dc_home_attack", "dc_home_defence", "dc_away_attack", "dc_away_defence",
]

NON_FEATURE_COLS = {
    "match_id", "match_date", "season", "league_name", "home_team", "away_team",
    "over_2_5", "behind_closed_doors",
    "avg_over_2_5", "avg_under_2_5", "bfe_over_2_5", "bfe_under_2_5",
    "bfe_close_over_2_5", "bfe_close_under_2_5",
    "league_over25_avg_to_date", "league_goals_avg_to_date",
}


def md_table(df, float_fmt="{:.4f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: float_fmt.format(v) if pd.notna(v) else "")
        else:
            df[c] = df[c].astype(str)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep] + rows)


def safe_log_loss(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])


# ---------------------------------------------------------------------------
# Dixon-Coles with time decay
# ---------------------------------------------------------------------------

def fit_dc_weighted(home_idx, away_idx, hg, ag, w, n_teams, x0=None):
    """Weighted independent-Poisson fit of attack/defence/home_adv, with an
    analytic gradient (fast), then rho fitted on top as a 1-D problem."""

    def nll_and_grad(params):
        attack = params[:n_teams]
        defence = params[n_teams:2 * n_teams]
        home_adv = params[-1]
        eta_h = attack[home_idx] + defence[away_idx] + home_adv
        eta_a = attack[away_idx] + defence[home_idx]
        lam_h = np.exp(eta_h)
        lam_a = np.exp(eta_a)
        ll = w * (hg * eta_h - lam_h + ag * eta_a - lam_a)
        s_attack = attack.sum()
        nll = -ll.sum() + 0.5 * RIDGE * (attack @ attack + defence @ defence) + 100.0 * s_attack ** 2

        r_h = w * (hg - lam_h)   # d ll / d eta_h
        r_a = w * (ag - lam_a)
        g_attack = -(np.bincount(home_idx, r_h, n_teams) + np.bincount(away_idx, r_a, n_teams))
        g_defence = -(np.bincount(away_idx, r_h, n_teams) + np.bincount(home_idx, r_a, n_teams))
        g_attack += RIDGE * attack + 200.0 * s_attack
        g_defence += RIDGE * defence
        g_home = -r_h.sum()
        return nll, np.concatenate([g_attack, g_defence, [g_home]])

    if x0 is None:
        x0 = np.zeros(2 * n_teams + 1)
    res = minimize(nll_and_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    params = res.x
    attack = params[:n_teams]
    defence = params[n_teams:2 * n_teams]
    home_adv = params[-1]

    lam_h = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
    lam_a = np.exp(attack[away_idx] + defence[home_idx])

    def neg_rho_ll(rho):
        tau = np.ones_like(lam_h)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
        tau[m01] = 1 + lam_h[m01] * rho
        tau[m10] = 1 + lam_a[m10] * rho
        tau[m11] = 1 - rho
        if np.any(tau <= 0):
            return 1e12
        return -(w * np.log(tau)).sum()

    rho = minimize_scalar(neg_rho_ll, bounds=RHO_BOUNDS, method="bounded").x
    return params, float(rho)


def dc_p_over(lam_h, lam_a, rho):
    """P(total goals >= 3) under Dixon-Coles: 1 - P(scorelines totalling <= 2),
    with the rho correction applied to 0-0, 1-0, 0-1, 1-1."""
    ph = [poisson_dist.pmf(k, lam_h) for k in range(3)]
    pa = [poisson_dist.pmf(k, lam_a) for k in range(3)]
    p_under = (
        ph[0] * pa[0] * (1 - lam_h * lam_a * rho)
        + ph[1] * pa[0] * (1 + lam_a * rho)
        + ph[0] * pa[1] * (1 + lam_h * rho)
        + ph[1] * pa[1] * (1 - rho)
        + ph[2] * pa[0]
        + ph[0] * pa[2]
    )
    return np.clip(1 - p_under, 1e-6, 1 - 1e-6)


def rolling_dc(matches, xi):
    """For every week, fit Dixon-Coles on matches strictly BEFORE that week
    (last WINDOW_DAYS days, weighted exp(-xi * days_ago)) and rate that
    week's matches. Returns one row per match_id with the DC_FEATURES."""
    m = matches.sort_values("match_date").reset_index(drop=True)
    m["week_start"] = m["match_date"] - pd.to_timedelta(m["match_date"].dt.weekday, unit="D")
    m["week_start"] = m["week_start"].dt.normalize()

    out_rows = []
    prev_params = {}   # team -> (attack, defence), for warm-starting the next week
    prev_home_adv = 0.0

    for week in sorted(m["week_start"].unique()):
        week = pd.Timestamp(week)
        hist = m[(m["match_date"] < week) & (m["match_date"] >= week - pd.Timedelta(days=WINDOW_DAYS))]
        target = m[m["week_start"] == week]
        if len(hist) < MIN_MATCHES_TO_FIT:
            continue

        teams = sorted(set(hist["home_team"]) | set(hist["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        days_ago = (week - hist["match_date"]).dt.days.values.astype(float)
        w = np.exp(-xi * days_ago)

        x0 = np.zeros(2 * n + 1)
        for t, i in idx.items():
            if t in prev_params:
                x0[i], x0[n + i] = prev_params[t]
        x0[-1] = prev_home_adv

        params, rho = fit_dc_weighted(
            hist["home_team"].map(idx).values, hist["away_team"].map(idx).values,
            hist["home_goals"].values.astype(float), hist["away_goals"].values.astype(float),
            w, n, x0,
        )
        attack = params[:n]
        defence = params[n:2 * n]
        home_adv = params[-1]
        prev_params = {t: (attack[i], defence[i]) for t, i in idx.items()}
        prev_home_adv = home_adv

        # League average rating, for teams with no history (e.g. promoted from non-league):
        # a team's league = the league of its most recent match in the window
        last_league = (
            pd.concat([
                hist[["match_date", "home_team", "league_name"]].rename(columns={"home_team": "team"}),
                hist[["match_date", "away_team", "league_name"]].rename(columns={"away_team": "team"}),
            ])
            .sort_values("match_date").groupby("team")["league_name"].last()
        )
        league_avg = {}
        for lg, grp in last_league.groupby(last_league):
            ii = [idx[t] for t in grp.index]
            league_avg[lg] = (attack[ii].mean(), defence[ii].mean())

        def rating(team, league):
            if team in idx:
                return attack[idx[team]], defence[idx[team]]
            return league_avg.get(league, (0.0, 0.0))

        for r in target.itertuples(index=False):
            ha, hd = rating(r.home_team, r.league_name)
            aa, ad = rating(r.away_team, r.league_name)
            lam_h = float(np.exp(ha + ad + home_adv))
            lam_a = float(np.exp(aa + hd))
            out_rows.append({
                "match_id": r.match_id,
                "dc_lam_home": lam_h, "dc_lam_away": lam_a, "dc_lam_total": lam_h + lam_a,
                "dc_p_over": float(dc_p_over(lam_h, lam_a, rho)),
                "dc_home_attack": ha, "dc_home_defence": hd,
                "dc_away_attack": aa, "dc_away_defence": ad,
            })
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Poisson regression (same as round 2)
# ---------------------------------------------------------------------------

def poisson_fit_predict(train, test, cols, goals_df):
    medians = train[cols].median(numeric_only=True)
    Xtr = train[cols].fillna(medians)
    Xte = test[cols].fillna(medians)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    tg = train[["match_id"]].merge(goals_df, on="match_id", how="left")
    m_home = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000).fit(Xtr_s, tg["home_goals"])
    m_away = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000).fit(Xtr_s, tg["away_goals"])
    lam = m_home.predict(Xte_s) + m_away.predict(Xte_s)
    return 1 - poisson_dist.cdf(2, lam)


def market_probs(test):
    """Same market reference as round 2: BFE closing odds for 2024/25+,
    average bookmaker odds before that. NaN where no odds."""
    season = test["season"].iloc[0]
    if season in ("2024/25", "2025/26"):
        o, u = test["bfe_close_over_2_5"], test["bfe_close_under_2_5"]
    else:
        o, u = test["avg_over_2_5"], test["avg_under_2_5"]
    io, iu = 1 / o, 1 / u
    return (io / (io + iu)).values


def main():
    t0 = time.time()
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])
    df = df[df["season"].isin(ALL_SEASONS_CHRONOLOGICAL)].reset_index(drop=True)
    clean = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    clean = clean[clean["season"].isin(ALL_SEASONS_CHRONOLOGICAL)]
    goals_df = clean[["match_id", "home_goals", "away_goals"]]
    base_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"{len(df)} matches, {len(base_cols)} base features")

    # --- Step 1: rolling Dixon-Coles for every xi -------------------------
    dc_by_xi = {}
    for xi in XI_GRID:
        t = time.time()
        dc_by_xi[xi] = rolling_dc(clean, xi)
        half = "none (flat 3-year window)" if xi == 0 else f"{np.log(2) / xi:.0f} days"
        print(f"  xi={xi:<7} half-life {half:<26} rated {len(dc_by_xi[xi])} matches in {time.time() - t:.0f}s")

    # --- Step 2: choose xi on 2018/19 + 2019/20 ONLY ----------------------
    xi_rows = []
    for xi, dc in dc_by_xi.items():
        d = df[["match_id", "season", "over_2_5"]].merge(dc[["match_id", "dc_p_over"]], on="match_id")
        sel = d[d["season"].isin(XI_SELECTION_SEASONS)]
        row = {"xi": xi, "half_life_days": (np.log(2) / xi) if xi > 0 else np.nan,
               "selection_log_loss": safe_log_loss(sel["over_2_5"], sel["dc_p_over"]),
               "selection_n": len(sel)}
        folds = d[d["season"].isin(FOLD_SEASONS)]
        row["fold_mean_log_loss_info_only"] = np.mean([
            safe_log_loss(g["over_2_5"], g["dc_p_over"]) for _, g in folds.groupby("season")
        ])
        xi_rows.append(row)
    xi_df = pd.DataFrame(xi_rows)
    xi_df.to_csv(XI_CSV, index=False)
    best_xi = float(xi_df.sort_values("selection_log_loss").iloc[0]["xi"])
    print("\nxi search (chosen on 2018/19 + 2019/20 only):")
    print(xi_df.to_string(index=False))
    print(f"Chosen xi: {best_xi}")

    dc = dc_by_xi[best_xi]
    dfx = df.merge(dc, on="match_id", how="left")

    # --- Step 3: walk-forward comparison ----------------------------------
    rows = []
    for i in range(WALK_FORWARD_MIN_TRAIN_SEASONS, len(ALL_SEASONS_CHRONOLOGICAL)):
        train_seasons = ALL_SEASONS_CHRONOLOGICAL[:i]
        season = ALL_SEASONS_CHRONOLOGICAL[i]
        train = dfx[dfx["season"].isin(train_seasons)]
        test = dfx[dfx["season"] == season]
        y = test["over_2_5"].values

        p_base = poisson_fit_predict(train, test, base_cols, goals_df)
        p_dc = test["dc_p_over"].fillna(pd.Series(p_base, index=test.index)).values
        p_both = poisson_fit_predict(train, test, base_cols + DC_FEATURES, goals_df)
        p_blend = 0.5 * p_base + 0.5 * p_dc
        p_mkt = market_probs(test)
        has_mkt = ~np.isnan(p_mkt)

        preds = {
            "Poisson baseline (round 2)": p_base,
            f"Dixon-Coles time-decay alone (xi={best_xi})": p_dc,
            "Poisson + rolling Dixon-Coles features": p_both,
            "50/50 blend: Poisson + Dixon-Coles": p_blend,
        }
        for name, p in preds.items():
            rows.append({
                "candidate": name, "predicts": season,
                "log_loss": safe_log_loss(y, p),
                "brier_score": brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                "log_loss_market_matches": safe_log_loss(y[has_mkt], p[has_mkt]),
                "n": len(y),
            })
        ll_mkt = safe_log_loss(y[has_mkt], p_mkt[has_mkt])
        rows.append({
            "candidate": "Market (reference)", "predicts": season,
            "log_loss": ll_mkt,
            "brier_score": brier_score_loss(y[has_mkt], p_mkt[has_mkt]),
            "log_loss_market_matches": ll_mkt, "n": int(has_mkt.sum()),
        })
        print(f"  {season}: " + ", ".join(
            f"{r['candidate'].split(' (')[0].split(':')[0]}={r['log_loss']:.4f}" for r in rows[-5:]))

    detail = pd.DataFrame(rows)
    detail.to_csv(COMPARISON_CSV, index=False)

    summary = (
        detail.groupby("candidate")
        .agg(mean_log_loss=("log_loss", "mean"), std_log_loss=("log_loss", "std"),
             mean_ll_on_market_matches=("log_loss_market_matches", "mean"),
             mean_brier=("brier_score", "mean"), n_folds=("log_loss", "count"))
        .reset_index().sort_values("mean_log_loss")
    )
    mkt = detail[detail["candidate"] == "Market (reference)"].set_index("predicts")["log_loss"]
    beats = (
        detail[detail["candidate"] != "Market (reference)"]
        .assign(beats=lambda d: d["log_loss_market_matches"] < d["predicts"].map(mkt))
        .groupby("candidate")["beats"].sum()
    )
    summary["folds_beating_market"] = summary["candidate"].map(beats).fillna(0).astype(int)
    pd.set_option("display.width", 200)
    print("\n" + summary.to_string(index=False))

    # --- Figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = xi_df["xi"].values
    axes[0].plot(xs, xi_df["selection_log_loss"], marker="o", label="2018/19 + 2019/20 (used to choose)")
    axes[0].plot(xs, xi_df["fold_mean_log_loss_info_only"], marker="s", linestyle="--",
                 label="6 folds (shown only, not used)")
    axes[0].axvline(best_xi, color="green", linestyle=":", label=f"chosen xi = {best_xi}")
    axes[0].set_xlabel("xi (decay per day)")
    axes[0].set_ylabel("Log Loss - Dixon-Coles alone")
    axes[0].set_title("Time-decay search")
    axes[0].legend(fontsize=8)
    for name in detail["candidate"].unique():
        sub = detail[detail["candidate"] == name]
        axes[1].plot(sub["predicts"], sub["log_loss"], marker="o",
                     linestyle="--" if name.startswith("Market") else "-", label=name, markersize=4)
    axes[1].set_xlabel("Season predicted")
    axes[1].set_ylabel("Log Loss")
    axes[1].set_title("Round 3 candidates vs market")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rd_round3_dixon_coles.png"), dpi=120)
    plt.close(fig)

    # --- Report ---------------------------------------------------------------
    models_only = summary[summary["candidate"] != "Market (reference)"].reset_index(drop=True)
    best_name, best_ll = models_only.iloc[0]["candidate"], models_only.iloc[0]["mean_log_loss"]
    market_ll = summary.loc[summary["candidate"] == "Market (reference)", "mean_log_loss"].values[0]
    base_ll = summary.loc[summary["candidate"] == "Poisson baseline (round 2)", "mean_log_loss"].values[0]

    lines = [
        "# R&D round 3 - Dixon-Coles with time decay (rolling, weekly re-fit)\n",
        "Round 2's Dixon-Coles ratings were fitted once per fold and used in-sample on the "
        "training matches, then held frozen for the whole test season. Round 3 re-fits every "
        "week on matches strictly before that week (last 3 years, weighted exp(-xi * days_ago)), "
        "with the rho low-score correction, so every match - training and test - gets a rating "
        "that is genuinely out-of-sample, exactly as the live pipeline would produce it.\n",
        "## Choosing xi (on 2018/19 + 2019/20 only - not on the folds)\n",
        md_table(xi_df) + "\n",
        f"\nChosen xi: **{best_xi}**"
        + (f" (half-life about {np.log(2) / best_xi:.0f} days)" if best_xi > 0 else " (no decay)") + ".\n",
        "\n## Walk-forward comparison (6 folds, 2020/21 - 2025/26)\n",
        md_table(summary) + "\n",
        "\n`mean_ll_on_market_matches` scores each model only on matches where market odds exist, "
        "for a like-for-like comparison. `folds_beating_market` counts folds where the model beat "
        "the market on those same matches.\n",
        "\n![Round 3](figures/rd_round3_dixon_coles.png)\n",
        f"\n**Best: {best_name}** - mean log loss {best_ll:.4f} vs Poisson baseline {base_ll:.4f} "
        f"and market {market_ll:.4f} ({'beats' if best_ll < market_ll else 'does NOT beat'} the market).\n",
        "\nHistorical sandbox only - not proof of a live edge.\n",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {REPORT_MD}, {COMPARISON_CSV}, {XI_CSV}, figures/rd_round3_dixon_coles.png")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

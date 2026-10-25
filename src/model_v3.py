"""
Model v3 - the frozen live model (Phase 11 / 13).

    P(Over 2.5) = 0.5 * Poisson  +  0.5 * Dixon-Coles

  - Poisson: two PoissonRegressors (home goals, away goals) on the 103
    Phase 5 features, alpha 1.3335. Trained ONCE by freeze_model_v3.py on
    2017/18 - 2025/26 and never retrained while it is being tracked.
  - Dixon-Coles: attack/defence/home advantage + rho, fitted on the last
    3 years of completed matches before the prediction date (no time
    decay: xi = 0, round 3's pick). Re-fitted every time predictions are
    made - that is part of the frozen METHOD, exactly as tested in round 3
    (weekly re-fit); its settings never change.

This is the "50/50 blend" from R&D round 3 (walk-forward mean log loss
0.6860). Every number here is a copy of what was tested - do not tune.
"""

import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson as poisson_dist
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "v3"
MODELS_DIR = os.path.join("..", "models")
MANIFEST_PATH = os.path.join(MODELS_DIR, "manifest_v3.json")
BUNDLE_PATH = os.path.join(MODELS_DIR, "model_v3.joblib")

TRAIN_SEASONS = [
    "2017/18", "2018/19", "2019/20", "2020/21", "2021/22",
    "2022/23", "2023/24", "2024/25", "2025/26",
]
POISSON_ALPHA = 1.3335
BLEND_WEIGHT_POISSON = 0.5

DC_XI = 0.0
DC_WINDOW_DAYS = 3 * 365
DC_MIN_MATCHES = 500
DC_RIDGE = 1.0
DC_RHO_BOUNDS = (-0.2, 0.2)

NON_FEATURE_COLS = {
    "match_id", "match_date", "season", "league_name", "home_team", "away_team",
    "over_2_5", "behind_closed_doors",
    "avg_over_2_5", "avg_under_2_5", "bfe_over_2_5", "bfe_under_2_5",
    "bfe_close_over_2_5", "bfe_close_under_2_5",
    "league_over25_avg_to_date", "league_goals_avg_to_date",
}


def feature_columns(features_df):
    return [c for c in features_df.columns if c not in NON_FEATURE_COLS]


# ---------------------------------------------------------------------------
# Poisson part (frozen once)
# ---------------------------------------------------------------------------

def train_poisson(train_features, goals_df, feature_cols):
    medians = train_features[feature_cols].median(numeric_only=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(train_features[feature_cols].fillna(medians))
    tg = train_features[["match_id"]].merge(goals_df, on="match_id", how="left")
    m_home = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000).fit(X, tg["home_goals"])
    m_away = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000).fit(X, tg["away_goals"])
    return {"feature_cols": list(feature_cols), "medians": medians, "scaler": scaler,
            "m_home": m_home, "m_away": m_away}


def predict_poisson(bundle, features_df):
    cols = bundle["feature_cols"]
    missing = [c for c in cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"Features missing for the frozen model: {missing}")
    X = bundle["scaler"].transform(features_df[cols].fillna(bundle["medians"]))
    lam_h = bundle["m_home"].predict(X)
    lam_a = bundle["m_away"].predict(X)
    return lam_h, lam_a, 1 - poisson_dist.cdf(2, lam_h + lam_a)


def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f"{BUNDLE_PATH} not found - run freeze_model_v3.py first.")
    return joblib.load(BUNDLE_PATH)


# ---------------------------------------------------------------------------
# Dixon-Coles part (same code as rd_experiments3.py)
# ---------------------------------------------------------------------------

def _fit_dc_weighted(home_idx, away_idx, hg, ag, w, n_teams, x0=None):
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
        nll = -ll.sum() + 0.5 * DC_RIDGE * (attack @ attack + defence @ defence) + 100.0 * s_attack ** 2
        r_h = w * (hg - lam_h)
        r_a = w * (ag - lam_a)
        g_attack = -(np.bincount(home_idx, r_h, n_teams) + np.bincount(away_idx, r_a, n_teams))
        g_defence = -(np.bincount(away_idx, r_h, n_teams) + np.bincount(home_idx, r_a, n_teams))
        g_attack += DC_RIDGE * attack + 200.0 * s_attack
        g_defence += DC_RIDGE * defence
        return nll, np.concatenate([g_attack, g_defence, [-r_h.sum()]])

    if x0 is None:
        x0 = np.zeros(2 * n_teams + 1)
    params = minimize(nll_and_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    attack, defence, home_adv = params[:n_teams], params[n_teams:2 * n_teams], params[-1]
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

    rho = minimize_scalar(neg_rho_ll, bounds=DC_RHO_BOUNDS, method="bounded").x
    return params, float(rho)


def dc_p_over(lam_h, lam_a, rho):
    lam_h, lam_a = np.asarray(lam_h, float), np.asarray(lam_a, float)
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


def fit_dc_asof(completed_matches, asof, x0_ratings=None):
    """Fit Dixon-Coles on completed matches strictly before `asof` (last
    DC_WINDOW_DAYS days). Returns a dict with everything needed to rate
    any fixture on or after `asof`."""
    asof = pd.Timestamp(asof).normalize()
    m = completed_matches
    hist = m[(m["match_date"] < asof) & (m["match_date"] >= asof - pd.Timedelta(days=DC_WINDOW_DAYS))]
    hist = hist.dropna(subset=["home_goals", "away_goals"])
    if len(hist) < DC_MIN_MATCHES:
        return None
    teams = sorted(set(hist["home_team"]) | set(hist["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    w = np.exp(-DC_XI * (asof - hist["match_date"]).dt.days.values.astype(float))

    x0 = np.zeros(2 * n + 1)
    if x0_ratings:
        for t, i in idx.items():
            if t in x0_ratings["ratings"]:
                x0[i], x0[n + i] = x0_ratings["ratings"][t]
        x0[-1] = x0_ratings["home_adv"]

    params, rho = _fit_dc_weighted(
        hist["home_team"].map(idx).values, hist["away_team"].map(idx).values,
        hist["home_goals"].values.astype(float), hist["away_goals"].values.astype(float), w, n, x0,
    )
    attack, defence, home_adv = params[:n], params[n:2 * n], params[-1]
    last_league = (
        pd.concat([
            hist[["match_date", "home_team", "league_name"]].rename(columns={"home_team": "team"}),
            hist[["match_date", "away_team", "league_name"]].rename(columns={"away_team": "team"}),
        ]).sort_values("match_date").groupby("team")["league_name"].last()
    )
    league_avg = {}
    for lg, grp in last_league.groupby(last_league):
        ii = [idx[t] for t in grp.index]
        league_avg[lg] = (attack[ii].mean(), defence[ii].mean())
    return {"ratings": {t: (attack[i], defence[i]) for t, i in idx.items()},
            "home_adv": home_adv, "rho": rho, "league_avg": league_avg, "n_matches": len(hist)}


def predict_dc(fit, fixtures):
    """fixtures: DataFrame with home_team, away_team, league_name."""
    def rating(team, league):
        if team in fit["ratings"]:
            return fit["ratings"][team]
        return fit["league_avg"].get(league, (0.0, 0.0))

    h = [rating(t, lg) for t, lg in zip(fixtures["home_team"], fixtures["league_name"])]
    a = [rating(t, lg) for t, lg in zip(fixtures["away_team"], fixtures["league_name"])]
    lam_h = np.array([np.exp(hr[0] + ar[1] + fit["home_adv"]) for hr, ar in zip(h, a)])
    lam_a = np.array([np.exp(ar[0] + hr[1]) for hr, ar in zip(h, a)])
    return lam_h, lam_a, dc_p_over(lam_h, lam_a, fit["rho"])


# ---------------------------------------------------------------------------
# The full v3 prediction
# ---------------------------------------------------------------------------

def predict_v3(bundle, fixture_features, completed_matches, asof):
    """Returns a DataFrame (one row per fixture) with both components and
    the blended probability."""
    lam_ph, lam_pa, p_pois = predict_poisson(bundle, fixture_features)
    fit = fit_dc_asof(completed_matches, asof)
    if fit is None:
        raise RuntimeError("Not enough history to fit Dixon-Coles.")
    lam_dh, lam_da, p_dc = predict_dc(fit, fixture_features)
    out = fixture_features[["match_id", "match_date", "season", "league_name", "home_team", "away_team"]].copy()
    out["poisson_lam_home"] = lam_ph
    out["poisson_lam_away"] = lam_pa
    out["p_poisson"] = p_pois
    out["dc_lam_home"] = lam_dh
    out["dc_lam_away"] = lam_da
    out["p_dc"] = p_dc
    out["p_over"] = BLEND_WEIGHT_POISSON * p_pois + (1 - BLEND_WEIGHT_POISSON) * p_dc
    return out


# ---------------------------------------------------------------------------
# Standard interface used by the live pipeline (league_groups.py)
# ---------------------------------------------------------------------------

def prepare_features(rows):
    """Feature table for played matches + upcoming fixtures (v3 = the 103
    Phase 5 features, unchanged)."""
    from build_features import build_feature_table
    return build_feature_table(rows)


def predict(bundle, fixture_features, completed_matches, asof):
    return predict_v3(bundle, fixture_features, completed_matches, asof)

"""
Model v3-eu-upper - frozen live model for La Liga, Serie A, Bundesliga and
Ligue 1 ("D" from rd_eu_upper2.py: second divisions + xG).

    P(Over 2.5) = 0.5 * Poisson  +  0.5 * Dixon-Coles

  - Poisson: home/away PoissonRegressor, alpha 1.3335, on the 103 Phase 5
    features (built on top AND second divisions, so promoted teams carry
    their form) + 23 Understat xG features. Trained ONCE on the top-division
    matches 2017/18 - 2025/26 by freeze_model_v3_eu_upper.py.
  - Dixon-Coles: fitted per COUNTRY on top + second division together
    (linked by promotion), last 3 years before the prediction date, no time
    decay - same frozen settings and code as v3 (model_v3.fit_dc_asof).

Backtest (rd_eu_upper2.py): log loss 0.6762 vs bookmaker average 0.6713 and
+0.0081 vs Betfair close; betting simulation -3.6% (early) / -5.0% (close).
Did NOT pass the +0.002 bar - tracked live as an experiment.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import joblib

import model_v3 as base   # frozen v3 code: Poisson training + Dixon-Coles

MODEL_VERSION = "v3-eu-upper"
MODELS_DIR = os.path.join("..", "models")
MANIFEST_PATH = os.path.join(MODELS_DIR, "manifest_v3_eu_upper.json")
BUNDLE_PATH = os.path.join(MODELS_DIR, "model_v3_eu_upper.joblib")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean_eu_upper.csv")
DB_PATH = os.path.join("..", "data", "over25.db")

TRAIN_SEASONS = base.TRAIN_SEASONS
COUNTRIES = {"La Liga": "La Liga 2", "Serie A": "Serie B", "Bundesliga": "2. Bundesliga", "Ligue 1": "Ligue 2"}
TOP = list(COUNTRIES)
UNDERSTAT_LEAGUES = TOP
BLEND_WEIGHT_POISSON = 0.5
NON_FEATURE_COLS = base.NON_FEATURE_COLS


# ---------------------------------------------------------------------------
# xG features (exact copy of rd_xg_upper.build_xg_features - frozen here so
# later R&D edits can never change the live model)
# ---------------------------------------------------------------------------

def build_xg_features(matches):
    home = pd.DataFrame({"match_id": matches["match_id"], "match_date": matches["match_date"],
                         "season": matches["season"], "team": matches["home_team"], "is_home": 1,
                         "xg_for": matches["home_xg"], "xg_against": matches["away_xg"],
                         "gf": matches["home_goals"], "ga": matches["away_goals"]})
    away = pd.DataFrame({"match_id": matches["match_id"], "match_date": matches["match_date"],
                         "season": matches["season"], "team": matches["away_team"], "is_home": 0,
                         "xg_for": matches["away_xg"], "xg_against": matches["home_xg"],
                         "gf": matches["away_goals"], "ga": matches["home_goals"]})
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "match_date", "match_id"])
    g = long.groupby("team")
    for col in ["xg_for", "xg_against", "gf", "ga"]:
        prev = g[col].shift(1)
        for w in (5, 10):
            long[f"{col}_r{w}"] = prev.groupby(long["team"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        long[f"{col}_season"] = prev.groupby([long["team"], long["season"]]).transform(lambda s: s.expanding().mean())
    for col in ["xg_for", "xg_against"]:
        prev_v = long.groupby(["team", "is_home"])[col].shift(1)
        long[f"{col}_venue_r5"] = prev_v.groupby([long["team"], long["is_home"]]).transform(
            lambda s: s.rolling(5, min_periods=1).mean())
    long["luck_for_r10"] = long["gf_r10"] - long["xg_for_r10"]
    long["luck_against_r10"] = long["ga_r10"] - long["xg_against_r10"]
    keep = ["xg_for_r5", "xg_for_r10", "xg_against_r5", "xg_against_r10", "xg_for_season",
            "xg_against_season", "xg_for_venue_r5", "xg_against_venue_r5", "luck_for_r10", "luck_against_r10"]
    h = long[long["is_home"] == 1].set_index("match_id")[keep].add_prefix("xg_home_")
    a = long[long["is_home"] == 0].set_index("match_id")[keep].add_prefix("xg_away_")
    out = h.join(a)
    out["xg_exp_home"] = (out["xg_home_xg_for_r10"] + out["xg_away_xg_against_r10"]) / 2
    out["xg_exp_away"] = (out["xg_away_xg_for_r10"] + out["xg_home_xg_against_r10"]) / 2
    out["xg_exp_total"] = out["xg_exp_home"] + out["xg_exp_away"]
    return out.reset_index()


def load_xg():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT match_id, home_xg, away_xg FROM match_xg", conn)


# ---------------------------------------------------------------------------
# Standard live interface (league_groups.py / weekly_predict.py)
# ---------------------------------------------------------------------------

def refresh_data():
    """Run by weekly_predict.py before predicting: latest Understat xG."""
    import download_understat
    for lg in UNDERSTAT_LEAGUES:
        download_understat.process(lg, offline=False)


def prepare_features(rows):
    """rows: played matches (all 8 divisions) + upcoming fixtures. Returns
    the Phase 5 features + xG features (top divisions)."""
    from build_features import build_feature_table
    feats = build_feature_table(rows)
    top = rows[rows["league_name"].isin(TOP)].merge(load_xg(), on="match_id", how="left")
    xg = build_xg_features(top.sort_values(["match_date", "match_id"]))
    return feats.merge(xg, on="match_id", how="left")


def feature_columns(features_df):
    return [c for c in features_df.columns if c not in NON_FEATURE_COLS]


def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f"{BUNDLE_PATH} not found - run freeze_model_v3_eu_upper.py first.")
    return joblib.load(BUNDLE_PATH)


def fit_dc_by_country(completed_matches, asof, prev=None):
    fits = {}
    for top, second in COUNTRIES.items():
        sub = completed_matches[completed_matches["league_name"].isin([top, second])]
        fits[top] = base.fit_dc_asof(sub, asof, (prev or {}).get(top))
    return fits


def predict_dc_by_country(fits, fixtures):
    p = pd.Series(np.nan, index=fixtures.index)
    lam_h = pd.Series(np.nan, index=fixtures.index)
    lam_a = pd.Series(np.nan, index=fixtures.index)
    for top in TOP:
        m = fixtures["league_name"] == top
        if m.any() and fits.get(top) is not None:
            lh, la, pp = base.predict_dc(fits[top], fixtures[m])
            p[m], lam_h[m], lam_a[m] = pp, lh, la
    return lam_h.values, lam_a.values, p.values


def predict(bundle, fixture_features, completed_matches, asof):
    lam_ph, lam_pa, p_pois = base.predict_poisson(bundle, fixture_features)
    fits = fit_dc_by_country(completed_matches, asof)
    lam_dh, lam_da, p_dc = predict_dc_by_country(fits, fixture_features)
    p_dc = np.where(np.isnan(p_dc), p_pois, p_dc)
    out = fixture_features[["match_id", "match_date", "season", "league_name", "home_team", "away_team"]].copy()
    out["poisson_lam_home"] = lam_ph
    out["poisson_lam_away"] = lam_pa
    out["p_poisson"] = p_pois
    out["dc_lam_home"] = lam_dh
    out["dc_lam_away"] = lam_da
    out["p_dc"] = p_dc
    out["p_over"] = BLEND_WEIGHT_POISSON * p_pois + (1 - BLEND_WEIGHT_POISSON) * p_dc
    return out

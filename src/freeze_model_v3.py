"""
Freezes model v3 for live paper-tracking (League One + League Two).

Trains the Poisson half of the 50/50 blend ONCE on every completed season
(2017/18 - 2025/26) and saves it with a manifest recording exactly what
was frozen. Refuses to run again if v3 already exists - a model being
tracked live must never change. (If the .joblib is ever lost, delete the
manifest and re-run: training is deterministic, and the script checks the
new model reproduces the manifest's fingerprint.)

Before saving it re-checks the method against R&D round 3: a walk-forward
of exactly this code must reproduce the round 3 blend (0.6860).

Run from inside src/:
    python freeze_model_v3.py
"""

import os
import sys
import json
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402
import model_v3 as mv  # noqa: E402

FEATURES_CSV = os.path.join("..", "data", "clean", "features.csv")
CLEAN_CSV = os.path.join("..", "data", "clean", "matches_clean.csv")
ROUND3_BLEND = 0.6860


def fingerprint(bundle):
    parts = [bundle["m_home"].coef_, bundle["m_away"].coef_,
             [bundle["m_home"].intercept_, bundle["m_away"].intercept_]]
    raw = np.concatenate([np.ravel(p) for p in parts]).round(10).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def walk_forward_check(feats, clean, goals_df, cols):
    """Round 3 walk-forward using model_v3's own functions (weekly DC re-fit)."""
    clean = clean.sort_values("match_date")
    weeks = clean["match_date"] - pd.to_timedelta(clean["match_date"].dt.weekday, unit="D")
    clean = clean.assign(week_start=weeks.dt.normalize())
    dc_rows, prev = [], None
    for week in sorted(clean.loc[clean["season"].isin(mv.TRAIN_SEASONS[3:]), "week_start"].unique()):
        fit = mv.fit_dc_asof(clean, week, prev)
        prev = fit
        tgt = clean[clean["week_start"] == week]
        _, _, p = mv.predict_dc(fit, tgt)
        dc_rows.append(pd.DataFrame({"match_id": tgt["match_id"].values, "p_dc": p}))
    dc = pd.concat(dc_rows)
    losses = []
    for i in range(3, len(mv.TRAIN_SEASONS)):
        train = feats[feats["season"].isin(mv.TRAIN_SEASONS[:i])]
        test = feats[feats["season"] == mv.TRAIN_SEASONS[i]].merge(dc, on="match_id", how="left")
        b = mv.train_poisson(train, goals_df, cols)
        _, _, p_pois = mv.predict_poisson(b, test)
        p = mv.BLEND_WEIGHT_POISSON * p_pois + (1 - mv.BLEND_WEIGHT_POISSON) * test["p_dc"].fillna(pd.Series(p_pois, index=test.index)).values
        losses.append(log_loss(test["over_2_5"], np.clip(p, 1e-6, 1 - 1e-6)))
        print(f"  {mv.TRAIN_SEASONS[i]}: {losses[-1]:.4f}")
    return float(np.mean(losses))


def main():
    if os.path.exists(mv.MANIFEST_PATH) and os.path.exists(mv.BUNDLE_PATH):
        print(f"{mv.MANIFEST_PATH} already exists - v3 is frozen. Nothing to do.")
        return

    feats = pd.read_csv(FEATURES_CSV, parse_dates=["match_date"])
    clean = pd.read_csv(CLEAN_CSV, parse_dates=["match_date"])
    feats = feats[feats["season"].isin(mv.TRAIN_SEASONS)].reset_index(drop=True)
    clean_hist = clean[clean["season"].isin(mv.TRAIN_SEASONS)]
    goals_df = clean_hist[["match_id", "home_goals", "away_goals"]]
    cols = mv.feature_columns(feats)

    print("Re-checking the method against R&D round 3 (walk-forward)...")
    wf = walk_forward_check(feats, clean_hist, goals_df, cols)
    print(f"Walk-forward mean log loss: {wf:.4f} (round 3: {ROUND3_BLEND:.4f})")
    if abs(wf - ROUND3_BLEND) > 0.0005:
        print("STOP: does not reproduce round 3 - not freezing.")
        sys.exit(1)

    print(f"\nTraining the frozen Poisson on {len(feats)} matches ({mv.TRAIN_SEASONS[0]} - {mv.TRAIN_SEASONS[-1]})...")
    bundle = mv.train_poisson(feats, goals_df, cols)
    fp = fingerprint(bundle)

    if os.path.exists(mv.MANIFEST_PATH):   # joblib lost: re-creating, must be identical
        with open(mv.MANIFEST_PATH, encoding="utf-8") as f:
            old = json.load(f)
        if old["poisson_fingerprint"] != fp:
            print(f"STOP: re-trained model fingerprint {fp} != frozen {old['poisson_fingerprint']}.")
            sys.exit(1)
        joblib.dump(bundle, mv.BUNDLE_PATH)
        print(f"Re-created {mv.BUNDLE_PATH} - identical to the frozen v3 (fingerprint {fp}).")
        return

    os.makedirs(mv.MODELS_DIR, exist_ok=True)
    joblib.dump(bundle, mv.BUNDLE_PATH)
    manifest = {
        "model_version": mv.MODEL_VERSION,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "description": "50/50 blend of Poisson regression (frozen) and Dixon-Coles (re-fitted on the "
                       "last 3 years of completed matches each time predictions are made, fixed settings)",
        "live_leagues": config.LIVE_LEAGUES,
        "trained_on_seasons": mv.TRAIN_SEASONS,
        "n_training_matches": int(len(feats)),
        "poisson": {"alpha": mv.POISSON_ALPHA, "n_features": len(cols), "features": cols},
        "dixon_coles": {"xi_per_day": mv.DC_XI, "window_days": mv.DC_WINDOW_DAYS, "ridge": mv.DC_RIDGE,
                        "rho_bounds": list(mv.DC_RHO_BOUNDS), "min_matches": mv.DC_MIN_MATCHES},
        "blend_weight_poisson": mv.BLEND_WEIGHT_POISSON,
        "walk_forward_check_log_loss": round(wf, 6),
        "poisson_fingerprint": fp,
        "betting_rules": {"commission": config.COMMISSION_RATE, "min_edge": config.MIN_EDGE,
                          "stake": config.PAPER_STAKE, "staking": "flat"},
        "evidence": "R&D rounds 1-4 + league_breakdown: does NOT beat the market historically. "
                    "Paper-tracking only; no real money unless live tracking over 200+ bets shows an edge.",
    }
    with open(mv.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nFROZEN: {mv.BUNDLE_PATH} + {mv.MANIFEST_PATH} (fingerprint {fp})")


if __name__ == "__main__":
    main()

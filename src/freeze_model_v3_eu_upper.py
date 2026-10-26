"""
Freezes model v3-eu-upper (La Liga, Serie A, Bundesliga, Ligue 1) for live
paper-tracking. Same rules as freeze_model_v3.py:
  - re-checks the method with a walk-forward using the live model's OWN code
    (weekly Dixon-Coles re-fit per country) - must reproduce rd_eu_upper2.py
    candidate D (log loss 0.6762 on 8,452 top-division matches)
  - trains the Poisson half ONCE on 2017/18 - 2025/26 top-division matches
  - writes models/model_v3_eu_upper.joblib + manifest_v3_eu_upper.json
  - refuses to run again once frozen (a tracked model never changes)

Needs: download_eu_upper.py, clean_data.py and download_understat.py run first.

Run from inside src/ (a few minutes - the walk-forward check is the slow part):
    python freeze_model_v3_eu_upper.py
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
import model_v3 as base  # noqa: E402
import model_v3_eu_upper as mv  # noqa: E402

ROUND2_D = 0.6762


def fingerprint(bundle):
    parts = [bundle["m_home"].coef_, bundle["m_away"].coef_,
             [bundle["m_home"].intercept_, bundle["m_away"].intercept_]]
    raw = np.concatenate([np.ravel(p) for p in parts]).round(10).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def main():
    if os.path.exists(mv.MANIFEST_PATH) and os.path.exists(mv.BUNDLE_PATH):
        print(f"{mv.MANIFEST_PATH} already exists - v3-eu-upper is frozen. Nothing to do.")
        return

    clean = pd.read_csv(mv.CLEAN_CSV, parse_dates=["match_date"])
    clean = clean[clean["season"].isin(mv.TRAIN_SEASONS)].sort_values("match_date")
    goals_df = clean[["match_id", "home_goals", "away_goals"]]
    print("Building features (top + second divisions, + xG)...")
    feats = mv.prepare_features(clean)
    feats = feats[feats["season"].isin(mv.TRAIN_SEASONS)].reset_index(drop=True)
    cols = mv.feature_columns(feats)
    top = feats[feats["league_name"].isin(mv.TOP)].reset_index(drop=True)
    print(f"{len(top)} top-division matches, {len(cols)} features")

    # --- walk-forward check with the live model's own code -------------------
    print("\nRe-checking the method (weekly Dixon-Coles per country, as live)...")
    weeks = (clean["match_date"] - pd.to_timedelta(clean["match_date"].dt.weekday, unit="D")).dt.normalize()
    clean = clean.assign(week_start=weeks)
    dc_rows, prev = [], None
    tgt_all = clean[clean["season"].isin(mv.TRAIN_SEASONS[3:]) & clean["league_name"].isin(mv.TOP)]
    for week in sorted(tgt_all["week_start"].unique()):
        fits = mv.fit_dc_by_country(clean, week, prev)
        prev = fits
        tgt = tgt_all[tgt_all["week_start"] == week]
        _, _, p = mv.predict_dc_by_country(fits, tgt)
        dc_rows.append(pd.DataFrame({"match_id": tgt["match_id"].values, "p_dc": p}))
    dc = pd.concat(dc_rows)
    ys, ps = [], []
    for i in range(3, len(mv.TRAIN_SEASONS)):
        train = top[top["season"].isin(mv.TRAIN_SEASONS[:i])]
        test = top[top["season"] == mv.TRAIN_SEASONS[i]].merge(dc, on="match_id", how="left")
        b = base.train_poisson(train, goals_df, cols)
        _, _, p_pois = base.predict_poisson(b, test)
        p = 0.5 * p_pois + 0.5 * test["p_dc"].fillna(pd.Series(p_pois, index=test.index)).values
        ys.append(test["over_2_5"].values)
        ps.append(p)
        print(f"  {mv.TRAIN_SEASONS[i]}: {log_loss(ys[-1], np.clip(p, 1e-6, 1 - 1e-6)):.4f}")
    wf = log_loss(np.concatenate(ys), np.clip(np.concatenate(ps), 1e-6, 1 - 1e-6))
    print(f"Walk-forward log loss (all {sum(len(y) for y in ys)} matches): {wf:.4f} (rd_eu_upper2 D: {ROUND2_D:.4f})")
    if abs(wf - ROUND2_D) > 0.0005:
        print("STOP: does not reproduce candidate D - not freezing.")
        sys.exit(1)

    print(f"\nTraining the frozen Poisson on {len(top)} top-division matches...")
    bundle = base.train_poisson(top, goals_df, cols)
    fp = fingerprint(bundle)

    if os.path.exists(mv.MANIFEST_PATH):
        with open(mv.MANIFEST_PATH, encoding="utf-8") as f:
            old = json.load(f)
        if old["poisson_fingerprint"] != fp:
            print(f"STOP: re-trained fingerprint {fp} != frozen {old['poisson_fingerprint']}.")
            sys.exit(1)
        joblib.dump(bundle, mv.BUNDLE_PATH)
        print(f"Re-created {mv.BUNDLE_PATH} - identical to the frozen model (fingerprint {fp}).")
        return

    os.makedirs(mv.MODELS_DIR, exist_ok=True)
    joblib.dump(bundle, mv.BUNDLE_PATH)
    manifest = {
        "model_version": mv.MODEL_VERSION,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "description": "50/50 blend: Poisson (103 base features built on top + second divisions, + 23 Understat "
                       "xG features; frozen) and Dixon-Coles per country on top + second division (re-fitted "
                       "on the last 3 years each time predictions are made, fixed settings)",
        "live_leagues": mv.TOP,
        "history_leagues": list(mv.COUNTRIES.values()),
        "trained_on_seasons": mv.TRAIN_SEASONS,
        "n_training_matches": int(len(top)),
        "poisson": {"alpha": base.POISSON_ALPHA, "n_features": len(cols), "features": cols},
        "dixon_coles": {"xi_per_day": base.DC_XI, "window_days": base.DC_WINDOW_DAYS, "ridge": base.DC_RIDGE,
                        "rho_bounds": list(base.DC_RHO_BOUNDS), "fitted": "per country, top + second division"},
        "blend_weight_poisson": mv.BLEND_WEIGHT_POISSON,
        "walk_forward_check_log_loss": round(wf, 6),
        "poisson_fingerprint": fp,
        "betting_rules": {"commission": config.COMMISSION_RATE, "min_edge": config.MIN_EDGE,
                          "stake": config.PAPER_STAKE, "staking": "flat"},
        "evidence": "rd_eu_upper2/3: +0.0049 vs bookmaker average, +0.0081 vs Betfair close; betting simulation "
                    "-3.6% (early) / -5.0% (close). Did NOT pass the +0.002 bar - experiment, paper only.",
    }
    with open(mv.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nFROZEN: {mv.BUNDLE_PATH} + {mv.MANIFEST_PATH} (fingerprint {fp})")


if __name__ == "__main__":
    main()

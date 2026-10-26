"""
League groups (step 3): one place that says which leagues belong together,
which features they use and which frozen model is tracked live for them.
The groups themselves are defined in config/config.py (LEAGUE_GROUPS).

To add a new live model for a group: write model_<version>.py exposing
MODEL_VERSION, load_bundle(), prepare_features(rows) and
predict(bundle, fixture_features, completed_matches, asof), register it in
MODEL_MODULES below, freeze it, then set live_model in config.
"""

import os
import sys
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402

MODEL_MODULES = {
    "v3": "model_v3",
    "v3-eu-upper": "model_v3_eu_upper",
}


def group(name):
    if name not in config.LEAGUE_GROUPS:
        raise SystemExit(f"Unknown league group '{name}'. Groups: {list(config.LEAGUE_GROUPS)}")
    return config.LEAGUE_GROUPS[name]


def live_model(name):
    """The frozen model module tracked live for this group (stops if none)."""
    version = group(name)["live_model"]
    if version is None:
        raise SystemExit(f"Group '{name}' has no frozen live model yet (still in R&D).")
    return importlib.import_module(MODEL_MODULES[version])


def group_of_league(league):
    for name, g in config.LEAGUE_GROUPS.items():
        if league in g["leagues"]:
            return name
    return None


def live_groups():
    return [n for n, g in config.LEAGUE_GROUPS.items() if g["live_model"] is not None]

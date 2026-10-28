# Project configuration — single source of truth for key values

# Betting / value model
COMMISSION_RATE = 0.02        # Betfair Exchange commission (e.g. 0.02 = 2%)
MIN_EDGE = 0.03                # minimum edge/EV before a bet is considered

# Database
DB_PATH = "data/over25.db"

# Seasons (time-based split, decided up front — do not change after seeing test results)
TRAIN_SEASONS = ["2017/18", "2018/19", "2019/20", "2020/21", "2021/22", "2022/23"]
VALIDATION_SEASONS = ["2023/24"]
TEST_SEASONS = ["2024/25", "2025/26"]

# Leagues
LEAGUES = ["Premier League", "Championship", "League One", "League Two"]

# Feature engineering (Phase 5)
ROLLING_WINDOWS = [5, 10]            # last-N-match rolling windows for form features
MIN_GAMES_SEASON_FOR_FULL_FORM = 5   # below this many games played THIS season, blend with prior season / league average
MIN_GAMES_CAREER_FOR_TRAINING = 3    # matches where either team has fewer career games than this are flagged ineligible for training
ELO_INITIAL = 1500                   # starting Elo rating for a team with no history
ELO_K = 20                           # Elo K-factor (how much one result moves a rating)
ELO_HOME_ADVANTAGE = 65              # Elo points added to the home team's rating before computing expected score
# Live paper-tracking (Phase 11)
PAPER_STAKE = 1.0                             # flat stake per paper bet, in units

# League groups (step 3). Each group has its own feature set and its own
# frozen model, trained and tracked separately. live_model = the frozen
# model version tracked live for that group (None = still in R&D, no live
# predictions). A group's live model must never change while it is tracked -
# a new model gets a new version and starts a new track record.
LEAGUE_GROUPS = {
    "lower": {
        "leagues": ["League One", "League Two"],
        "features": "base",               # the 103 Phase 5 features
        "live_model": "v3",
    },
    "upper": {
        "leagues": ["Premier League", "Championship"],
        "features": "base",               # xG (step 4) didn't beat the market - not used live
        "live_model": "v3",               # same frozen v3, separate track record (from 25 Oct 2026)
    },
    "eu_upper": {
        "leagues": ["La Liga", "Serie A", "Bundesliga", "Ligue 1"],
        "features": "base (top + 2nd division history) + Understat xG",
        "live_model": "v3-eu-upper",      # experiment - did not pass the +0.002 bar (rd_eu_upper2/3)
    },
    "conference": {
        "leagues": ["National League"],
        "features": "goals-based + shots on target + Bzzoiro xG (rd_conference3 C+SOT+xG)",
        "live_model": "v3-conf",          # experiment - did not pass the +0.002 bar (rd_conference3);
                                          # predicted with predict_conference.py, not weekly_predict.py
    },
}
LIVE_LEAGUES = LEAGUE_GROUPS["lower"]["leagues"]   # kept: recorded in manifest_v3.json

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
LIVE_LEAGUES = ["League One", "League Two"]   # leagues model v3 is tracked on
PAPER_STAKE = 1.0                             # flat stake per paper bet, in units

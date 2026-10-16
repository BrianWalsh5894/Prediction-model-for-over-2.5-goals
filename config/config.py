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
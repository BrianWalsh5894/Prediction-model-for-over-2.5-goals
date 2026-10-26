# v3-eu-upper improvement round 2 - calibration + how far off

## Bias check (all folds)

| league_name | matches | actual_over_rate | model_avg | bookmaker_avg | model_minus_actual |
|---|---|---|---|---|---|
| Bundesliga | 1836 | 0.6062 | 0.6022 | 0.5909 | -0.0040 |
| La Liga | 2280 | 0.4680 | 0.4661 | 0.4739 | -0.0018 |
| Ligue 1 | 2058 | 0.5364 | 0.5142 | 0.5250 | -0.0222 |
| Serie A | 2280 | 0.5018 | 0.5179 | 0.5109 | 0.0162 |


## Log loss (2021/22 - 2025/26)

| vs | method | model_ll | market_ll | diff | ci_low | ci_high | n |
|---|---|---|---|---|---|---|---|
| Bookmaker average | D as is | 0.6754 | 0.6709 | 0.0046 | 0.0024 | 0.0067 | 7007 |
| Bookmaker average | D + Platt (all leagues) | 0.6760 | 0.6709 | 0.0051 | 0.0031 | 0.0072 | 7007 |
| Bookmaker average | D + Platt (per league) | 0.6766 | 0.6709 | 0.0058 | 0.0035 | 0.0081 | 7007 |
| Bookmaker average | D + shift (per league) | 0.6761 | 0.6709 | 0.0053 | 0.0031 | 0.0076 | 7007 |
| Betfair close | D as is | 0.6741 | 0.6660 | 0.0081 | 0.0039 | 0.0121 | 2655 |
| Betfair close | D + Platt (all leagues) | 0.6741 | 0.6660 | 0.0081 | 0.0037 | 0.0122 | 2655 |
| Betfair close | D + Platt (per league) | 0.6744 | 0.6660 | 0.0084 | 0.0044 | 0.0127 | 2655 |
| Betfair close | D + shift (per league) | 0.6750 | 0.6660 | 0.0089 | 0.0047 | 0.0130 | 2655 |


## How far off (2024/25 + 2025/26, 2655 matches)

Average gap to Betfair close: 4.2 percentage points; correlation 0.87.

| method | price | bets | share_of_matches | avg_model_EV | profit_units | ROI | ROI_ci_low | ROI_ci_high |
|---|---|---|---|---|---|---|---|---|
| D as is | Betfair early price | 1383 | 0.5255 | 0.0996 | -49.2606 | -0.0356 | -0.0952 | 0.0266 |
| D as is | Betfair closing price | 1765 | 0.6648 | 0.1230 | -89.3332 | -0.0506 | -0.1061 | 0.0024 |
| D + Platt (all leagues) | Betfair early price | 1394 | 0.5296 | 0.1013 | -56.1290 | -0.0403 | -0.0994 | 0.0202 |
| D + Platt (all leagues) | Betfair closing price | 1787 | 0.6731 | 0.1248 | -98.0966 | -0.0549 | -0.1117 | 0.0001 |
| D + Platt (per league) | Betfair early price | 1380 | 0.5243 | 0.0963 | -78.0922 | -0.0566 | -0.1144 | 0.0027 |
| D + Platt (per league) | Betfair closing price | 1737 | 0.6542 | 0.1211 | -116.8758 | -0.0673 | -0.1203 | -0.0145 |
| D + shift (per league) | Betfair early price | 1369 | 0.5201 | 0.1011 | -62.1090 | -0.0454 | -0.1012 | 0.0142 |
| D + shift (per league) | Betfair closing price | 1770 | 0.6667 | 0.1222 | -116.4196 | -0.0658 | -0.1200 | -0.0128 |


**Best vs Betfair close: D + Platt (all leagues), gap +0.0081.**

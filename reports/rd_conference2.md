# National League - R&D round 2 (tuning + level/trend features)

Tuned on results only, 2020/21 - 2023/24 (1200 settings + 3 calibrations). Locked setting tested once on 2024/25 - 2025/26. Bar: within +0.002 of the Betfair close.


**Locked setting:** `{'fset': '+level', 'alpha': 0.3, 'window': 1, 'xi': 0.0, 'newcomer': 'weak', 'w': 0.5, 'calib': 'none'}`


Tuning log loss: round-1 C 0.6858, best 0.6810.


## Top 10 settings (tuning seasons)

| fset | alpha | window | xi | newcomer | w | tune_ll |
|---|---|---|---|---|---|---|
| +level | 0.3000 | 1 | 0.0000 | weak | 0.5000 | 0.6810 |
| +level | 0.3000 | 1 | 0.0000 | weak | 0.6000 | 0.6810 |
| +level | 0.3000 | 1 | 0.0020 | weak | 0.5000 | 0.6810 |
| +level | 0.3000 | 1 | 0.0020 | weak | 0.6000 | 0.6810 |
| +level | 0.6000 | 1 | 0.0000 | weak | 0.5000 | 0.6810 |
| +level | 0.3000 | 1 | 0.0000 | avg | 0.6000 | 0.6810 |
| +level | 0.3000 | 1 | 0.0000 | avg | 0.5000 | 0.6810 |
| +level | 0.6000 | 1 | 0.0000 | weak | 0.6000 | 0.6811 |
| +level | 0.3000 | 1 | 0.0020 | avg | 0.6000 | 0.6811 |
| +level | 0.6000 | 1 | 0.0020 | weak | 0.5000 | 0.6811 |


## Calibration (2021/22 - 2023/24)

| calibration | tune_ll_2021-24 |
|---|---|
| none | 0.6781 |
| shift | 0.6787 |
| Platt | 0.6802 |


## Final test

| vs | model | model_ll | market_ll | diff | ci_low | ci_high | n |
|---|---|---|---|---|---|---|---|
| Bookmaker average | round 2 (locked) | 0.6908 | 0.6815 | 0.0093 | 0.0016 | 0.0167 | 1092 |
| Bookmaker average | round 1 C | 0.6879 | 0.6815 | 0.0065 | -0.0003 | 0.0134 | 1092 |
| Betfair close | round 2 (locked) | 0.6906 | 0.6817 | 0.0088 | 0.0010 | 0.0172 | 1087 |
| Betfair close | round 1 C | 0.6878 | 0.6817 | 0.0061 | -0.0020 | 0.0138 | 1087 |


## By team type (final test)

| team_type | n | round2_vs_book | round1_vs_book |
|---|---|---|---|
| established | 605.0000 | 0.0074 | 0.0030 |
| promoted_from_N/S | 341.0000 | 0.0050 | 0.0080 |
| relegated_from_L2 | 146.0000 | 0.0275 | 0.0175 |


## Predicted vs actual Over rate (final test)

| season | over_rate | round2_avg_p | round1_avg_p | book_avg_p |
|---|---|---|---|---|
| 2024/25 | 0.5036 | 0.5222 | 0.5116 | 0.5136 |
| 2025/26 | 0.5704 | 0.5187 | 0.5054 | 0.5495 |


Gap to Betfair close 5.1 pp, correlation 0.67.


## Betting simulation

| model | price | bets | share_of_matches | avg_model_EV | profit_units | ROI | ROI_ci_low | ROI_ci_high |
|---|---|---|---|---|---|---|---|---|
| round 2 (locked) | Betfair early | 368 | 0.3552 | 0.1102 | -17.0594 | -0.0464 | -0.1466 | 0.0587 |
| round 2 (locked) | Betfair close | 690 | 0.6348 | 0.1235 | -3.4756 | -0.0050 | -0.0882 | 0.0710 |
| round 1 C | Betfair early | 373 | 0.3600 | 0.1117 | -24.0026 | -0.0644 | -0.1719 | 0.0404 |
| round 1 C | Betfair close | 664 | 0.6109 | 0.1354 | -16.2062 | -0.0244 | -0.1055 | 0.0567 |


**Round 2 vs Betfair close: +0.0088 - does not pass the bar.**

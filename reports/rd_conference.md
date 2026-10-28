# National League - R&D round 1 (goals-only v3 recipe)

No shots before 2026/27, so goals-based features only. v3 settings, untuned. Walk-forward 2020/21 - 2025/26. `diff` < 0 = model better; 95% bootstrap interval.

Success bar set before running: within +0.002 of the Betfair close.


## vs Bookmaker average

| model | model_ll | market_ll | diff | ci_low | ci_high | seasons_beating_market | n |
|---|---|---|---|---|---|---|---|
| A. NL only (Poisson NL, DC NL) | 0.6874 | 0.6836 | 0.0038 | -0.0003 | 0.0079 | 1/6 | 3177 |
| B. Poisson L2+NL, DC NL | 0.6875 | 0.6836 | 0.0039 | 0.0001 | 0.0081 | 1/6 | 3177 |
| C. Poisson L2+NL, DC L2+NL | 0.6866 | 0.6836 | 0.0030 | -0.0010 | 0.0069 | 1/6 | 3177 |


## vs Betfair close

| model | model_ll | market_ll | diff | ci_low | ci_high | seasons_beating_market | n |
|---|---|---|---|---|---|---|---|
| A. NL only (Poisson NL, DC NL) | 0.6891 | 0.6817 | 0.0074 | -0.0004 | 0.0154 | 0/2 | 1087 |
| B. Poisson L2+NL, DC NL | 0.6891 | 0.6817 | 0.0074 | -0.0002 | 0.0152 | 0/2 | 1087 |
| C. Poisson L2+NL, DC L2+NL | 0.6878 | 0.6817 | 0.0061 | -0.0013 | 0.0139 | 0/2 | 1087 |


## Per season vs bookmaker average (best model)

| season | n | over_rate | model_avg_p | book_avg_p | diff_vs_book |
|---|---|---|---|---|---|
| 2020/21 | 476.0000 | 0.5273 | 0.4817 | 0.5033 | 0.0015 |
| 2021/22 | 506.0000 | 0.4960 | 0.4873 | 0.5123 | 0.0036 |
| 2022/23 | 552.0000 | 0.5072 | 0.5020 | 0.5256 | -0.0034 |
| 2023/24 | 551.0000 | 0.5535 | 0.5105 | 0.5422 | 0.0032 |
| 2024/25 | 552.0000 | 0.5036 | 0.5116 | 0.5136 | 0.0029 |
| 2025/26 | 540.0000 | 0.5704 | 0.5054 | 0.5495 | 0.0101 |


## By team type (best model)

| team_type | n | diff_vs_book |
|---|---|---|
| established | 1954.0000 | 0.0008 |
| promoted_from_N/S | 843.0000 | 0.0042 |
| relegated_from_L2 | 380.0000 | 0.0115 |


Gap to Betfair close 5.0 pp, correlation 0.67.


## Betting simulation

| model | price | bets | share_of_matches | avg_model_EV | profit_units | ROI | ROI_ci_low | ROI_ci_high |
|---|---|---|---|---|---|---|---|---|
| A. NL only (Poisson NL, DC NL) | Betfair early | 379 | 0.3658 | 0.1070 | -27.3934 | -0.0723 | -0.1777 | 0.0323 |
| A. NL only (Poisson NL, DC NL) | Betfair close | 658 | 0.6053 | 0.1292 | -15.6828 | -0.0238 | -0.1054 | 0.0570 |
| B. Poisson L2+NL, DC NL | Betfair early | 382 | 0.3687 | 0.1076 | -28.5122 | -0.0746 | -0.1756 | 0.0270 |
| B. Poisson L2+NL, DC NL | Betfair close | 666 | 0.6127 | 0.1306 | -19.5080 | -0.0293 | -0.1113 | 0.0477 |
| C. Poisson L2+NL, DC L2+NL | Betfair early | 373 | 0.3600 | 0.1117 | -24.0026 | -0.0644 | -0.1713 | 0.0501 |
| C. Poisson L2+NL, DC L2+NL | Betfair close | 664 | 0.6109 | 0.1354 | -16.2062 | -0.0244 | -0.1056 | 0.0554 |


**Best vs Betfair close: C. Poisson L2+NL, DC L2+NL, gap +0.0061 - does not pass the bar.**

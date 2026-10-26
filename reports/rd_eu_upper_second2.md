# v3-eu-upper-second - R&D round 2 (tuning)

Tuned on results only, 2020/21 - 2023/24 (720 settings + 4 calibrations). Locked setting tested once on 2024/25 - 2025/26. Bar: within +0.002 of the Betfair close.


**Locked setting:** `{'alpha': 0.3, 'train': 'both', 'window': 3, 'xi': 0.002, 'newcomer': 'avg', 'w': 0.7, 'calib': 'Platt (per league)'}`


Tuning log loss: round-1 setting 0.6781, best 0.6755.


## Top 10 settings (tuning seasons)

| alpha | train | window | xi | newcomer | w | tune_ll |
|---|---|---|---|---|---|---|
| 0.3000 | both | 3 | 0.0020 | avg | 0.7000 | 0.6755 |
| 0.3000 | both | 3 | 0.0020 | weak | 0.7000 | 0.6756 |
| 0.3000 | both | 3 | 0.0000 | avg | 0.7000 | 0.6756 |
| 0.3000 | both | 3 | 0.0000 | weak | 0.7000 | 0.6756 |
| 0.3000 | both | 2 | 0.0020 | avg | 0.7000 | 0.6757 |
| 0.3000 | both | 2 | 0.0020 | weak | 0.7000 | 0.6757 |
| 0.6000 | both | 3 | 0.0020 | avg | 0.7000 | 0.6758 |
| 0.3000 | both | 2 | 0.0000 | avg | 0.7000 | 0.6758 |
| 0.6000 | both | 3 | 0.0020 | weak | 0.7000 | 0.6758 |
| 0.6000 | both | 3 | 0.0000 | avg | 0.7000 | 0.6758 |


## Calibration (2021/22 - 2023/24)

| calibration | tune_ll_2021-24 |
|---|---|
| Platt (per league) | 0.6776 |
| Platt (all leagues) | 0.6778 |
| none | 0.6778 |
| shift (per league) | 0.6789 |


## Final test

| league | vs | model | model_ll | market_ll | diff | ci_low | ci_high | n |
|---|---|---|---|---|---|---|---|---|
| ALL | Bookmaker average | round 2 (locked) | 0.6896 | 0.6821 | 0.0075 | 0.0041 | 0.0107 | 2907 |
| ALL | Bookmaker average | round 1 B | 0.6910 | 0.6821 | 0.0089 | 0.0050 | 0.0126 | 2907 |
| ALL | Betfair close | round 2 (locked) | 0.6897 | 0.6819 | 0.0078 | 0.0040 | 0.0118 | 2820 |
| ALL | Betfair close | round 1 B | 0.6912 | 0.6819 | 0.0093 | 0.0052 | 0.0134 | 2820 |
| La Liga 2 | Bookmaker average | round 2 (locked) | 0.6929 | 0.6819 | 0.0109 | 0.0045 | 0.0172 | 924 |
| La Liga 2 | Bookmaker average | round 1 B | 0.6921 | 0.6819 | 0.0102 | 0.0037 | 0.0165 | 924 |
| La Liga 2 | Betfair close | round 2 (locked) | 0.6929 | 0.6818 | 0.0112 | 0.0037 | 0.0190 | 898 |
| La Liga 2 | Betfair close | round 1 B | 0.6919 | 0.6818 | 0.0102 | 0.0026 | 0.0177 | 898 |
| Serie B | Bookmaker average | round 2 (locked) | 0.6942 | 0.6911 | 0.0031 | -0.0028 | 0.0089 | 760 |
| Serie B | Bookmaker average | round 1 B | 0.6992 | 0.6911 | 0.0081 | 0.0012 | 0.0147 | 760 |
| Serie B | Betfair close | round 2 (locked) | 0.6953 | 0.6921 | 0.0032 | -0.0046 | 0.0103 | 739 |
| Serie B | Betfair close | round 1 B | 0.7008 | 0.6921 | 0.0086 | 0.0010 | 0.0164 | 739 |
| 2. Bundesliga | Bookmaker average | round 2 (locked) | 0.6772 | 0.6715 | 0.0057 | -0.0007 | 0.0126 | 612 |
| 2. Bundesliga | Bookmaker average | round 1 B | 0.6754 | 0.6715 | 0.0039 | -0.0035 | 0.0113 | 612 |
| 2. Bundesliga | Betfair close | round 2 (locked) | 0.6762 | 0.6701 | 0.0061 | -0.0016 | 0.0140 | 591 |
| 2. Bundesliga | Betfair close | round 1 B | 0.6739 | 0.6701 | 0.0038 | -0.0043 | 0.0124 | 591 |
| Ligue 2 | Bookmaker average | round 2 (locked) | 0.6913 | 0.6819 | 0.0094 | 0.0024 | 0.0158 | 611 |
| Ligue 2 | Bookmaker average | round 1 B | 0.6949 | 0.6819 | 0.0130 | 0.0045 | 0.0219 | 611 |
| Ligue 2 | Betfair close | round 2 (locked) | 0.6915 | 0.6814 | 0.0102 | 0.0021 | 0.0179 | 592 |
| Ligue 2 | Betfair close | round 1 B | 0.6955 | 0.6814 | 0.0141 | 0.0043 | 0.0238 | 592 |


## Bias check

| league_name | actual | round2 | round1 | market |
|---|---|---|---|---|
| 2. Bundesliga | 0.5866 | 0.5990 | 0.5740 | 0.5827 |
| La Liga 2 | 0.4935 | 0.4359 | 0.4452 | 0.4458 |
| Ligue 2 | 0.4779 | 0.4707 | 0.4482 | 0.4673 |
| Serie B | 0.4711 | 0.4403 | 0.4571 | 0.4444 |


Spread: round 2 0.076, round 1 0.081, Betfair close 0.087. Gap to Betfair close 4.0 pp, correlation 0.81.


## Betting simulation

| model | league | price | bets | share_of_matches | avg_model_EV | profit_units | ROI | ROI_ci_low | ROI_ci_high |
|---|---|---|---|---|---|---|---|---|---|
| round 2 (locked) | ALL | Betfair early | 1021 | 0.3892 | 0.0977 | -74.5670 | -0.0730 | -0.1430 | -0.0048 |
| round 2 (locked) | ALL | Betfair close | 1788 | 0.6340 | 0.1115 | -68.9178 | -0.0385 | -0.0901 | 0.0104 |
| round 2 (locked) | La Liga 2 | Betfair early | 308 | 0.3919 | 0.1055 | -32.2866 | -0.1048 | -0.2295 | 0.0185 |
| round 2 (locked) | La Liga 2 | Betfair close | 600 | 0.6682 | 0.1228 | -32.7458 | -0.0546 | -0.1481 | 0.0386 |
| round 2 (locked) | Serie B | Betfair early | 227 | 0.3197 | 0.1013 | -9.1370 | -0.0403 | -0.1901 | 0.1102 |
| round 2 (locked) | Serie B | Betfair close | 460 | 0.6225 | 0.1081 | 16.5882 | 0.0361 | -0.0697 | 0.1444 |
| round 2 (locked) | 2. Bundesliga | Betfair early | 223 | 0.3838 | 0.0845 | -16.5210 | -0.0741 | -0.1988 | 0.0614 |
| round 2 (locked) | 2. Bundesliga | Betfair close | 360 | 0.6091 | 0.1049 | -19.3044 | -0.0536 | -0.1607 | 0.0504 |
| round 2 (locked) | Ligue 2 | Betfair early | 263 | 0.4817 | 0.0969 | -16.6224 | -0.0632 | -0.1957 | 0.0748 |
| round 2 (locked) | Ligue 2 | Betfair close | 368 | 0.6216 | 0.1038 | -33.4558 | -0.0909 | -0.2017 | 0.0251 |
| round 1 B | ALL | Betfair early | 1078 | 0.4110 | 0.1010 | -64.2316 | -0.0596 | -0.1256 | 0.0027 |
| round 1 B | ALL | Betfair close | 1820 | 0.6454 | 0.1145 | -49.8426 | -0.0274 | -0.0760 | 0.0225 |
| round 1 B | La Liga 2 | Betfair early | 319 | 0.4059 | 0.1026 | -40.6170 | -0.1273 | -0.2447 | -0.0144 |
| round 1 B | La Liga 2 | Betfair close | 619 | 0.6893 | 0.1202 | -19.7102 | -0.0318 | -0.1207 | 0.0584 |
| round 1 B | Serie B | Betfair early | 254 | 0.3577 | 0.0973 | -8.9258 | -0.0351 | -0.1766 | 0.0970 |
| round 1 B | Serie B | Betfair close | 456 | 0.6171 | 0.1047 | 2.6930 | 0.0059 | -0.0963 | 0.1113 |
| round 1 B | 2. Bundesliga | Betfair early | 231 | 0.3976 | 0.0990 | -2.2040 | -0.0095 | -0.1552 | 0.1277 |
| round 1 B | 2. Bundesliga | Betfair close | 375 | 0.6345 | 0.1137 | 8.3504 | 0.0223 | -0.0893 | 0.1322 |
| round 1 B | Ligue 2 | Betfair early | 274 | 0.5018 | 0.1044 | -12.4848 | -0.0456 | -0.1699 | 0.0795 |
| round 1 B | Ligue 2 | Betfair close | 370 | 0.6250 | 0.1176 | -41.1758 | -0.1113 | -0.2240 | -0.0021 |


**Round 2 vs Betfair close: +0.0078 - does not pass the bar.**

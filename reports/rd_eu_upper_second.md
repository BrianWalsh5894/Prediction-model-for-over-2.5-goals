# v3-eu-upper-second - R&D round 1 (second divisions as targets)

La Liga 2, Serie B, 2. Bundesliga, Ligue 2. 6-fold walk-forward (2020/21-2025/26). `diff` < 0 = model better; 95% bootstrap interval.

Success bar set before running: within +0.002 of the Betfair close.


## vs Bookmaker average

| league | model | model_ll | market_ll | diff | ci_low | ci_high | seasons_beating_market | n |
|---|---|---|---|---|---|---|---|---|
| ALL | A. Poisson 2nd div + DC per country | 0.6826 | 0.6749 | 0.0077 | 0.0056 | 0.0099 | 0/6 | 9015 |
| ALL | B. Poisson both divs + DC per country | 0.6822 | 0.6749 | 0.0073 | 0.0053 | 0.0094 | 0/6 | 9015 |
| ALL | C. Poisson 2nd div + DC 2nd div only | 0.6833 | 0.6749 | 0.0084 | 0.0062 | 0.0106 | 0/6 | 9015 |
| La Liga 2 | A. Poisson 2nd div + DC per country | 0.6731 | 0.6647 | 0.0085 | 0.0048 | 0.0124 | 0/6 | 2772 |
| La Liga 2 | B. Poisson both divs + DC per country | 0.6728 | 0.6647 | 0.0081 | 0.0043 | 0.0119 | 0/6 | 2772 |
| La Liga 2 | C. Poisson 2nd div + DC 2nd div only | 0.6734 | 0.6647 | 0.0087 | 0.0045 | 0.0127 | 0/6 | 2772 |
| Serie B | A. Poisson 2nd div + DC per country | 0.6931 | 0.6855 | 0.0076 | 0.0034 | 0.0118 | 1/6 | 2277 |
| Serie B | B. Poisson both divs + DC per country | 0.6935 | 0.6855 | 0.0079 | 0.0039 | 0.0123 | 1/6 | 2277 |
| Serie B | C. Poisson 2nd div + DC 2nd div only | 0.6947 | 0.6855 | 0.0092 | 0.0046 | 0.0139 | 1/6 | 2277 |
| 2. Bundesliga | A. Poisson 2nd div + DC per country | 0.6812 | 0.6729 | 0.0083 | 0.0040 | 0.0126 | 1/6 | 1836 |
| 2. Bundesliga | B. Poisson both divs + DC per country | 0.6798 | 0.6729 | 0.0069 | 0.0027 | 0.0112 | 1/6 | 1836 |
| 2. Bundesliga | C. Poisson 2nd div + DC 2nd div only | 0.6806 | 0.6729 | 0.0077 | 0.0033 | 0.0122 | 2/6 | 1836 |
| Ligue 2 | A. Poisson 2nd div + DC per country | 0.6851 | 0.6787 | 0.0064 | 0.0016 | 0.0111 | 1/6 | 2130 |
| Ligue 2 | B. Poisson both divs + DC per country | 0.6847 | 0.6787 | 0.0060 | 0.0015 | 0.0107 | 2/6 | 2130 |
| Ligue 2 | C. Poisson 2nd div + DC 2nd div only | 0.6864 | 0.6787 | 0.0077 | 0.0025 | 0.0129 | 1/6 | 2130 |


## vs Betfair close

| league | model | model_ll | market_ll | diff | ci_low | ci_high | seasons_beating_market | n |
|---|---|---|---|---|---|---|---|---|
| ALL | A. Poisson 2nd div + DC per country | 0.6920 | 0.6819 | 0.0100 | 0.0059 | 0.0140 | 0/2 | 2820 |
| ALL | B. Poisson both divs + DC per country | 0.6912 | 0.6819 | 0.0093 | 0.0049 | 0.0133 | 0/2 | 2820 |
| ALL | C. Poisson 2nd div + DC 2nd div only | 0.6935 | 0.6819 | 0.0116 | 0.0071 | 0.0160 | 0/2 | 2820 |
| La Liga 2 | A. Poisson 2nd div + DC per country | 0.6932 | 0.6818 | 0.0114 | 0.0038 | 0.0189 | 0/2 | 898 |
| La Liga 2 | B. Poisson both divs + DC per country | 0.6919 | 0.6818 | 0.0102 | 0.0024 | 0.0178 | 0/2 | 898 |
| La Liga 2 | C. Poisson 2nd div + DC 2nd div only | 0.6951 | 0.6818 | 0.0133 | 0.0052 | 0.0217 | 0/2 | 898 |
| Serie B | A. Poisson 2nd div + DC per country | 0.7009 | 0.6921 | 0.0087 | 0.0014 | 0.0159 | 0/2 | 739 |
| Serie B | B. Poisson both divs + DC per country | 0.7008 | 0.6921 | 0.0086 | 0.0013 | 0.0159 | 0/2 | 739 |
| Serie B | C. Poisson 2nd div + DC 2nd div only | 0.7028 | 0.6921 | 0.0107 | 0.0022 | 0.0194 | 0/2 | 739 |
| 2. Bundesliga | A. Poisson 2nd div + DC per country | 0.6746 | 0.6701 | 0.0046 | -0.0041 | 0.0131 | 0/2 | 591 |
| 2. Bundesliga | B. Poisson both divs + DC per country | 0.6739 | 0.6701 | 0.0038 | -0.0050 | 0.0120 | 0/2 | 591 |
| 2. Bundesliga | C. Poisson 2nd div + DC 2nd div only | 0.6772 | 0.6701 | 0.0071 | -0.0015 | 0.0157 | 0/2 | 591 |
| Ligue 2 | A. Poisson 2nd div + DC per country | 0.6965 | 0.6814 | 0.0151 | 0.0054 | 0.0250 | 0/2 | 592 |
| Ligue 2 | B. Poisson both divs + DC per country | 0.6955 | 0.6814 | 0.0141 | 0.0049 | 0.0232 | 0/2 | 592 |
| Ligue 2 | C. Poisson 2nd div + DC 2nd div only | 0.6958 | 0.6814 | 0.0145 | 0.0038 | 0.0251 | 0/2 | 592 |


## How far off (2024/25 + 2025/26, 2820 matches)

- A. Poisson 2nd div + DC per country: average gap 4.3 pp, correlation 0.79
- B. Poisson both divs + DC per country: average gap 4.2 pp, correlation 0.79
- C. Poisson 2nd div + DC 2nd div only: average gap 4.6 pp, correlation 0.76

## Betting simulation (flat 1 unit, EV >= 3% after 2% commission)

| model | league | price | bets | share_of_matches | avg_model_EV | profit_units | ROI | ROI_ci_low | ROI_ci_high |
|---|---|---|---|---|---|---|---|---|---|
| A. Poisson 2nd div + DC per country | ALL | Betfair early | 1080 | 0.4117 | 0.1021 | -100.5984 | -0.0931 | -0.1570 | -0.0312 |
| A. Poisson 2nd div + DC per country | ALL | Betfair close | 1808 | 0.6411 | 0.1160 | -85.4394 | -0.0473 | -0.0976 | 0.0046 |
| A. Poisson 2nd div + DC per country | La Liga 2 | Betfair early | 311 | 0.3957 | 0.1022 | -43.5656 | -0.1401 | -0.2573 | -0.0175 |
| A. Poisson 2nd div + DC per country | La Liga 2 | Betfair close | 601 | 0.6693 | 0.1203 | -33.1914 | -0.0552 | -0.1407 | 0.0365 |
| A. Poisson 2nd div + DC per country | Serie B | Betfair early | 247 | 0.3479 | 0.0976 | -18.7752 | -0.0760 | -0.2176 | 0.0619 |
| A. Poisson 2nd div + DC per country | Serie B | Betfair close | 453 | 0.6130 | 0.1035 | -9.4108 | -0.0208 | -0.1167 | 0.0865 |
| A. Poisson 2nd div + DC per country | 2. Bundesliga | Betfair early | 239 | 0.4114 | 0.1026 | -20.4094 | -0.0854 | -0.2262 | 0.0518 |
| A. Poisson 2nd div + DC per country | 2. Bundesliga | Betfair close | 379 | 0.6413 | 0.1183 | 9.9936 | 0.0264 | -0.0922 | 0.1371 |
| A. Poisson 2nd div + DC per country | Ligue 2 | Betfair early | 283 | 0.5183 | 0.1056 | -17.8482 | -0.0631 | -0.1881 | 0.0592 |
| A. Poisson 2nd div + DC per country | Ligue 2 | Betfair close | 375 | 0.6334 | 0.1219 | -52.8308 | -0.1409 | -0.2458 | -0.0384 |
| B. Poisson both divs + DC per country | ALL | Betfair early | 1078 | 0.4110 | 0.1010 | -64.2316 | -0.0596 | -0.1248 | 0.0055 |
| B. Poisson both divs + DC per country | ALL | Betfair close | 1820 | 0.6454 | 0.1145 | -49.8426 | -0.0274 | -0.0767 | 0.0226 |
| B. Poisson both divs + DC per country | La Liga 2 | Betfair early | 319 | 0.4059 | 0.1026 | -40.6170 | -0.1273 | -0.2440 | -0.0042 |
| B. Poisson both divs + DC per country | La Liga 2 | Betfair close | 619 | 0.6893 | 0.1202 | -19.7102 | -0.0318 | -0.1233 | 0.0573 |
| B. Poisson both divs + DC per country | Serie B | Betfair early | 254 | 0.3577 | 0.0973 | -8.9258 | -0.0351 | -0.1657 | 0.1087 |
| B. Poisson both divs + DC per country | Serie B | Betfair close | 456 | 0.6171 | 0.1047 | 2.6930 | 0.0059 | -0.1008 | 0.1171 |
| B. Poisson both divs + DC per country | 2. Bundesliga | Betfair early | 231 | 0.3976 | 0.0990 | -2.2040 | -0.0095 | -0.1486 | 0.1244 |
| B. Poisson both divs + DC per country | 2. Bundesliga | Betfair close | 375 | 0.6345 | 0.1137 | 8.3504 | 0.0223 | -0.0858 | 0.1396 |
| B. Poisson both divs + DC per country | Ligue 2 | Betfair early | 274 | 0.5018 | 0.1044 | -12.4848 | -0.0456 | -0.1701 | 0.0782 |
| B. Poisson both divs + DC per country | Ligue 2 | Betfair close | 370 | 0.6250 | 0.1176 | -41.1758 | -0.1113 | -0.2099 | -0.0057 |
| C. Poisson 2nd div + DC 2nd div only | ALL | Betfair early | 1146 | 0.4369 | 0.1090 | -98.9922 | -0.0864 | -0.1449 | -0.0242 |
| C. Poisson 2nd div + DC 2nd div only | ALL | Betfair close | 1867 | 0.6621 | 0.1229 | -90.8606 | -0.0487 | -0.0991 | 0.0005 |
| C. Poisson 2nd div + DC 2nd div only | La Liga 2 | Betfair early | 338 | 0.4300 | 0.1071 | -33.5446 | -0.0992 | -0.2099 | 0.0212 |
| C. Poisson 2nd div + DC 2nd div only | La Liga 2 | Betfair close | 623 | 0.6938 | 0.1261 | -42.8328 | -0.0688 | -0.1565 | 0.0267 |
| C. Poisson 2nd div + DC 2nd div only | Serie B | Betfair early | 265 | 0.3732 | 0.1098 | -11.0148 | -0.0416 | -0.1766 | 0.0886 |
| C. Poisson 2nd div + DC 2nd div only | Serie B | Betfair close | 473 | 0.6401 | 0.1147 | -3.4744 | -0.0073 | -0.1098 | 0.0926 |
| C. Poisson 2nd div + DC 2nd div only | 2. Bundesliga | Betfair early | 236 | 0.4062 | 0.1030 | -35.5904 | -0.1508 | -0.2835 | 0.0010 |
| C. Poisson 2nd div + DC 2nd div only | 2. Bundesliga | Betfair close | 381 | 0.6447 | 0.1174 | -5.5410 | -0.0145 | -0.1225 | 0.0962 |
| C. Poisson 2nd div + DC 2nd div only | Ligue 2 | Betfair early | 307 | 0.5623 | 0.1149 | -18.8424 | -0.0614 | -0.1721 | 0.0515 |
| C. Poisson 2nd div + DC 2nd div only | Ligue 2 | Betfair close | 390 | 0.6588 | 0.1332 | -39.0124 | -0.1000 | -0.2051 | 0.0036 |


**Best vs Betfair close: B. Poisson both divs + DC per country, gap +0.0093 - does not pass the bar.**

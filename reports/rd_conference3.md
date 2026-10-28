# National League - R&D round 3 (round 1 C + Bzzoiro shots on target and xG)

No tuning. Variants fixed before running. Test 2024/25 - 2025/26 (third look at these seasons).

Bar: within +0.002 of the Betfair close.


## vs Bookmaker average

| model | model_ll | market_ll | diff | ci_low | ci_high | n |
|---|---|---|---|---|---|---|
| C (round 1) | 0.6879 | 0.6815 | 0.0065 | -0.0008 | 0.0131 | 1092 |
| C+SOT | 0.6863 | 0.6815 | 0.0049 | -0.0017 | 0.0112 | 1092 |
| C+SOT+xG | 0.6856 | 0.6815 | 0.0042 | -0.0028 | 0.0110 | 1092 |


## vs Betfair close

| model | model_ll | market_ll | diff | ci_low | ci_high | n |
|---|---|---|---|---|---|---|
| C (round 1) | 0.6878 | 0.6817 | 0.0061 | -0.0017 | 0.0138 | 1087 |
| C+SOT | 0.6865 | 0.6817 | 0.0048 | -0.0022 | 0.0120 | 1087 |
| C+SOT+xG | 0.6859 | 0.6817 | 0.0042 | -0.0028 | 0.0109 | 1087 |


## Gain from new data vs round 1 C

- C+SOT: -0.0016 (95% interval -0.0035 to +0.0002)
- C+SOT+xG: -0.0023 (95% interval -0.0056 to +0.0006)

## By team type

| team_type | n | C (round 1) | C+SOT | C+SOT+xG |
|---|---|---|---|---|
| established | 605.0000 | 0.0030 | 0.0030 | 0.0027 |
| promoted_from_N/S | 341.0000 | 0.0080 | 0.0051 | 0.0053 |
| relegated_from_L2 | 146.0000 | 0.0175 | 0.0118 | 0.0074 |


## Betting simulation

| model | price | bets | share_of_matches | avg_model_EV | profit_units | ROI | ROI_ci_low | ROI_ci_high |
|---|---|---|---|---|---|---|---|---|
| C (round 1) | Betfair early | 373 | 0.3600 | 0.1117 | -24.0026 | -0.0644 | -0.1745 | 0.0468 |
| C (round 1) | Betfair close | 664 | 0.6109 | 0.1354 | -16.2062 | -0.0244 | -0.1102 | 0.0593 |
| C+SOT | Betfair early | 346 | 0.3340 | 0.1008 | -21.8492 | -0.0631 | -0.1738 | 0.0448 |
| C+SOT | Betfair close | 651 | 0.5989 | 0.1213 | 13.6542 | 0.0210 | -0.0628 | 0.1067 |
| C+SOT+xG | Betfair early | 344 | 0.3320 | 0.0978 | -17.9370 | -0.0521 | -0.1647 | 0.0514 |
| C+SOT+xG | Betfair close | 632 | 0.5814 | 0.1182 | 2.8586 | 0.0045 | -0.0778 | 0.0836 |


**Best vs Betfair close: C+SOT+xG, gap +0.0042 - does not pass the bar.**

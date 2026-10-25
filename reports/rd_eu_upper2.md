# v3-eu-upper improvement round 1 - second divisions + xG

Top-division matches only. 6-fold walk-forward. `diff` < 0 = model better; 95% bootstrap interval.

Success bar set before running: within +0.002 of the Betfair close.


## vs Bookmaker average

| league | model | model_ll | market_ll | diff | ci_low | ci_high | seasons_beating_market | n |
|---|---|---|---|---|---|---|---|---|
| ALL | A. Round-1 baseline (top divisions only) | 0.6768 | 0.6713 | 0.0055 | 0.0034 | 0.0076 | 0/6 | 8452 |
| ALL | B. + 2nd divisions (Poisson on both) | 0.6765 | 0.6713 | 0.0052 | 0.0032 | 0.0071 | 0/6 | 8452 |
| ALL | C. + 2nd divisions (Poisson on top only) | 0.6766 | 0.6713 | 0.0053 | 0.0034 | 0.0072 | 0/6 | 8452 |
| ALL | D. + 2nd divisions + xG | 0.6762 | 0.6713 | 0.0049 | 0.0031 | 0.0068 | 0/6 | 8452 |
| ALL | E. Compact: xG + Elo + DC | 0.6765 | 0.6713 | 0.0052 | 0.0034 | 0.0071 | 0/6 | 8452 |
| La Liga | A. Round-1 baseline (top divisions only) | 0.6761 | 0.6711 | 0.0050 | 0.0010 | 0.0090 | 0/6 | 2280 |
| La Liga | B. + 2nd divisions (Poisson on both) | 0.6749 | 0.6711 | 0.0038 | 0.0001 | 0.0074 | 1/6 | 2280 |
| La Liga | C. + 2nd divisions (Poisson on top only) | 0.6749 | 0.6711 | 0.0038 | 0.0003 | 0.0072 | 0/6 | 2280 |
| La Liga | D. + 2nd divisions + xG | 0.6747 | 0.6711 | 0.0036 | 0.0003 | 0.0071 | 0/6 | 2280 |
| La Liga | E. Compact: xG + Elo + DC | 0.6769 | 0.6711 | 0.0059 | 0.0022 | 0.0094 | 0/6 | 2280 |
| Serie A | A. Round-1 baseline (top divisions only) | 0.6882 | 0.6832 | 0.0050 | 0.0013 | 0.0090 | 1/6 | 2278 |
| Serie A | B. + 2nd divisions (Poisson on both) | 0.6874 | 0.6832 | 0.0042 | 0.0005 | 0.0079 | 2/6 | 2278 |
| Serie A | C. + 2nd divisions (Poisson on top only) | 0.6880 | 0.6832 | 0.0047 | 0.0010 | 0.0088 | 1/6 | 2278 |
| Serie A | D. + 2nd divisions + xG | 0.6875 | 0.6832 | 0.0042 | 0.0006 | 0.0080 | 2/6 | 2278 |
| Serie A | E. Compact: xG + Elo + DC | 0.6870 | 0.6832 | 0.0038 | 0.0000 | 0.0078 | 1/6 | 2278 |
| Bundesliga | A. Round-1 baseline (top divisions only) | 0.6537 | 0.6497 | 0.0041 | -0.0005 | 0.0087 | 1/6 | 1836 |
| Bundesliga | B. + 2nd divisions (Poisson on both) | 0.6547 | 0.6497 | 0.0051 | 0.0012 | 0.0092 | 0/6 | 1836 |
| Bundesliga | C. + 2nd divisions (Poisson on top only) | 0.6545 | 0.6497 | 0.0048 | 0.0008 | 0.0087 | 0/6 | 1836 |
| Bundesliga | D. + 2nd divisions + xG | 0.6543 | 0.6497 | 0.0047 | 0.0005 | 0.0087 | 0/6 | 1836 |
| Bundesliga | E. Compact: xG + Elo + DC | 0.6554 | 0.6497 | 0.0057 | 0.0021 | 0.0092 | 1/6 | 1836 |
| Ligue 1 | A. Round-1 baseline (top divisions only) | 0.6855 | 0.6776 | 0.0079 | 0.0038 | 0.0120 | 1/6 | 2058 |
| Ligue 1 | B. + 2nd divisions (Poisson on both) | 0.6856 | 0.6776 | 0.0080 | 0.0038 | 0.0123 | 1/6 | 2058 |
| Ligue 1 | C. + 2nd divisions (Poisson on top only) | 0.6855 | 0.6776 | 0.0078 | 0.0038 | 0.0118 | 1/6 | 2058 |
| Ligue 1 | D. + 2nd divisions + xG | 0.6849 | 0.6776 | 0.0073 | 0.0034 | 0.0112 | 1/6 | 2058 |
| Ligue 1 | E. Compact: xG + Elo + DC | 0.6834 | 0.6776 | 0.0057 | 0.0022 | 0.0096 | 1/6 | 2058 |


## vs Betfair close

| league | model | model_ll | market_ll | diff | ci_low | ci_high | seasons_beating_market | n |
|---|---|---|---|---|---|---|---|---|
| ALL | A. Round-1 baseline (top divisions only) | 0.6747 | 0.6660 | 0.0086 | 0.0042 | 0.0131 | 0/2 | 2655 |
| ALL | B. + 2nd divisions (Poisson on both) | 0.6745 | 0.6660 | 0.0085 | 0.0044 | 0.0126 | 0/2 | 2655 |
| ALL | C. + 2nd divisions (Poisson on top only) | 0.6744 | 0.6660 | 0.0084 | 0.0040 | 0.0127 | 0/2 | 2655 |
| ALL | D. + 2nd divisions + xG | 0.6741 | 0.6660 | 0.0081 | 0.0038 | 0.0121 | 0/2 | 2655 |
| ALL | E. Compact: xG + Elo + DC | 0.6745 | 0.6660 | 0.0085 | 0.0042 | 0.0129 | 0/2 | 2655 |
| La Liga | A. Round-1 baseline (top divisions only) | 0.6746 | 0.6647 | 0.0099 | 0.0009 | 0.0190 | 0/2 | 729 |
| La Liga | B. + 2nd divisions (Poisson on both) | 0.6743 | 0.6647 | 0.0096 | 0.0009 | 0.0185 | 0/2 | 729 |
| La Liga | C. + 2nd divisions (Poisson on top only) | 0.6736 | 0.6647 | 0.0089 | 0.0004 | 0.0173 | 0/2 | 729 |
| La Liga | D. + 2nd divisions + xG | 0.6732 | 0.6647 | 0.0085 | 0.0001 | 0.0172 | 0/2 | 729 |
| La Liga | E. Compact: xG + Elo + DC | 0.6741 | 0.6647 | 0.0094 | -0.0006 | 0.0190 | 0/2 | 729 |
| Serie A | A. Round-1 baseline (top divisions only) | 0.6915 | 0.6884 | 0.0032 | -0.0048 | 0.0112 | 1/2 | 739 |
| Serie A | B. + 2nd divisions (Poisson on both) | 0.6895 | 0.6884 | 0.0012 | -0.0066 | 0.0091 | 1/2 | 739 |
| Serie A | C. + 2nd divisions (Poisson on top only) | 0.6902 | 0.6884 | 0.0019 | -0.0056 | 0.0098 | 1/2 | 739 |
| Serie A | D. + 2nd divisions + xG | 0.6900 | 0.6884 | 0.0016 | -0.0065 | 0.0096 | 1/2 | 739 |
| Serie A | E. Compact: xG + Elo + DC | 0.6886 | 0.6884 | 0.0003 | -0.0083 | 0.0086 | 1/2 | 739 |
| Bundesliga | A. Round-1 baseline (top divisions only) | 0.6466 | 0.6373 | 0.0093 | -0.0006 | 0.0190 | 0/2 | 594 |
| Bundesliga | B. + 2nd divisions (Poisson on both) | 0.6493 | 0.6373 | 0.0120 | 0.0039 | 0.0205 | 0/2 | 594 |
| Bundesliga | C. + 2nd divisions (Poisson on top only) | 0.6488 | 0.6373 | 0.0116 | 0.0030 | 0.0197 | 0/2 | 594 |
| Bundesliga | D. + 2nd divisions + xG | 0.6484 | 0.6373 | 0.0111 | 0.0028 | 0.0193 | 0/2 | 594 |
| Bundesliga | E. Compact: xG + Elo + DC | 0.6503 | 0.6373 | 0.0130 | 0.0056 | 0.0209 | 0/2 | 594 |
| Ligue 1 | A. Round-1 baseline (top divisions only) | 0.6820 | 0.6687 | 0.0133 | 0.0038 | 0.0225 | 0/2 | 593 |
| Ligue 1 | B. + 2nd divisions (Poisson on both) | 0.6812 | 0.6687 | 0.0125 | 0.0034 | 0.0211 | 0/2 | 593 |
| Ligue 1 | C. + 2nd divisions (Poisson on top only) | 0.6815 | 0.6687 | 0.0128 | 0.0038 | 0.0218 | 0/2 | 593 |
| Ligue 1 | D. + 2nd divisions + xG | 0.6812 | 0.6687 | 0.0125 | 0.0036 | 0.0209 | 0/2 | 593 |
| Ligue 1 | E. Compact: xG + Elo + DC | 0.6818 | 0.6687 | 0.0132 | 0.0044 | 0.0214 | 0/2 | 593 |


**Best vs Betfair close: D. + 2nd divisions + xG, gap +0.0081 - does not pass the bar.**

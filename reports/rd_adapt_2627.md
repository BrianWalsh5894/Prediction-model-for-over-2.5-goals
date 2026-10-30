# Faster-adapting variants vs the frozen live models - 2026/27 so far

Variants fixed before testing: A = Dixon-Coles time decay 0.002/day; B = Dixon-Coles 1-year window; C = frozen + per-league shift from the previous 200 matches. 2026/27 never used for tuning.

| group | matches | actual_over | BASE_avg_p | BASE_ll | A_avg_p | A_ll | B_avg_p | B_ll | C_avg_p | C_ll | n_Bookmaker average | Bookmaker average_avg_p | Bookmaker average_ll | BASE_vs_Bookmaker average | A_vs_Bookmaker average | B_vs_Bookmaker average | C_vs_Bookmaker average | n_Betfair close | Betfair close_avg_p | Betfair close_ll | BASE_vs_Betfair close | A_vs_Betfair close | B_vs_Betfair close | C_vs_Betfair close | A_minus_BASE | A_minus_BASE_ci | B_minus_BASE | B_minus_BASE_ci | C_minus_BASE | C_minus_BASE_ci |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 842 | 0.5594 | 0.5166 | 0.6726 | 0.5154 | 0.6724 | 0.5142 | 0.6741 | 0.5492 | 0.6707 | 842 | 0.5416 | 0.6653 | 0.0072 | 0.0071 | 0.0088 | 0.0053 | 842 | 0.5497 | 0.6623 | 0.0103 | 0.0101 | 0.0118 | 0.0084 | -0.0002 | -0.0025 to +0.0021 | 0.0015 | -0.0039 to +0.0068 | -0.0019 | -0.0072 to +0.0038 |
| lower | 294 | 0.4932 | 0.4942 | 0.6836 | 0.4903 | 0.6812 | 0.4872 | 0.6820 | 0.5245 | 0.6869 | 294 | 0.5258 | 0.6806 | 0.0031 | 0.0007 | 0.0014 | 0.0064 | 294 | 0.5320 | 0.6828 | 0.0008 | -0.0015 | -0.0008 | 0.0041 | -0.0024 | -0.0060 to +0.0012 | -0.0017 | -0.0096 to +0.0067 | 0.0033 | -0.0043 to +0.0111 |
| upper | 224 | 0.5670 | 0.5139 | 0.6882 | 0.5121 | 0.6935 | 0.5136 | 0.6974 | 0.5629 | 0.6809 | 224 | 0.5380 | 0.6848 | 0.0034 | 0.0087 | 0.0126 | -0.0039 | 224 | 0.5425 | 0.6820 | 0.0062 | 0.0115 | 0.0154 | -0.0011 | 0.0053 | +0.0012 to +0.0096 | 0.0093 | -0.0000 to +0.0191 | -0.0073 | -0.0234 to +0.0082 |
| eu_upper | 324 | 0.6142 | 0.5387 | 0.6518 | 0.5404 | 0.6498 | 0.5390 | 0.6509 | 0.5621 | 0.6489 | 324 | 0.5585 | 0.6380 | 0.0137 | 0.0118 | 0.0128 | 0.0108 | 324 | 0.5706 | 0.6300 | 0.0217 | 0.0198 | 0.0209 | 0.0189 | -0.0019 | -0.0060 to +0.0022 | -0.0009 | -0.0104 to +0.0086 | -0.0029 | -0.0100 to +0.0044 |

**Best variant overall: C.**

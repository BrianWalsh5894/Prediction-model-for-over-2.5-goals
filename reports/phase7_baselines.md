# Phase 7 - Baselines

Evaluated on VALIDATION_SEASONS only (['2023/24'], 2036 matches). TRAIN_SEASONS (['2017/18', '2018/19', '2019/20', '2020/21', '2021/22', '2022/23'], 11952 matches) used to fit Baseline A and the Logistic Regression. TEST_SEASONS ['2024/25', '2025/26'] were not loaded by this script at all.

## Results (sorted by Log Loss, lower is better)

| model | log_loss | brier_score | n_matches |
|---|---|---|---|
| Baseline C - market (de-margined avg odds) | 0.6751 | 0.2411 | 2036 |
| Baseline B - Poisson goals model | 0.6914 | 0.2488 | 2035 |
| Logistic Regression (engineered features) | 0.6914 | 0.2483 | 2036 |
| Baseline A - league historical rate | 0.6966 | 0.2517 | 2036 |

**Brier Score** (also lower is better) tells the same story as Log Loss here but penalises confident wrong predictions less harshly - included per Phase 0's primary metric pair.

## Notes

- Baseline A is fit on TRAIN only (per-league Over 2.5 rate), then applied as a constant per league to every validation match.

- Baseline B needs no separate fitting: it converts the Phase 5 `combined_home_goal_rate`/`combined_away_goal_rate` features (already pre-match, shift-based) into a Poisson P(total goals >= 3).

- Baseline C uses `avg_over_2_5`/`avg_under_2_5` (bookmaker average), not Betfair Exchange, because BFE odds don't exist yet in the 2023/24 validation season (data_dictionary.md) - the real BFE closing-price comparison only becomes possible in Phase 9/10 on the TEST seasons.

- Logistic Regression excludes all odds columns from its feature set - it only sees the Phase 5 engineered features, never the market's own price, so beating Baseline C with it is a genuine test of the feature engineering, not the model re-deriving the market's answer from the market's own odds.


**Does anything beat the market here?** The best non-market model does NOT beat Baseline C on Log Loss in this validation run. Per Phase 0 principle 6 ("assume any edge above ~5% is a bug until proven otherwise"), a result where Logistic Regression clearly beats the market this early, with no tuning, would itself be a reason to go back and check for a leak before believing it - not a reason to celebrate.

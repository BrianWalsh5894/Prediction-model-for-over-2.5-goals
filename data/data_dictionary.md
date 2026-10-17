# Data Dictionary — Raw Source Files

Source: football-data.co.uk, downloaded 2026-10-17
Divisions: E0 (Premier League), E1 (Championship), E2 (League One), E3 (League Two)
Seasons: 2017/18 through 2025/26

## Odds coverage by season (applies across all four divisions)

| Seasons              | Over/Under 2.5 odds available          | Betfair Exchange odds |
|-----------------------|-----------------------------------------|------------------------|
| 2017/18 - 2018/19      | None (no Avg>2.5, no BFE)               | No                     |
| 2019/20 - 2023/24      | Avg>2.5 (de-margined bookmaker average) | No                     |
| 2024/25 - 2025/26      | Avg>2.5 and BFE>2.5                     | Yes                    |

## Implication for project phases

- All 9 seasons used for training the goals/Over 2.5 model (Phase 5-8) — odds
  coverage doesn't matter here, only match results and stats.
- Baseline C (market de-margined odds, Phase 7) only usable from 2019/20 onward.
- Phase 10 (Betfair value/EV backtest) only possible on 2024/25 and 2025/26 —
  which happens to match the TEST_SEASONS already set in config.py. This is
  intentional: it's the only overlap where real BFE prices exist.
- 2017/18 and 2018/19 have no odds-based baseline comparison available —
  kept in training data regardless, since match/goal data matters more than
  odds coverage for the core model.

## Core fields present in every file

Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, Referee,
HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR

## COVID-affected seasons

Checked directly (home win rate, avg total goals per match) across all four
divisions combined:

| Season   | Home win rate | Avg total goals | Note                          |
|----------|----------------|------------------|--------------------------------|
| 2018/19  | 0.440          | 2.660            | baseline (normal, fans present)|
| 2019/20  | 0.435          | 2.620            | partial - stoppage then finished behind closed doors; barely moved |
| 2020/21  | 0.401          | 2.479            | fully behind closed doors - clear outlier, ~8% relative drop in home win rate, ~0.15-0.18 fewer goals/match |
| 2021/22  | 0.436          | 2.605            | fans back - snapped straight back to normal |

DECISION: keep 2019/20 and 2020/21 in training data (data volume matters
more than odds/market coverage for this model), but add a
`behind_closed_doors` binary feature in Phase 5 so the model can learn
2020/21 as a different regime rather than treating it as a normal season.
Partial-impact 2019/20 may need a date-based split rather than a simple
season-level flag, since only part of that season was affected - decide
this properly when we get to Phase 5.

## xG (expected goals)

DEFERRED, not skipped. Understat (free xG source) covers Premier League
well but is patchy/absent for Championship, League One, League Two —
inconsistent across the four divisions this model targets.

Decision: build first model on shots/shots-on-target (available for all
divisions, all seasons) as the attacking-quality proxy. Revisit xG as a
Premier-League-only enhancement IF Phase 9's "performance by league"
breakdown shows PL underperforming relative to the lower divisions,
suggesting shot-count alone isn't capturing enough signal at that level.
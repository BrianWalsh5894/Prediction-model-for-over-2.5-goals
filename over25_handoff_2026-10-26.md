# Over 2.5 goals model: handoff report (26 Oct 2026)

Read this first in a new conversation. It covers what exists, what has been proven, what is live, and what's next.

## 1. Project in one paragraph
A local machine-learning system (Python, SQLite, Anaconda env `over25`) that predicts P(Over 2.5 goals) per match. It tests honestly whether those probabilities beat the betting market, using the Betfair Exchange closing price as the benchmark. All R&D is done with time-based walk-forward testing. Three models are now **frozen and paper-tracked live**. No real money until a model proves itself (rules in section 7).

- Project folder: `C:\Users\brian\Desktop\Projects\over2.5`. GitHub: `BrianWalsh5894/Prediction-model-for-over-2.5-goals`.
- Data: football-data.co.uk (results, shots, odds incl. Betfair early + closing) and Understat (xG, top 5 leagues only).
- `data/`, `*.db`, `models/*.joblib`, `data/live/` and `reports/live_tracking.xlsx` are gitignored (kept private).

## 2. How we work (keep doing this)
1. Claude writes/edits a script, tests it on a staged copy of Brian's data, and copies it into `src/` on Brian's PC.
2. Brian runs it in the Anaconda Prompt (`conda activate over25`, `cd ...\over2.5\src`) and pastes the output.
3. The outputs are compared for an exact match. Then Brian runs the git commands himself.
4. Instructions must be explicit, step by step, with copy-paste commands.

**Known gotchas:**
- **VS Code must be closed (or auto-save off) while Claude updates files.** It has repeatedly saved old versions over new ones (config.py, weekly_predict.py, clean_data.py, rd_eu_upper2.py, etc.). After pushing files, Claude checks the sizes on the PC.
- **Close Excel** before running scripts that write a CSV/XLSX it has open. Claude saves filled odds sheets as `..._filled.csv` to avoid the clash.
- Football-Data publishes weekend results Sunday night or Monday. `weekly_predict.py` stops until last Saturday's results are in. Use `--force --only-date YYYY-MM-DD` only for teams that didn't play Saturday (e.g. Sunday/Monday games).

## 3. Live models (all frozen; paper only)
| Group (`--group`) | Leagues | Model | Fingerprint | Frozen |
|---|---|---|---|---|
| `lower` (default) | League One, League Two | **v3** | ca435d362d0a3369 | 25 Oct 2026 |
| `upper` | Premier League, Championship | **v3** (same model, separate record) | same | tracked from 25 Oct |
| `eu_upper` | La Liga, Serie A, Bundesliga, Ligue 1 | **v3-eu-upper** | 1aebf944ffe7cec7 | 26 Oct 2026 |

- **v3** = 50/50 blend of Poisson regression (103 features, alpha 1.3335, trained 2017/18–2025/26, 18,060 matches) and Dixon-Coles team ratings (re-fitted weekly on the last 3 years, rho correction, no time decay). Walk-forward log loss 0.6860.
- **v3-eu-upper** = the same recipe plus second-division history (La Liga 2, Serie B, 2. Bundesliga, Ligue 2) for promoted teams, plus 23 Understat xG features, with Dixon-Coles per country. Walk-forward 0.6762. It **failed** the pre-set +0.002 bar, so it's tracked as an experiment.
- Config: `config/config.py` → `LEAGUE_GROUPS`. Registry: `src/league_groups.py` (`MODEL_MODULES`). A new model = new version + new record. A tracked model never changes.

## 4. Weekly routine (also in `over25_cheatsheet.txt`)
```
python weekly_predict.py [--group lower|upper|eu_upper]   # stores predictions, writes data\live\odds_entry_<group>_<date>.csv
# fill betfair_over_odds, betfair_under_odds (blue BACK prices), checked_at (2026-10-31 14:05) -> save, close Excel
python weekly_bets.py ..\data\live\odds_entry_<group>_<date>.csv   # records prices, logs paper bets (EV >= 3% after 2% commission, 1 unit flat)
python weekly_settle.py        # after matches: settles bets, voids postponed (7 days), writes reports\live_tracking_<group>.md + reports\live_tracking.xlsx
python export_tracking_xlsx.py # refresh the spreadsheet on its own
```
- It predicts each team's NEXT match only. The first prediction and first price per match are kept forever.
- Matches that have already kicked off are skipped. Prices should be checked on matchday (two exceptions so far were priced the night before).
- The `eu_upper` group also refreshes Understat xG automatically.

## 5. Paper bets so far (all 1 unit)
| Date | Match | Group | Bet | Odds | Model EV | Result |
|---|---|---|---|---|---|---|
| 25 Oct | Liverpool v Brighton | upper | Under | 2.60 | +4.4% | won (unsettled in DB) |
| 25 Oct | Man United v Bournemouth | upper | Over | 1.88 | +5.6% | won (unsettled in DB) |
| 26 Oct | Walsall v Salford | lower | Under | 2.24 | +7.2% | pending |
| 26 Oct | Sevilla v Osasuna | eu_upper | Over | 2.34 | +10.6% | pending |
| 26 Oct | PSG v Lyon | eu_upper | Under | 3.40 | +15.6% | pending (very thin market, €236 matched) |

No-bet predictions logged on 25 Oct: Crystal Palace v Newcastle, Hull v Brentford, Sunderland v Leeds.
**Next action: run `python weekly_settle.py` on Tuesday** to settle all 5.

## 6. What R&D has proven (all verified on Brian's machine)
**England**
- Phase 9: locked Random Forest on TEST (2024/25–2025/26): 0.6904 vs Betfair close 0.6851. Doesn't beat the market.
- Rounds 1–3: best was the Poisson + Dixon-Coles blend at 0.6860 vs market 0.6825 (6-fold walk-forward). Time decay didn't help.
- Round 4: building on the market's own price made things worse; the early bookmaker price ≈ Betfair close.
- Per-league: every league loses to the bookmaker average over 6 seasons. League Two was −0.0010 vs Betfair close over 2 seasons (noise-level).
- Premier League + Understat xG: improves by 0.0006; still loses to the market.
- **Betting simulation** (v3 recipe, 2024/25–25/26, EV ≥ 3%): **about break-even**, +0.2% at the early price / +0.3% at close (CI about −4% to +5%). League Two +2.8%, League One −4.0%.

**Europe (top 4 leagues)**
- v3 recipe: +0.0055 vs bookmaker average, +0.0086 vs Betfair close. Pooled = per-country.
- + second divisions + xG (D): +0.0049 / +0.0081. Calibration (Platt / per-league shift) doesn't help.
- Model is on average 4.2 points from Betfair close (correlation 0.87). **Betting simulation: −3.6% (early) / −5.0% (close)**. Claimed EV averaged about +11%, so model EVs are inflated.

**Europe second divisions: v3-eu-upper-second R&D (26 Oct, `rd_eu_upper_second.py`, verified on Brian's machine)**
- Targets La Liga 2, Serie B, 2. Bundesliga, Ligue 2 (group name if ever frozen: `eu_upper_second`). v3 recipe, no xG (Understat has no second divisions).
- Best = B (Poisson trained on both divisions + Dixon-Coles per country): **+0.0073 vs bookmaker average, +0.0093 vs Betfair close** (CI +0.0049 to +0.0133). Bar +0.002 → **fails**. Worse than the top 4 leagues.
- 0/6 seasons beat the bookmaker average. Correlation with Betfair close 0.79. Closest league: 2. Bundesliga +0.0038 vs close; worst: Ligue 2 +0.0141.
- Betting simulation (2024/25–25/26): **−6.0% early (1,078 bets) / −2.8% close (1,821 bets)**; claimed EV about +10%.
- Teams promoted from the third tiers have no history (+0.0131 on their matches), but even established teams after October lose by +0.006.
- **Round 2 tuning** (`rd_eu_upper_second2.py`, verified): 720 settings + 4 calibrations tuned on results only (2020/21–2023/24), locked, tested once on 2024/25–2025/26. Locked: alpha 0.3, Poisson on both divisions, DC 3-year window, xi 0.002, newcomer = league average, 70% Poisson / 30% DC, per-league Platt. Tuning log loss 0.6781 → 0.6756 (blend weight mattered most; window, decay and newcomer rating didn't).
- Final test: **+0.0075 vs bookmaker average, +0.0078 vs Betfair close** (round 1: +0.0089 / +0.0093). Still fails the bar. Betting: −7.3% early / −3.9% close. Serie B closest (+0.0032 vs close, CI includes 0). La Liga 2 Over rate jumped to 49% in the test seasons; model and market both lag.
- The test seasons 2024/25–2025/26 have now been used once for this group; further tuning against them would flatter the model.
- Not frozen, not tracked.

**Side research: Bzzoiro Sports Data API (26 Oct)** - see `claude/bzzoiro-side-research-2026-10-26.md` (Project) / `scratch\` (PC). League Two xG and line-ups are excellent quality but add no edge; the market already prices team news.

**Bottom line:** with free public stats, no model beats the market. England is roughly break-even; Europe loses about 4%; the European second divisions are no softer (worse than the top 4). The gap is information the market has (line-ups, injuries, late money), not more of the same features.

**Data notes:** FBref lost its Opta advanced data around Feb 2026. Understat covers the EPL + top 5 leagues (no Championship). One Ligue 2 match (Bastia v Red Star, 5 Dec 2025) has no score and is excluded from the export.

## 7. Rules agreed
- Judge models by the walk-forward mean across seasons; the TEST seasons are spent.
- Never change a tracked model. With several models tracked, one can look good by luck, so every model meets the same bar.
- **No real money** until a frozen model beats the Betfair close on log loss **and** shows positive ROI with the whole 95% interval above zero over **200+ settled bets**. 200 bets is a minimum (ROI CI is roughly ±14% at that size). Watch closing-line value (CLV) first.
- Paper staking: 1 unit flat (1% of a notional 100-unit bank). If ever real: a fixed bankroll you can afford to lose, 1% flat, review at −30%, quarter-Kelly capped at 2% only after a proven edge.
- Brian's direction: **nothing gets parked; build as many models as possible.**

## 8. Key files (`src/`)
- **Data pipeline:** `download_data.py`, `import_data.py`, `import_helpers.py` (DIVISIONS incl. SP1/I1/D1/F1 + SP2/I2/D2/F2), `update_current_season.py`, `clean_data.py` (exports England → `matches_clean.csv`, Europe → `matches_clean_eu_upper.csv`), `build_features.py` (`build_feature_table`), `build_features_eu_upper.py`, `download_eu_upper.py`, `download_understat.py` (all 5 leagues → `match_xg` table).
- **Models:** `model_v3.py`, `freeze_model_v3.py`, `model_v3_eu_upper.py`, `freeze_model_v3_eu_upper.py`, `league_groups.py`.
- **Live:** `weekly_predict.py`, `weekly_bets.py`, `weekly_settle.py`, `live_db.py` (tables: `live_fixtures`, `live_prices`, `predictions` + components, `bets`), `export_tracking_xlsx.py`.
- **R&D:** `rd_experiments.py`–`rd_experiments4.py`, `league_breakdown.py`, `rd_xg_upper.py`, `rd_eu_upper.py`, `rd_eu_upper2.py` (has `build_predictions()`), `rd_eu_upper3.py`, `rd_eu_upper_second.py`, `rd_eu_upper_second2.py`, `test_evaluation.py`. Side research scripts and data (Bzzoiro) are in `scratch\` (not in git). Reports are in `reports/`.

## 9. Open items / ideas
- Tuesday: `weekly_settle.py` → first live numbers.
- Commit the v3-eu-upper freeze if not done yet (config, league_groups, weekly_predict, model_v3_eu_upper, freeze_model_v3_eu_upper, models/manifest_v3_eu_upper.json, cheatsheet).
- Small fix: `weekly_bets.py` says "no price cleared EV" when re-run on an already-recorded sheet. It should say "nothing new recorded".
- Commit `src/rd_eu_upper_second.py` and `src/rd_eu_upper_second2.py` plus their reports in `reports/`.
- Next R&D direction (agreed 26 Oct): stop adding more of the same stats. Test **(1) betting at the best bookmaker price** instead of Betfair, **(2) Pinnacle-close vs best-price value with no model**, **(3) whether the model predicts price movement (open → close, i.e. CLV)**. Raw football-data files already contain Pinnacle (P/PC), best price (Max/MaxC), Bet365, average, Betfair, opening and closing, plus Asian handicap.
- Odds check (England 2024/25–25/26): Betfair after 2% commission beats the average bookmaker 83–100% of the time, but the best bookmaker price beats Betfair-net about 60–67% of the time at opening. Closing margins: Betfair 0.8%, Pinnacle 3.7%, average bookmaker 6.0%.
- Regenerate the Bzzoiro API token (it is in plain text in the `scratch\` scripts).
- Next model idea: **v3-eu-lower** (Netherlands N1, Belgium B1 in the same format; Sweden/Denmark are in Football-Data's "extra leagues" format without shots, so they'd need a slimmer feature set).
- Possible later work: automated Betfair prices (API) if manual entry gets heavy (about 38 European games a week); a Power BI dashboard on the SQLite tables (Phase 12).

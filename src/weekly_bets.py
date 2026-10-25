"""
Phase 11 - weekly step 2: record the Betfair prices you entered and place
PAPER bets where the frozen model v3 sees value.

For every row of the odds sheet with BOTH prices filled in:
  - stores the prices + the time you checked them (live_prices). Only the
    first price entered for a match is kept - no going back later to
    "improve" a price.
  - uses the probability STORED by weekly_predict.py (not the CSV column,
    so editing the sheet can't change the model's number)
  - EV per 1 unit staked, after commission (Phase 10 formula):
        EV = p * (odds - 1) * (1 - commission) - (1 - p)
    for Over and for Under. If the better side has EV >= MIN_EDGE
    (config, 3%), a flat paper bet of PAPER_STAKE is recorded as 'pending'.

Rows with blank prices are skipped (fill them in and run again - already
recorded matches are skipped automatically).

Run from inside src/:
    python weekly_bets.py ..\\data\\live\\odds_entry_2026-10-31.csv
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402
import league_groups  # noqa: E402
import live_db  # noqa: E402


def ev(p, odds, commission):
    return p * (odds - 1) * (1 - commission) - (1 - p)


def parse_checked_at(value):
    """Accepts 2026-10-31 14:05 (ISO) or 31/10/2026 14:05 (how Excel on a UK
    PC re-saves it). Day always before month for the / format."""
    if pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S",
                "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("odds_csv")
    args = ap.parse_args()

    sheet = pd.read_csv(args.odds_csv, dtype=str, keep_default_na=False)
    conn = live_db.connect()
    c, min_edge, stake = config.COMMISSION_RATE, config.MIN_EDGE, config.PAPER_STAKE
    now = datetime.now()

    filled = sheet[sheet["betfair_over_odds"].str.strip().ne("")
                   & sheet["betfair_under_odds"].str.strip().ne("")]
    print(f"{len(filled)} of {len(sheet)} rows have both prices filled in.")

    recorded, skipped, bets, problems = 0, 0, [], []
    for r in filled.itertuples(index=False):
        try:
            o_over, o_under = float(r.betfair_over_odds), float(r.betfair_under_odds)
        except ValueError:
            problems.append(f"{r.home} v {r.away}: odds not numbers ({r.betfair_over_odds}, {r.betfair_under_odds})")
            continue
        if o_over <= 1.01 or o_under <= 1.01 or o_over > 20 or o_under > 20:
            problems.append(f"{r.home} v {r.away}: odds look wrong ({o_over}, {o_under})")
            continue
        if 1 / o_over + 1 / o_under < 0.97:
            problems.append(f"{r.home} v {r.away}: {o_over}/{o_under} add up to under 97% - typo?")
            continue
        checked = parse_checked_at(r.checked_at)
        if checked is None:
            problems.append(f"{r.home} v {r.away}: checked_at missing or unreadable ('{r.checked_at}') "
                            f"- use e.g. 2026-10-31 14:05")
            continue
        fixture = conn.execute("SELECT match_date, league_name FROM live_fixtures WHERE match_id = ?",
                               (r.match_id,)).fetchone()
        if fixture is None:
            problems.append(f"{r.home} v {r.away}: unknown match_id (run weekly_predict.py first)")
            continue
        match_date = datetime.strptime(fixture[0], "%Y-%m-%d").date()   # from the DB, not the sheet
        if checked.date() > match_date:
            problems.append(f"{r.home} v {r.away}: checked_at {checked} is after the match date - not allowed")
            continue

        grp = league_groups.group_of_league(fixture[1])
        version = league_groups.group(grp)["live_model"] if grp else None
        row = conn.execute("""SELECT prediction_id, predicted_prob_over_2_5 FROM predictions
                              WHERE match_id = ? AND model_version = ?""",
                           (r.match_id, version)).fetchone()
        if row is None:
            problems.append(f"{r.home} v {r.away}: no stored prediction (run weekly_predict.py first)")
            continue
        pred_id, p = row

        cur = conn.execute("""INSERT OR IGNORE INTO live_prices
                              (match_id, over_odds, under_odds, checked_at, entered_at)
                              VALUES (?, ?, ?, ?, ?)""",
                           (r.match_id, o_over, o_under, checked.isoformat(timespec="minutes"),
                            now.isoformat(timespec="seconds")))
        if cur.rowcount == 0:
            skipped += 1
            continue
        recorded += 1

        ev_over, ev_under = ev(p, o_over, c), ev(1 - p, o_under, c)
        side, odds, best_ev = ("over", o_over, ev_over) if ev_over >= ev_under else ("under", o_under, ev_under)
        if best_ev >= min_edge:
            conn.execute("""INSERT INTO bets (match_id, prediction_id, stake, odds_taken, bet_side, commission,
                                              result, profit_loss, bet_date, expected_value)
                            VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, ?, ?)""",
                         (r.match_id, pred_id, stake, odds, side, c,
                          checked.isoformat(timespec="minutes"), best_ev))
            bets.append({"date": match_date.isoformat(), "match": f"{r.home} v {r.away}", "side": side.upper(),
                         "odds": odds, "model_p": round(p if side == "over" else 1 - p, 3),
                         "EV": f"{best_ev:+.1%}"})
    conn.commit()

    print(f"Prices recorded: {recorded} new, {skipped} already recorded earlier (kept the first price).")
    if problems:
        print("\nNot recorded - fix these rows and run again:")
        for msg in problems:
            print(f"  - {msg}")
    if bets:
        print(f"\n{len(bets)} PAPER bet(s) at {stake:g} unit flat stake (EV >= {min_edge:.0%} after "
              f"{c:.0%} commission):")
        print(pd.DataFrame(bets).to_string(index=False))
    else:
        print(f"\nNo paper bets this time - no price cleared EV >= {min_edge:.0%}.")
    print("\nAfter the matches: python weekly_settle.py")


if __name__ == "__main__":
    main()

"""
Manual settle - for when football-data.co.uk hasn't published results yet.

Settles PENDING paper bets from a small results file you (or Claude) write,
using the same win/loss maths as weekly_settle.py. It does NOT touch the
matches table (the data the model learns from), so no half-complete rows
get in there. The scores go in a separate manual_results table (shown in
the spreadsheet) and the fixtures are marked played. Safe to re-run: bets
already settled are left alone.

When football-data later publishes these matches, run weekly_settle.py as
normal: it fills in the scores, Betfair closing odds, log loss and the
closing-line checks automatically. Bets already settled here stay settled.

Results file (CSV): match_id,home_goals,away_goals
    e.g. ..\\data\\live\\manual_results_2026-10-27.csv

Run from inside src/:
    python manual_settle.py ..\\data\\live\\manual_results_2026-10-27.csv
"""

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import live_db  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print("Usage: python manual_settle.py <results csv>")
        sys.exit(1)
    res = pd.read_csv(sys.argv[1])
    res.columns = [c.strip() for c in res.columns]

    conn = live_db.connect()
    known = {r[0] for r in conn.execute("SELECT match_id FROM live_fixtures")}
    unknown = sorted(set(res["match_id"]) - known)
    if unknown:
        print("STOP: these match_ids are not in live_fixtures (check spelling):")
        for m in unknown:
            print("  ", m)
        sys.exit(1)

    now = datetime.now().isoformat(timespec="seconds") + " (manual)"
    conn.execute("""CREATE TABLE IF NOT EXISTS manual_results (
                        match_id TEXT PRIMARY KEY, home_goals INTEGER NOT NULL, away_goals INTEGER NOT NULL,
                        entered_at TEXT NOT NULL)""")
    total, rows = 0.0, []
    for r in res.itertuples(index=False):
        goals = int(r.home_goals) + int(r.away_goals)
        over = goals > 2
        # keep the score (shown in the spreadsheet until football-data publishes it) and mark
        # the fixture played, so weekly_settle never voids it as "no result after 7 days"
        conn.execute("INSERT OR REPLACE INTO manual_results VALUES (?, ?, ?, ?)",
                     (r.match_id, int(r.home_goals), int(r.away_goals), now))
        conn.execute("UPDATE live_fixtures SET status = 'played' WHERE match_id = ? AND status = 'scheduled'",
                     (r.match_id,))
        bets = conn.execute("""SELECT bet_id, stake, odds_taken, bet_side, commission FROM bets
                               WHERE match_id = ? AND result = 'pending'""", (r.match_id,)).fetchall()
        for bet_id, stake, odds, side, comm in bets:
            won = over if side == "over" else not over
            pl = stake * (odds - 1) * (1 - comm) if won else -stake
            conn.execute("UPDATE bets SET result = ?, profit_loss = ?, settled_at = ? WHERE bet_id = ?",
                         ("win" if won else "loss", pl, now, bet_id))
            total += pl
            rows.append((r.match_id, f"{r.home_goals}-{r.away_goals}", side, odds, "win" if won else "loss", round(pl, 2)))
        if not bets:
            rows.append((r.match_id, f"{r.home_goals}-{r.away_goals}", "-", None, "no pending bet", 0.0))
    conn.commit()

    out = pd.DataFrame(rows, columns=["match_id", "score", "side", "odds", "result", "profit"])
    print(out.to_string(index=False))
    print(f"\nSettled now: {sum(1 for x in rows if x[2] != '-')} bets, profit {total:+.2f} units")
    allb = pd.read_sql_query("SELECT result, profit_loss FROM bets", conn)
    s = allb[allb["result"].isin(["win", "loss"])]
    print(f"All-time: {len(s)} settled, {(s['result'] == 'win').sum()} wins, "
          f"profit {s['profit_loss'].sum():+.2f} units, {(allb['result'] == 'pending').sum()} pending")
    print("\nNext: python export_tracking_xlsx.py   (close Excel first)")


if __name__ == "__main__":
    main()

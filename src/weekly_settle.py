"""
Phase 11 - weekly step 3: settle paper bets and report the live track record
of the frozen model v3.

  1. Downloads the latest 2026/27 results into the database.
  2. Marks each predicted fixture 'played' once its result is in. A fixture
     still without a result 7+ days after its date (postponed / moved) is
     marked 'void' and any bet on it is voided (profit 0), as Betfair would.
  3. Settles pending bets: win = stake * (odds - 1) * (1 - commission),
     loss = -stake.
  4. Reports, all-time:
       - model log loss vs YOUR recorded Betfair price and vs Betfair's
         CLOSING price (from football-data.co.uk, once published), on the
         same matches
       - bets: P/L, ROI with a 95% bootstrap interval, closing-line value
         (did you get better odds than the close?)
     Written to reports/live_tracking.md and printed.

Run from inside src/ (after the weekend's matches):
    python weekly_settle.py
"""

import os
import sys
import argparse
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
import config  # noqa: E402
import model_v3 as mv  # noqa: E402
import live_db  # noqa: E402

REPORT_MD = os.path.join("..", "reports", "live_tracking.md")
VOID_AFTER_DAYS = 7
MIN_BETS_FOR_CI = 30
SEED = 424410


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def mean_ll(y, p):
    p = clip(p)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def demargin(o, u):
    return (1 / o) / (1 / o + 1 / u)


def md_table(df):
    df = df.copy().astype(object).where(df.notna(), "")
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep] + rows)


def settle(conn, today):
    now = datetime.now().isoformat(timespec="seconds")
    played = conn.execute("""
        UPDATE live_fixtures SET status = 'played'
        WHERE status = 'scheduled'
          AND match_id IN (SELECT match_id FROM matches WHERE over_2_5 IS NOT NULL)
    """).rowcount
    cutoff = (today - timedelta(days=VOID_AFTER_DAYS)).isoformat()
    voided = conn.execute("""
        UPDATE live_fixtures SET status = 'void'
        WHERE status = 'scheduled' AND match_date < ?
    """, (cutoff,)).rowcount

    pending = conn.execute("""
        SELECT b.bet_id, b.stake, b.odds_taken, b.bet_side, b.commission, f.status, m.over_2_5
        FROM bets b
        JOIN live_fixtures f ON f.match_id = b.match_id
        LEFT JOIN matches m ON m.match_id = b.match_id
        WHERE b.result = 'pending'
    """).fetchall()
    n_settled = 0
    for bet_id, stake, odds, side, comm, status, over in pending:
        if status == "void":
            result, pl = "void", 0.0
        elif status == "played":
            won = (over == 1) if side == "over" else (over == 0)
            result = "win" if won else "loss"
            pl = stake * (odds - 1) * (1 - comm) if won else -stake
        else:
            continue
        conn.execute("UPDATE bets SET result = ?, profit_loss = ?, settled_at = ? WHERE bet_id = ?",
                     (result, pl, now, bet_id))
        n_settled += 1
    conn.commit()
    return played, voided, n_settled


def build_report(conn):
    rng = np.random.default_rng(SEED)
    lines = [f"# Live paper-tracking - model {mv.MODEL_VERSION}\n",
             f"Updated {datetime.now():%Y-%m-%d %H:%M}. Leagues: {', '.join(config.LIVE_LEAGUES)}. "
             f"Flat {config.PAPER_STAKE:g}-unit paper stakes, {config.COMMISSION_RATE:.0%} commission, "
             f"min EV {config.MIN_EDGE:.0%}. Model frozen - never changed during tracking.\n"]

    preds = pd.read_sql_query("""
        SELECT p.match_id, f.league_name, f.match_date, f.status, p.predicted_prob_over_2_5 AS p_model,
               m.over_2_5, lp.over_odds, lp.under_odds,
               o.bfe_close_over_2_5 AS close_over, o.bfe_close_under_2_5 AS close_under
        FROM predictions p
        JOIN live_fixtures f ON f.match_id = p.match_id
        LEFT JOIN matches m ON m.match_id = p.match_id
        LEFT JOIN live_prices lp ON lp.match_id = p.match_id
        LEFT JOIN odds o ON o.match_id = p.match_id
        WHERE p.model_version = ?
    """, conn, params=(mv.MODEL_VERSION,))
    done = preds[(preds["status"] == "played") & preds["over_2_5"].notna()].copy()
    counts = preds["status"].value_counts().to_dict()
    lines.append(f"Predictions: {len(preds)} total - {counts.get('played', 0)} played, "
                 f"{counts.get('scheduled', 0)} upcoming, {counts.get('void', 0)} void.\n")

    # --- Probability quality vs the market -----------------------------------
    rows = []
    for label, sub in [("All leagues", done)] + [(lg, done[done["league_name"] == lg]) for lg in config.LIVE_LEAGUES]:
        y = sub["over_2_5"].values.astype(float)
        row = {"league": label, "played": len(sub),
               "model_log_loss": round(mean_ll(y, sub["p_model"]), 4) if len(sub) else None}
        for name, o_col, u_col in [("your_price", "over_odds", "under_odds"),
                                   ("betfair_close", "close_over", "close_under")]:
            m = sub[o_col].notna() & sub[u_col].notna()
            if m.sum():
                ym = y[m.values]
                p_mkt = demargin(sub.loc[m, o_col], sub.loc[m, u_col])
                row[f"n_vs_{name}"] = int(m.sum())
                row[f"model_vs_{name}"] = round(mean_ll(ym, sub.loc[m, "p_model"]), 4)
                row[f"{name}_log_loss"] = round(mean_ll(ym, p_mkt), 4)
        rows.append(row)
    quality = pd.DataFrame(rows)
    lines.append("## Are the probabilities better than the market? (lower log loss = better)\n")
    lines.append(md_table(quality) + "\n")

    # --- Bets --------------------------------------------------------------------
    bets = pd.read_sql_query("""
        SELECT b.*, f.league_name, f.home_team, f.away_team, f.match_date,
               o.bfe_close_over_2_5 AS close_over, o.bfe_close_under_2_5 AS close_under
        FROM bets b
        JOIN live_fixtures f ON f.match_id = b.match_id
        LEFT JOIN odds o ON o.match_id = b.match_id
    """, conn)
    settled = bets[bets["result"].isin(["win", "loss"])].copy()
    lines.append("\n## Paper bets\n")
    if settled.empty:
        lines.append(f"No settled bets yet ({(bets['result'] == 'pending').sum()} pending).\n")
        summary = None
    else:
        n = len(settled)
        pl = settled["profit_loss"].values
        roi = pl.sum() / settled["stake"].sum()
        settled["close_odds"] = np.where(settled["bet_side"] == "over", settled["close_over"], settled["close_under"])
        has_close = settled["close_odds"].notna()
        clv = (settled.loc[has_close, "odds_taken"] / settled.loc[has_close, "close_odds"] - 1)
        summary = {
            "settled_bets": n,
            "pending": int((bets["result"] == "pending").sum()),
            "void": int((bets["result"] == "void").sum()),
            "wins": int((settled["result"] == "win").sum()),
            "strike_rate": f"{(settled['result'] == 'win').mean():.1%}",
            "avg_odds": round(settled["odds_taken"].mean(), 2),
            "avg_expected_value": f"{settled['expected_value'].mean():+.1%}",
            "profit_units": round(pl.sum(), 2),
            "roi": f"{roi:+.1%}",
        }
        if n >= MIN_BETS_FOR_CI:
            stakes = settled["stake"].values
            boots = []
            for _ in range(5000):
                i = rng.integers(0, n, n)
                boots.append(pl[i].sum() / stakes[i].sum())
            summary["roi_95pct_interval"] = f"{np.percentile(boots, 2.5):+.1%} to {np.percentile(boots, 97.5):+.1%}"
        else:
            summary["roi_95pct_interval"] = f"needs {MIN_BETS_FOR_CI}+ settled bets"
        if has_close.any():
            summary["avg_closing_line_value"] = f"{clv.mean():+.1%} ({has_close.sum()} bets)"
            summary["bets_beating_close"] = f"{(clv > 0).mean():.0%}"
        lines.append(md_table(pd.DataFrame([summary]).T.reset_index().set_axis(["metric", "value"], axis=1)) + "\n")

        by_league = settled.groupby("league_name").agg(
            bets=("bet_id", "count"), profit_units=("profit_loss", "sum"), staked=("stake", "sum")).reset_index()
        by_league["roi"] = (by_league["profit_units"] / by_league["staked"]).map(lambda v: f"{v:+.1%}")
        by_league["profit_units"] = by_league["profit_units"].round(2)
        lines.append("\n### By league\n")
        lines.append(md_table(by_league.drop(columns="staked")) + "\n")

        recent = bets.sort_values("bet_date", ascending=False).head(20)
        recent = recent[["match_date", "league_name", "home_team", "away_team", "bet_side",
                         "odds_taken", "expected_value", "result", "profit_loss"]].copy()
        recent["expected_value"] = recent["expected_value"].map(lambda v: f"{v:+.1%}")
        lines.append("\n### Last 20 bets\n")
        lines.append(md_table(recent) + "\n")

    lines.append("\n**Rule:** no real money unless the model beats Betfair's closing price on log loss AND the "
                 "bets show positive ROI with the whole 95% interval above zero, over 200+ settled bets.\n")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return quality, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-refresh", action="store_true", help="don't download results (testing only)")
    ap.add_argument("--today", help="pretend today is YYYY-MM-DD (testing only)")
    args = ap.parse_args()
    today = date.fromisoformat(args.today) if args.today else date.today()

    if not args.skip_refresh:
        import update_current_season
        print("Updating 2026/27 results from football-data.co.uk...")
        update_current_season.main()

    conn = live_db.connect()
    played, voided, n_settled = settle(conn, today)
    print(f"\nFixtures newly played: {played}, newly void: {voided}. Bets settled now: {n_settled}.")

    quality, summary = build_report(conn)
    pd.set_option("display.width", 200)
    print("\nProbability quality (lower log loss = better):")
    print(quality.to_string(index=False))
    if summary:
        print("\nPaper bets:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    else:
        print("\nNo settled bets yet.")
    print(f"\nWrote {REPORT_MD}")


if __name__ == "__main__":
    main()

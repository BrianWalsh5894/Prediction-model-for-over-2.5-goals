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
       - closing-line value per bet: your odds vs the close, and the bet's
         value if Betfair's closing price is the truth ("value at close",
         after commission, margin removed)
       - market movement: for EVERY played prediction you priced (bet or
         not), did Betfair's price move towards the model's view between
         your price and the close? Shows up long before profit does.
     Written to reports/live_tracking_<group>.md (one per live league group) and printed.

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
import league_groups  # noqa: E402
import live_db  # noqa: E402

REPORT_DIR = os.path.join("..", "reports")
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


def boot_ci(values, rng, n_boot=5000):
    v = np.asarray(values, dtype=float)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def add_clv(bets):
    """Per bet: close_odds (Betfair close on your side), clv = your odds vs the
    close (raw), value_at_close = the bet's EV after commission if Betfair's
    closing price (margin removed) is the true probability."""
    b = bets.copy()
    over = b["bet_side"] == "over"
    b["close_odds"] = np.where(over, b["close_over"], b["close_under"])
    p_over_close = demargin(b["close_over"], b["close_under"])
    p_side = np.where(over, p_over_close, 1 - p_over_close)
    b["clv"] = b["odds_taken"] / b["close_odds"] - 1
    b["value_at_close"] = p_side * (b["odds_taken"] - 1) * (1 - b["commission"]) - (1 - p_side)
    return b


def market_movement(done, rng):
    """For played predictions with both your price and the close: did the
    market move towards the model? Positive = towards the model."""
    m = done[done[["over_odds", "under_odds", "close_over", "close_under"]].notna().all(axis=1)].copy()
    if m.empty:
        return None, m
    m["p_yours"] = demargin(m["over_odds"], m["under_odds"])
    m["p_close"] = demargin(m["close_over"], m["close_under"])
    direction = np.sign(m["p_model"] - m["p_yours"])
    m["move_to_model_pp"] = direction * (m["p_close"] - m["p_yours"]) * 100
    n = len(m)
    out = {"priced_and_played": n,
           "moved_towards_model": f"{(m['move_to_model_pp'] > 0).mean():.0%}",
           "moved_away": f"{(m['move_to_model_pp'] < 0).mean():.0%}",
           "no_move": f"{(m['move_to_model_pp'] == 0).mean():.0%}",
           "avg_move_towards_model_pp": f"{m['move_to_model_pp'].mean():+.2f}"}
    if n >= MIN_BETS_FOR_CI:
        lo, hi = boot_ci(m["move_to_model_pp"].values, rng)
        out["avg_move_95pct_interval"] = f"{lo:+.2f} to {hi:+.2f}"
    else:
        out["avg_move_95pct_interval"] = f"needs {MIN_BETS_FOR_CI}+ matches"
    return out, m


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


def build_report(conn, group_name):
    rng = np.random.default_rng(SEED)
    grp = league_groups.group(group_name)
    leagues, version = grp["leagues"], grp["live_model"]
    lines = [f"# Live paper-tracking - group '{group_name}', model {version}\n",
             f"Updated {datetime.now():%Y-%m-%d %H:%M}. Leagues: {', '.join(leagues)}. "
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
    """, conn, params=(version,))
    preds = preds[preds["league_name"].isin(leagues)]
    done = preds[(preds["status"] == "played") & preds["over_2_5"].notna()].copy()
    counts = preds["status"].value_counts().to_dict()
    lines.append(f"Predictions: {len(preds)} total - {counts.get('played', 0)} played, "
                 f"{counts.get('scheduled', 0)} upcoming, {counts.get('void', 0)} void.\n")

    # --- Probability quality vs the market -----------------------------------
    rows = []
    for label, sub in [("All leagues", done)] + [(lg, done[done["league_name"] == lg]) for lg in leagues]:
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

    # --- Market movement: did Betfair move towards the model? -----------------
    movement, _ = market_movement(done, rng)
    lines.append("\n## Did the market move towards the model? (all priced predictions, bet or not)\n")
    lines.append("Between the price you recorded and Betfair's close. `avg_move_towards_model_pp` is in "
                 "percentage points of P(Over); positive = the market moved the model's way. A real edge "
                 "shows up here as a positive average with the interval above zero.\n")
    if movement is None:
        lines.append("No played matches with both your price and the Betfair close yet.\n")
    else:
        lines.append(md_table(pd.DataFrame([movement]).T.reset_index().set_axis(["metric", "value"], axis=1)) + "\n")

    # --- Bets --------------------------------------------------------------------
    bets = pd.read_sql_query("""
        SELECT b.*, f.league_name, f.home_team, f.away_team, f.match_date,
               o.bfe_close_over_2_5 AS close_over, o.bfe_close_under_2_5 AS close_under
        FROM bets b
        JOIN live_fixtures f ON f.match_id = b.match_id
        LEFT JOIN odds o ON o.match_id = b.match_id
    """, conn)
    bets = bets[bets["league_name"].isin(leagues)]
    settled = bets[bets["result"].isin(["win", "loss"])].copy()
    lines.append("\n## Paper bets\n")
    if settled.empty:
        lines.append(f"No settled bets yet ({(bets['result'] == 'pending').sum()} pending).\n")
        summary = None
    else:
        n = len(settled)
        pl = settled["profit_loss"].values
        roi = pl.sum() / settled["stake"].sum()
        settled = add_clv(settled)
        has_close = settled["close_odds"].notna()
        clv = settled.loc[has_close, "clv"]
        value_at_close = settled.loc[has_close, "value_at_close"]
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
            summary["avg_value_at_close"] = f"{value_at_close.mean():+.1%}"
            if has_close.sum() >= MIN_BETS_FOR_CI:
                lo, hi = boot_ci(value_at_close.values, rng)
                summary["value_at_close_95pct_interval"] = f"{lo:+.1%} to {hi:+.1%}"
            else:
                summary["value_at_close_95pct_interval"] = f"needs {MIN_BETS_FOR_CI}+ bets with a close"
        lines.append(md_table(pd.DataFrame([summary]).T.reset_index().set_axis(["metric", "value"], axis=1)) + "\n")

        by_league = settled.groupby("league_name").agg(
            bets=("bet_id", "count"), profit_units=("profit_loss", "sum"), staked=("stake", "sum")).reset_index()
        by_league["roi"] = (by_league["profit_units"] / by_league["staked"]).map(lambda v: f"{v:+.1%}")
        by_league["profit_units"] = by_league["profit_units"].round(2)
        lines.append("\n### By league\n")
        lines.append(md_table(by_league.drop(columns="staked")) + "\n")

        recent = add_clv(bets.sort_values("bet_date", ascending=False).head(20))
        recent = recent[["match_date", "league_name", "home_team", "away_team", "bet_side",
                         "odds_taken", "close_odds", "clv", "value_at_close", "expected_value",
                         "result", "profit_loss"]].copy()
        for c in ["expected_value", "clv", "value_at_close"]:
            recent[c] = recent[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
        recent["profit_loss"] = recent["profit_loss"].round(2)
        lines.append("\n### Last 20 bets\n")
        lines.append(md_table(recent) + "\n")

    lines.append("\n**Rule:** no real money unless the model beats Betfair's closing price on log loss AND the "
                 "bets show positive ROI with the whole 95% interval above zero, over 200+ settled bets.\n")
    report_md = os.path.join(REPORT_DIR, f"live_tracking_{group_name}.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return quality, summary, movement, report_md


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

    pd.set_option("display.width", 200)
    for group_name in league_groups.live_groups():
        quality, summary, movement, report_md = build_report(conn, group_name)
        grp = league_groups.group(group_name)
        print(f"\n===== Group '{group_name}' ({', '.join(grp['leagues'])}), model {grp['live_model']} =====")
        print("Probability quality (lower log loss = better):")
        print(quality.to_string(index=False))
        if movement:
            print("\nDid the market move towards the model? (your price -> Betfair close, all priced matches):")
            for k, v in movement.items():
                print(f"  {k}: {v}")
        if summary:
            print("\nPaper bets:")
            for k, v in summary.items():
                print(f"  {k}: {v}")
        else:
            print("\nNo settled bets yet.")
        print(f"\nWrote {report_md}")

    # Ongoing results spreadsheet (reports/live_tracking.xlsx)
    try:
        import export_tracking_xlsx
        path, nb, npred = export_tracking_xlsx.build()
        print(f"Wrote {path} ({nb} bets, {npred} predictions)")
    except ImportError:
        print("Spreadsheet not updated: openpyxl missing - run  pip install openpyxl")
    except PermissionError:
        print("Spreadsheet not updated: live_tracking.xlsx is open in Excel - close it and run "
              "python export_tracking_xlsx.py")


if __name__ == "__main__":
    main()

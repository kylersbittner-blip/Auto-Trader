#!/usr/bin/env python3
"""
run_session.py — Run a complete trading day.

Start before market open. It will:
  1. Wait for 9:15 ET → run pre-market scan
  2. Wait for 10:00 ET → classify regime, initialize strategies
  3. Process 30-min bars through close
  4. Print session summary and save to logs/

Usage:
  python run_session.py              # paper trading
  python run_session.py --dry-run    # scan + classify only, no trades

Requires .env with ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
import zoneinfo

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from data.adapters import (
    AlpacaDataSource,
    PolygonNewsSource,
    fetch_30min_bars_batch,
)
from engine.session_runner import SessionRunner, SessionSummary, TradeRecord
from engine.regime import DayRegime, SessionPhase, get_session_phase

ET = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")

PREMARKET_SCAN_TIME = time(9, 15)
SESSION_INIT_TIME = time(10, 0)
AFTERNOON_ACTIVATE_TIME = time(13, 15)
SESSION_CLOSE_TIME = time(15, 55)

BAR_TIMES = [
    time(10, 0), time(10, 30), time(11, 0), time(11, 30),
    time(12, 0), time(12, 30), time(13, 0), time(13, 30),
    time(14, 0), time(14, 30), time(15, 0), time(15, 30),
]


def log(msg, level="INFO"):
    now = datetime.now(ET).strftime("%H:%M:%S ET")
    print(f"[{now}] [{level}] {msg}")


def log_trade(record):
    if record.action == "entry":
        log(f"  ENTRY {record.direction.upper()} {record.ticker} | qty={record.qty} @ ${record.entry_price:.2f} | stop=${record.stop_price:.2f} target=${record.target_price:.2f} | risk=${record.risk_usd:.0f} | strategy={record.strategy}", "TRADE")
    elif record.action == "exit":
        log(f"  EXIT {record.ticker} | qty={record.qty} @ ${record.exit_price:.2f} | reason={record.exit_reason} | strategy={record.strategy}", "TRADE")
    elif record.action == "skip":
        log(f"  SKIP {record.ticker} | {record.reason} | strategy={record.strategy}", "WARN")


def print_summary(summary):
    print("\n" + "=" * 60)
    print(f"  SESSION SUMMARY — {summary.date}")
    print("=" * 60)
    print(f"  Regime:       {summary.regime}")
    print(f"  Scanned:      {summary.candidates_scanned} tickers")
    print(f"  Passed:       {summary.candidates_passed} candidates")
    print(f"  Entries:      {summary.trades_entered}")
    print(f"  Exits:        {summary.trades_exited}")
    print(f"  Skipped:      {summary.trades_skipped}")
    if summary.trade_log:
        print(f"\n  Trade Log:")
        for r in summary.trade_log:
            t = r.timestamp.strftime("%H:%M") if r.timestamp else "?"
            if r.action == "entry":
                print(f"    {t} | {r.action.upper():5} | {r.direction:5} {r.ticker:5} | qty={r.qty} @ ${r.entry_price:.2f}")
            elif r.action == "exit":
                print(f"    {t} | {r.action.upper():5} | {r.ticker:5} | qty={r.qty} @ ${r.exit_price:.2f} ({r.exit_reason})")
            else:
                print(f"    {t} | {r.action.upper():5} | {r.ticker:5} | {r.reason}")
    print("=" * 60 + "\n")


def now_et():
    return datetime.now(ET)


async def wait_until(target_time, label=""):
    now = now_et()
    target = now.replace(hour=target_time.hour, minute=target_time.minute, second=0, microsecond=0)
    if now >= target:
        return
    delta = (target - now).total_seconds()
    if delta > 0:
        log(f"Waiting until {target_time.strftime('%H:%M')} ET{' — ' + label if label else ''}")
        await asyncio.sleep(delta)


async def wait_for_next_bar():
    now = now_et()
    current = now.time()
    for bt in BAR_TIMES:
        if bt > current:
            await wait_until(bt, f"next bar at {bt.strftime('%H:%M')}")
            return bt
    return time(16, 0)


async def fetch_vix(api_key, secret_key):
    try:
        from data.adapters import fetch_30min_bars_sync
        df = fetch_30min_bars_sync("VIXY", api_key, secret_key, days=1)
        if not df.empty:
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    return 18.0


async def run_trading_day(dry_run=False):
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    polygon_key = os.getenv("POLYGON_API_KEY")

    if not api_key or not secret_key:
        log("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env", "ERROR")
        sys.exit(1)
    if not polygon_key:
        log("POLYGON_API_KEY must be set in .env (free tier works)", "ERROR")
        sys.exit(1)

    data_source = AlpacaDataSource(api_key, secret_key)
    news_source = PolygonNewsSource(polygon_key)

    universe = [
        "NVDA", "TSLA", "AAPL", "META", "AMD", "MSFT", "AMZN",
        "GOOGL", "NFLX", "AVGO", "CRM", "ORCL", "ADBE", "INTC",
        "MU", "QCOM", "COIN", "SQ", "SHOP", "PLTR",
    ]

    runner = SessionRunner(
        data_source=data_source,
        news_source=news_source,
        account_equity=10_000,
        universe=universe,
    )

    today = now_et().strftime("%Y-%m-%d")
    log(f"Auto-Trader session starting for {today}")
    log(f"Mode: {'DRY RUN (no trades)' if dry_run else 'PAPER TRADING'}")
    log(f"Universe: {len(universe)} tickers")

    await wait_until(PREMARKET_SCAN_TIME, "pre-market scan")

    log("Running pre-market scan...")
    scan_result = runner.run_premarket_scan()

    log(f"Scan complete: {scan_result.total_scanned} scanned, {scan_result.total_passed} passed")
    if scan_result.candidates:
        for c in scan_result.candidates:
            log(f"  #{c.rank} {c.ticker} — score {c.score:.1f} | RVOL {c.relative_volume:.1f}x | gap {c.gap_pct:+.1%} | ATR ${c.atr:.2f} | {'NEWS' if c.has_news else 'no news'}")
    else:
        log("No candidates found. Session will idle.", "WARN")

    if not runner.active_tickers:
        log("No Stocks in Play today. Exiting.")
        print_summary(runner.get_summary(today))
        return

    if dry_run:
        log("Dry run — stopping after scan.", "INFO")
        print_summary(runner.get_summary(today))
        return

    await wait_until(SESSION_INIT_TIME, "opening range formation")

    log("Fetching 30-min bars for candidates + SPY...")
    all_tickers = runner.active_tickers + ["SPY"]
    bars_by_ticker = await fetch_30min_bars_batch(all_tickers, api_key, secret_key, days=3)

    spy_bars = bars_by_ticker.pop("SPY", pd.DataFrame())
    spy_prev_close = float(data_source.get_prev_close("SPY"))
    vix = await fetch_vix(api_key, secret_key)

    prev_closes = {t: data_source.get_prev_close(t) for t in runner.active_tickers}

    log("Initializing session — classifying regime...")
    init = runner.initialize_session(
        bars_by_ticker=bars_by_ticker, prev_closes=prev_closes,
        spy_bars=spy_bars, spy_prev_close=spy_prev_close, vix=vix,
    )

    regime = init["regime"]
    log(f"Regime: {regime.regime.value.upper()} (confidence {regime.confidence:.0%})")
    log(f"  Position size modifier: {regime.position_size_modifier:.1f}x")
    if regime.suppress_orb:
        log("  ORB suppressed (ranging day)", "WARN")
    if regime.suppress_vwap:
        log("  VWAP reversion suppressed (trending day)", "WARN")
    for sig in regime.signals:
        log(f"  • {sig}")

    for ticker, info in init["init_results"].items():
        status = "ACTIVE" if info["orb_active"] else "SKIPPED"
        reason = info["orb_status"].get("skip_reason", "")
        log(f"  {ticker} ORB: {status}" + (f" — {reason}" if reason else ""))

    log("Entering morning session — scanning for ORB breakouts...")
    afternoon_activated = False

    while True:
        bar_time = await wait_for_next_bar()
        if bar_time >= time(15, 55):
            break

        current_time = now_et()
        phase = get_session_phase(current_time)

        if not afternoon_activated and current_time.time() >= AFTERNOON_ACTIVATE_TIME:
            log("Activating afternoon session (VWAP reversion)...")
            bars_by_ticker = await fetch_30min_bars_batch(runner.active_tickers, api_key, secret_key, days=1)
            runner.activate_afternoon(bars_by_ticker)
            afternoon_activated = True
            log("Afternoon session active.")

        bars_by_ticker = await fetch_30min_bars_batch(runner.active_tickers, api_key, secret_key, days=1)

        log(f"Bar {bar_time.strftime('%H:%M')} — phase={phase.value}")

        for ticker in runner.active_tickers:
            bars = bars_by_ticker.get(ticker, pd.DataFrame())
            if bars.empty:
                continue
            current_price = float(bars["close"].iloc[-1])
            record = runner.process_bar(ticker, bars, current_price, current_time)
            if record:
                log_trade(record)

    log("Session closing. Checking for open positions...")
    close_time = now_et()
    bars_by_ticker = await fetch_30min_bars_batch(runner.active_tickers, api_key, secret_key, days=1)

    for ticker in list(runner.positions.keys()):
        bars = bars_by_ticker.get(ticker, pd.DataFrame())
        if bars.empty:
            continue
        current_price = float(bars["close"].iloc[-1])
        record = runner.process_bar(ticker, bars, current_price, close_time)
        if record:
            log_trade(record)

    summary = runner.get_summary(today)
    print_summary(summary)

    summary_dir = Path("logs")
    summary_dir.mkdir(exist_ok=True)
    summary_path = summary_dir / f"session_{today}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "date": summary.date, "regime": summary.regime,
            "candidates_scanned": summary.candidates_scanned,
            "candidates_passed": summary.candidates_passed,
            "trades_entered": summary.trades_entered,
            "trades_exited": summary.trades_exited,
            "trades_skipped": summary.trades_skipped,
            "trade_log": [
                {"timestamp": r.timestamp.isoformat() if r.timestamp else None,
                 "ticker": r.ticker, "strategy": r.strategy, "action": r.action,
                 "direction": r.direction, "entry_price": r.entry_price,
                 "exit_price": r.exit_price, "exit_reason": r.exit_reason,
                 "qty": r.qty, "risk_usd": r.risk_usd, "regime": r.regime,
                 "reason": r.reason}
                for r in summary.trade_log
            ],
        }, f, indent=2)

    log(f"Session log saved to {summary_path}")
    log("Session complete.")


def main():
    parser = argparse.ArgumentParser(description="Auto-Trader Session Runner")
    parser.add_argument("--dry-run", action="store_true", help="Run pre-market scan only, no trades")
    args = parser.parse_args()

    now = now_et()
    if now.weekday() >= 5:
        log("Market is closed (weekend). Exiting.", "WARN")
        sys.exit(0)

    asyncio.run(run_trading_day(dry_run=args.dry_run))


if __name__ == "__main__":
    main()

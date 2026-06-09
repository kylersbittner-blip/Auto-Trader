"""
Phase 6 tests — Integration.
Run with: pytest tests/test_phase6.py -v
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import zoneinfo

from engine.session_runner import (
    SessionRunner, compute_position_size, TradeRecord, SessionSummary,
)
from engine.scanner import ScannerConfig, MarketDataSource, NewsSource
from engine.regime import RegimeConfig

ET = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")


class MockMarketData:
    def __init__(self, stocks):
        self.stocks = stocks
    def get_premarket_volume(self, ticker): return self.stocks.get(ticker, {}).get("pm_vol", 0)
    def get_avg_daily_volume(self, ticker, lookback_days=20): return self.stocks.get(ticker, {}).get("adv", 0)
    def get_prev_close(self, ticker): return self.stocks.get(ticker, {}).get("prev_close", 0)
    def get_premarket_price(self, ticker): return self.stocks.get(ticker, {}).get("pm_price", 0)
    def get_atr(self, ticker, period=14): return self.stocks.get(ticker, {}).get("atr", 0)


class MockNews:
    def __init__(self, catalysts):
        self.catalysts = catalysts
    def has_catalyst(self, ticker): return self.catalysts.get(ticker) is not None
    def get_headline(self, ticker): return self.catalysts.get(ticker)


def make_session_bars(date_str="2025-06-02", start_price=100.0, moves=None, volumes=None, n_bars=13):
    if moves is None:
        moves = [0.008] + [0.003] * (n_bars - 1)
    if volumes is None:
        volumes = [500_000] * n_bars
    base_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=9, minute=30, tzinfo=ET)
    timestamps = [base_dt + timedelta(minutes=30 * i) for i in range(n_bars)]
    utc_timestamps = [t.astimezone(UTC) for t in timestamps]
    rows = []
    price = start_price
    for i, move in enumerate(moves):
        open_p = price
        close_p = price * (1 + move)
        high_p = max(open_p, close_p) * (1 + abs(move) * 0.3)
        low_p = min(open_p, close_p) * (1 - abs(move) * 0.3)
        rows.append({"open": round(open_p, 4), "high": round(high_p, 4), "low": round(low_p, 4), "close": round(close_p, 4), "volume": volumes[i]})
        price = close_p
    return pd.DataFrame(rows, index=pd.DatetimeIndex(utc_timestamps))


class TestPositionSizing:

    def test_basic_1pct_risk(self):
        result = compute_position_size(account_equity=100_000, entry_price=100.0, stop_price=89.0, risk_pct=0.01)
        assert result["qty"] == 90
        assert result["risk_per_share"] == 11.0
        assert result["total_risk_usd"] == 990.0

    def test_regime_modifier_halves_size(self):
        normal = compute_position_size(account_equity=100_000, entry_price=100.0, stop_price=89.0, risk_pct=0.01, regime_modifier=1.0)
        chaotic = compute_position_size(account_equity=100_000, entry_price=100.0, stop_price=89.0, risk_pct=0.01, regime_modifier=0.5)
        assert chaotic["qty"] == normal["qty"] // 2

    def test_max_position_cap(self):
        result = compute_position_size(account_equity=10_000, entry_price=5.0, stop_price=4.99, risk_pct=0.01, max_position_pct=0.10)
        position_value = result["qty"] * 5.0
        assert position_value <= 10_000 * 0.10 + 5.0

    def test_zero_risk_per_share(self):
        result = compute_position_size(account_equity=10_000, entry_price=100.0, stop_price=100.0)
        assert result["qty"] == 0

    def test_zero_equity(self):
        result = compute_position_size(account_equity=0, entry_price=100.0, stop_price=98.0)
        assert result["qty"] == 0

    def test_risk_pct_of_equity_accurate(self):
        result = compute_position_size(account_equity=100_000, entry_price=150.0, stop_price=135.0, risk_pct=0.01)
        assert result["risk_pct_of_equity"] <= 1.0
        assert result["risk_pct_of_equity"] > 0.8

    def test_short_position_sizing(self):
        result = compute_position_size(account_equity=10_000, entry_price=98.0, stop_price=100.0, risk_pct=0.01)
        assert result["qty"] > 0
        assert result["risk_per_share"] == 2.0


class TestSessionRunnerLifecycle:

    def _make_runner(self):
        stocks = {
            "NVDA": {"pm_vol": 3_000_000, "adv": 1_000_000, "prev_close": 100.0, "pm_price": 103.0, "atr": 2.50},
            "TSLA": {"pm_vol": 2_500_000, "adv": 800_000, "prev_close": 200.0, "pm_price": 206.0, "atr": 5.00},
        }
        catalysts = {"NVDA": "NVIDIA beats earnings", "TSLA": "Tesla deliveries up 20%"}
        return SessionRunner(data_source=MockMarketData(stocks), news_source=MockNews(catalysts), account_equity=10_000, universe=["NVDA", "TSLA"])

    def test_premarket_scan(self):
        runner = self._make_runner()
        result = runner.run_premarket_scan()
        assert result.total_scanned == 2
        assert len(runner.active_tickers) > 0

    def test_initialize_session(self):
        runner = self._make_runner()
        runner.run_premarket_scan()
        bars = make_session_bars(start_price=103.0)
        spy_bars = make_session_bars(start_price=450.0)
        init = runner.initialize_session(
            bars_by_ticker={t: bars for t in runner.active_tickers},
            prev_closes={t: 100.0 for t in runner.active_tickers},
            spy_bars=spy_bars, spy_prev_close=445.0, vix=18.0,
        )
        assert "regime" in init

    def test_process_bar_during_morning(self):
        runner = self._make_runner()
        runner.run_premarket_scan()
        bars = make_session_bars(start_price=103.0)
        runner.initialize_session(
            bars_by_ticker={t: bars for t in runner.active_tickers},
            prev_closes={t: 100.0 for t in runner.active_tickers},
            spy_bars=make_session_bars(start_price=450.0), spy_prev_close=445.0, vix=18.0,
        )
        morning = datetime(2025, 6, 2, 10, 30, tzinfo=ET)
        for ticker in runner.active_tickers:
            runner.process_bar(ticker, bars, 104.0, morning)

    def test_process_bar_during_lunch_no_entry(self):
        runner = self._make_runner()
        runner.run_premarket_scan()
        bars = make_session_bars(start_price=103.0)
        runner.initialize_session(
            bars_by_ticker={t: bars for t in runner.active_tickers},
            prev_closes={t: 100.0 for t in runner.active_tickers}, vix=18.0,
        )
        lunch = datetime(2025, 6, 2, 12, 30, tzinfo=ET)
        for ticker in runner.active_tickers:
            record = runner.process_bar(ticker, bars, 104.0, lunch)
            assert record is None

    def test_position_tracking(self):
        runner = self._make_runner()
        runner.positions["TEST"] = {"direction": "long", "strategy": "orb", "qty": 50, "entry_price": 100.0}
        assert runner.positions["TEST"]["qty"] == 50

    def test_summary_generation(self):
        runner = self._make_runner()
        runner.run_premarket_scan()
        summary = runner.get_summary("2025-06-02")
        assert summary.date == "2025-06-02"

    def test_reset_clears_state(self):
        runner = self._make_runner()
        runner.run_premarket_scan()
        assert len(runner.active_tickers) > 0
        runner.reset()
        assert len(runner.active_tickers) == 0
        assert len(runner.positions) == 0


class TestTradeLogging:

    def test_trade_record_creation(self):
        record = TradeRecord(timestamp=datetime.now(ET), ticker="NVDA", strategy="orb", action="entry", direction="long", entry_price=105.0, stop_price=102.0, target_price=109.5, qty=33, risk_usd=99.0, regime="trending")
        assert record.ticker == "NVDA"
        assert record.qty == 33

    def test_exit_record(self):
        record = TradeRecord(timestamp=datetime.now(ET), ticker="TSLA", strategy="vwap", action="exit", direction="short", exit_price=198.0, exit_reason="target_1", qty=10, regime="ranging")
        assert record.exit_reason == "target_1"

    def test_skip_record(self):
        record = TradeRecord(timestamp=datetime.now(ET), ticker="AAPL", strategy="orb", action="skip", reason="Position size computed to 0 shares", regime="chaotic")
        assert record.action == "skip"

    def test_session_summary(self):
        summary = SessionSummary(date="2025-06-02", regime="trending", candidates_scanned=10, candidates_passed=3, trades_entered=2, trades_exited=2, trades_skipped=1)
        assert summary.trades_entered == 2


class TestEndToEnd:

    def test_full_day_no_crash(self):
        stocks = {"NVDA": {"pm_vol": 3_000_000, "adv": 1_000_000, "prev_close": 100.0, "pm_price": 103.0, "atr": 2.50}}
        catalysts = {"NVDA": "Earnings beat"}
        runner = SessionRunner(data_source=MockMarketData(stocks), news_source=MockNews(catalysts), account_equity=10_000, universe=["NVDA"])
        scan = runner.run_premarket_scan()
        assert scan.total_scanned == 1
        bars = make_session_bars(start_price=103.0)
        spy_bars = make_session_bars(start_price=450.0)
        runner.initialize_session(bars_by_ticker={"NVDA": bars}, prev_closes={"NVDA": 100.0}, spy_bars=spy_bars, spy_prev_close=445.0, vix=18.0)
        bar_times = [(10, 0), (10, 30), (11, 0), (11, 30), (12, 0), (12, 30), (13, 0), (13, 30), (14, 0), (14, 30), (15, 0), (15, 30), (15, 50)]
        afternoon_activated = False
        for hour, minute in bar_times:
            current_time = datetime(2025, 6, 2, hour, minute, tzinfo=ET)
            if hour == 13 and minute == 30 and not afternoon_activated:
                runner.activate_afternoon({"NVDA": bars})
                afternoon_activated = True
            for ticker in runner.active_tickers:
                price = 103.0 + (hour - 10) * 0.5
                runner.process_bar(ticker, bars, price, current_time)
        summary = runner.get_summary("2025-06-02")
        assert summary.date == "2025-06-02"
        assert summary.regime in ("trending", "ranging", "chaotic", "unknown")

    def test_no_candidates_day(self):
        stocks = {"JUNK": {"pm_vol": 1_000, "adv": 50_000, "prev_close": 5.0, "pm_price": 5.01, "atr": 0.10}}
        runner = SessionRunner(data_source=MockMarketData(stocks), news_source=MockNews({"JUNK": None}), account_equity=10_000, universe=["JUNK"])
        scan = runner.run_premarket_scan()
        assert scan.total_passed == 0
        summary = runner.get_summary()
        assert summary.trades_entered == 0

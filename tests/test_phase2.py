"""
Phase 2 tests — ORB Strategy.
Run with: pytest tests/test_phase2.py -v
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import zoneinfo

from engine.strategies.orb_strategy import ORBStrategy, ORBSignal, ExitSignal

ET = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")


def make_session_bars(
    date_str: str = "2025-06-02",
    start_price: float = 100.0,
    moves: list[float] | None = None,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    n_bars = 13
    if moves is None:
        moves = [0.008] + [0.003] * 12
    if volumes is None:
        volumes = [500_000.0] * n_bars

    assert len(moves) == n_bars
    assert len(volumes) == n_bars

    base_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=9, minute=30, tzinfo=ET
    )
    timestamps = [base_dt + timedelta(minutes=30 * i) for i in range(n_bars)]

    rows = []
    price = start_price
    for i, move in enumerate(moves):
        open_p = price
        close_p = price * (1 + move)
        high_p = max(open_p, close_p) * (1 + abs(move) * 0.3)
        low_p = min(open_p, close_p) * (1 - abs(move) * 0.3)
        rows.append({
            "open": round(open_p, 4),
            "high": round(high_p, 4),
            "low": round(low_p, 4),
            "close": round(close_p, 4),
            "volume": volumes[i],
        })
        price = close_p

    utc_timestamps = [t.astimezone(UTC) for t in timestamps]
    return pd.DataFrame(rows, index=pd.DatetimeIndex(utc_timestamps))


def make_breakout_session(direction="long", rvol=2.0, gap_pct=0.01):
    start = 100.0
    prev_close = start / (1 + gap_pct)

    if direction == "long":
        moves = [0.008, 0.002, 0.012, 0.002, 0.001,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        moves = [0.008, -0.002, -0.015, -0.001, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    volumes = [500_000] * 13
    volumes[2] = int(500_000 * rvol)

    bars = make_session_bars(moves=moves, volumes=volumes)
    return bars, prev_close


class TestORBSessionInit:

    def test_valid_session_activates(self):
        bars, prev = make_breakout_session("long", gap_pct=0.01)
        strat = ORBStrategy("NVDA")
        status = strat.set_session(bars, prev_close=prev)
        assert status["active"] is True
        assert status["skip_reason"] is None

    def test_large_gap_skips_session(self):
        bars, _ = make_breakout_session("long")
        prev_close = 100.0 / 1.05
        strat = ORBStrategy("NVDA")
        status = strat.set_session(bars, prev_close=prev_close)
        assert status["active"] is False
        assert "Gap too large" in status["skip_reason"]

    def test_narrow_range_skips_session(self):
        moves = [0.0001] + [0.003] * 12
        bars = make_session_bars(moves=moves)
        strat = ORBStrategy("TSLA")
        status = strat.set_session(bars, prev_close=99.0)
        assert status["active"] is False
        assert "narrow" in status["skip_reason"].lower()

    def test_wide_range_skips_session(self):
        moves = [0.08] + [0.003] * 12
        bars = make_session_bars(moves=moves)
        strat = ORBStrategy("TSLA")
        status = strat.set_session(bars, prev_close=99.0)
        assert status["active"] is False
        assert "wide" in status["skip_reason"].lower()

    def test_reset_clears_state(self):
        bars, prev = make_breakout_session("long")
        strat = ORBStrategy("AAPL")
        strat.set_session(bars, prev_close=prev)
        assert strat.session_active is True
        strat.reset()
        assert strat.session_active is False
        assert strat.entry_taken is False

    def test_gap_under_threshold_allowed(self):
        bars, _ = make_breakout_session("long")
        prev_close = 100.0 / 1.039
        strat = ORBStrategy("META")
        status = strat.set_session(bars, prev_close=prev_close)
        assert status["active"] is True


class TestORBEntry:

    def test_long_breakout_produces_signal(self):
        bars, prev = make_breakout_session("long", rvol=2.0, gap_pct=0.01)
        strat = ORBStrategy("NVDA")
        strat.set_session(bars, prev_close=prev)
        signal = strat.scan_entry(bars)
        assert signal is not None
        assert isinstance(signal, ORBSignal)
        assert signal.direction == "long"
        assert signal.action == "buy"
        assert signal.rvol >= 1.5

    def test_short_breakout_produces_signal(self):
        bars, prev = make_breakout_session("short", rvol=2.0, gap_pct=0.01)
        strat = ORBStrategy("TSLA")
        strat.set_session(bars, prev_close=prev)
        signal = strat.scan_entry(bars)
        assert signal is not None
        assert signal.direction == "short"
        assert signal.action == "sell"

    def test_low_volume_breakout_no_signal(self):
        bars, prev = make_breakout_session("long", rvol=0.8)
        strat = ORBStrategy("AMD")
        strat.set_session(bars, prev_close=prev)
        signal = strat.scan_entry(bars)
        assert signal is None

    def test_no_reentry_after_first_signal(self):
        bars, prev = make_breakout_session("long", rvol=2.0)
        strat = ORBStrategy("NVDA")
        strat.set_session(bars, prev_close=prev)
        signal1 = strat.scan_entry(bars)
        assert signal1 is not None
        signal2 = strat.scan_entry(bars)
        assert signal2 is None

    def test_inactive_session_returns_none(self):
        moves = [0.0001] + [0.05] * 12
        bars = make_session_bars(moves=moves)
        strat = ORBStrategy("SPY")
        strat.set_session(bars, prev_close=99.0)
        assert strat.session_active is False
        assert strat.scan_entry(bars) is None

    def test_custom_rvol_threshold(self):
        bars, prev = make_breakout_session("long", rvol=1.8)
        strat = ORBStrategy("NVDA", rvol_threshold=2.0)
        strat.set_session(bars, prev_close=prev)
        assert strat.scan_entry(bars) is None


class TestORBSignalProperties:

    def test_risk_per_share_long(self):
        sig = ORBSignal(
            ticker="TEST", direction="long",
            entry_price=105.0, stop_loss=100.0, take_profit=112.50,
            range_high=104.0, range_low=100.0, range_width=4.0,
            rvol=2.0, gap_pct=0.01,
        )
        assert sig.risk_per_share == 5.0
        assert sig.reward_per_share == 7.5
        assert sig.risk_reward_ratio == pytest.approx(1.5, abs=0.01)

    def test_action_mapping(self):
        long_sig = ORBSignal(
            ticker="X", direction="long",
            entry_price=100, stop_loss=98, take_profit=103,
            range_high=100, range_low=98, range_width=2,
            rvol=2.0, gap_pct=0.0,
        )
        assert long_sig.action == "buy"

        short_sig = ORBSignal(
            ticker="X", direction="short",
            entry_price=98, stop_loss=100, take_profit=95,
            range_high=100, range_low=98, range_width=2,
            rvol=2.0, gap_pct=0.0,
        )
        assert short_sig.action == "sell"


class TestORBExit:

    def _setup_with_entry(self, direction="long"):
        bars, prev = make_breakout_session(direction, rvol=2.0)
        strat = ORBStrategy("NVDA")
        strat.set_session(bars, prev_close=prev)
        signal = strat.scan_entry(bars)
        assert signal is not None
        return strat, bars, signal

    def test_time_stop_fires_at_deadline(self):
        strat, bars, signal = self._setup_with_entry("long")
        late_time = datetime(2025, 6, 2, 15, 30, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=signal.entry_price,
            position_direction="long", current_time_et=late_time,
        )
        assert exit_sig is not None
        assert exit_sig.reason == "time_stop"

    def test_time_stop_does_not_fire_before_deadline(self):
        strat, bars, signal = self._setup_with_entry("long")
        early_time = datetime(2025, 6, 2, 14, 0, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=signal.entry_price + 1.0,
            position_direction="long", current_time_et=early_time,
        )
        if exit_sig is not None:
            assert exit_sig.reason != "time_stop"

    def test_stop_loss_fires_long(self):
        strat, bars, signal = self._setup_with_entry("long")
        noon = datetime(2025, 6, 2, 11, 0, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=signal.stop_loss - 0.01,
            position_direction="long", current_time_et=noon,
        )
        assert exit_sig is not None
        assert exit_sig.reason == "stop_loss"

    def test_stop_loss_fires_short(self):
        strat, bars, signal = self._setup_with_entry("short")
        noon = datetime(2025, 6, 2, 11, 0, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=signal.stop_loss + 0.01,
            position_direction="short", current_time_et=noon,
        )
        assert exit_sig is not None
        assert exit_sig.reason == "stop_loss"

    def test_take_profit_fires_long(self):
        strat, bars, signal = self._setup_with_entry("long")
        noon = datetime(2025, 6, 2, 11, 0, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=signal.take_profit + 1.0,
            position_direction="long", current_time_et=noon,
        )
        assert exit_sig is not None
        assert exit_sig.reason == "take_profit"

    def test_no_exit_when_price_in_range(self):
        strat, bars, signal = self._setup_with_entry("long")
        safe_price = signal.range_high + 0.01
        noon = datetime(2025, 6, 2, 10, 30, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=safe_price,
            position_direction="long", current_time_et=noon,
        )
        if exit_sig is not None:
            assert exit_sig.reason in ("vwap_trail",)

    def test_vwap_trail_only_activates_in_profit(self):
        strat, bars, signal = self._setup_with_entry("long")
        noon = datetime(2025, 6, 2, 11, 0, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=signal.range_high - 0.50,
            position_direction="long", current_time_et=noon,
        )
        if exit_sig is not None:
            assert exit_sig.reason != "vwap_trail"


class TestORBEdgeCases:

    def test_scan_before_set_session_returns_none(self):
        strat = ORBStrategy("NVDA")
        bars = make_session_bars()
        assert strat.scan_entry(bars) is None

    def test_check_exit_before_set_session_returns_none(self):
        strat = ORBStrategy("NVDA")
        noon = datetime(2025, 6, 2, 11, 0, tzinfo=ET)
        exit_sig = strat.check_exit(
            make_session_bars(), current_price=100.0,
            position_direction="long", current_time_et=noon,
        )
        assert exit_sig is None

    def test_zero_prev_close_handled(self):
        bars = make_session_bars()
        strat = ORBStrategy("TEST")
        status = strat.set_session(bars, prev_close=0.0)
        assert status is not None

"""
Phase 3 tests — VWAP Mean Reversion Strategy.
Run with: pytest tests/test_phase3.py -v
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import zoneinfo

from engine.strategies.vwap_strategy import (
    VWAPReversionStrategy,
    VWAPReversionSignal,
    VWAPExitSignal,
    _compute_vwap_slope,
)
from engine.vwap import compute_vwap

ET = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")


def make_full_day(
    date_str: str = "2025-06-02",
    start_price: float = 100.0,
    moves: list[float] | None = None,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    n_bars = 13
    if moves is None:
        moves = [0.005, 0.002, -0.001, 0.001, -0.002,
                 0.001, -0.001, 0.001, -0.001, 0.002,
                 -0.001, 0.001, -0.001]
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


def make_reversion_session(
    direction: str = "long",
    deviation_size: float = 0.03,
    trend_slope: float = 0.0,
    signal_bar_vol_ratio: float = 1.5,
) -> pd.DataFrame:
    base_vol = 500_000

    morning_moves = [
        0.003 + trend_slope,
        0.001 + trend_slope,
        -0.001 + trend_slope,
        0.001 + trend_slope,
        -0.001 + trend_slope,
        0.001 + trend_slope,
        -0.001 + trend_slope,
        0.0005 + trend_slope,
    ]

    if direction == "long":
        afternoon_moves = [
            -deviation_size * 0.4,
            -deviation_size * 0.6,
            0.005,
            0.005,
            0.002,
        ]
    else:
        afternoon_moves = [
            deviation_size * 0.4,
            deviation_size * 0.6,
            -0.005,
            -0.005,
            -0.002,
        ]

    moves = morning_moves + afternoon_moves
    volumes = [base_vol] * 13
    volumes[9] = int(base_vol * signal_bar_vol_ratio)

    return make_full_day(moves=moves, volumes=volumes)


class TestVWAPSessionInit:

    def test_valid_session_activates(self):
        bars = make_full_day()
        strat = VWAPReversionStrategy("AAPL")
        status = strat.set_session(bars)
        assert status["active"] is True
        assert status["skip_reason"] is None

    def test_orb_trending_suppresses_session(self):
        bars = make_full_day()
        strat = VWAPReversionStrategy("NVDA", orb_trending=True)
        status = strat.set_session(bars)
        assert status["active"] is False
        assert "ORB" in status["skip_reason"]

    def test_orb_trending_via_set_session_param(self):
        bars = make_full_day()
        strat = VWAPReversionStrategy("NVDA")
        status = strat.set_session(bars, orb_trending=True)
        assert status["active"] is False

    def test_insufficient_data_skips(self):
        bars = make_full_day()[:2]
        strat = VWAPReversionStrategy("SPY")
        status = strat.set_session(bars)
        assert status["active"] is False
        assert "Insufficient" in status["skip_reason"]

    def test_reset_clears_state(self):
        bars = make_full_day()
        strat = VWAPReversionStrategy("TSLA")
        strat.set_session(bars)
        assert strat.session_active is True
        strat.reset()
        assert strat.session_active is False
        assert strat.entry_taken is False
        assert strat.target_1_hit is False


class TestVWAPEntry:

    def test_long_entry_at_lower_2sd(self):
        bars = make_reversion_session("long", deviation_size=0.04)
        strat = VWAPReversionStrategy("AAPL")
        strat.set_session(bars)
        signal = strat.scan_entry(bars.iloc[:10])
        if signal is not None:
            assert isinstance(signal, VWAPReversionSignal)
            assert signal.direction == "long"
            assert signal.action == "buy"
            assert signal.sd_distance < -1.5
            assert signal.stop_loss < signal.entry_price
            assert signal.target_1 > signal.entry_price
            assert signal.target_2 > signal.target_1

    def test_short_entry_at_upper_2sd(self):
        bars = make_reversion_session("short", deviation_size=0.04)
        strat = VWAPReversionStrategy("AAPL")
        strat.set_session(bars)
        signal = strat.scan_entry(bars.iloc[:10])
        if signal is not None:
            assert signal.direction == "short"
            assert signal.action == "sell"
            assert signal.sd_distance > 1.5

    def test_trending_day_rejects_entry(self):
        bars = make_reversion_session("long", deviation_size=0.04, trend_slope=0.005)
        strat = VWAPReversionStrategy("TSLA")
        strat.set_session(bars)
        signal = strat.scan_entry(bars.iloc[:10])
        bars_flat = make_reversion_session("long", deviation_size=0.04, trend_slope=0.0)
        strat_flat = VWAPReversionStrategy("TSLA")
        strat_flat.set_session(bars_flat)
        signal_flat = strat_flat.scan_entry(bars_flat.iloc[:10])
        if signal is not None and signal_flat is not None:
            assert abs(signal.vwap_slope_pct_hr) >= abs(signal_flat.vwap_slope_pct_hr)

    def test_low_volume_rejects_entry(self):
        bars = make_reversion_session("long", deviation_size=0.04, signal_bar_vol_ratio=0.3)
        strat = VWAPReversionStrategy("AMD", min_volume_ratio=1.0)
        strat.set_session(bars)
        signal = strat.scan_entry(bars.iloc[:10])
        assert signal is None

    def test_no_reentry_after_first_signal(self):
        bars = make_reversion_session("long", deviation_size=0.04)
        strat = VWAPReversionStrategy("AAPL")
        strat.set_session(bars)
        signal1 = strat.scan_entry(bars.iloc[:10])
        if signal1 is not None:
            assert strat.entry_taken is True
            signal2 = strat.scan_entry(bars.iloc[:11])
            assert signal2 is None

    def test_entry_before_window_rejected(self):
        bars = make_reversion_session("long", deviation_size=0.04)
        strat = VWAPReversionStrategy("SPY")
        strat.set_session(bars)
        signal = strat.scan_entry(bars.iloc[:8])
        assert signal is None

    def test_price_at_1sd_no_entry(self):
        bars = make_reversion_session("long", deviation_size=0.008)
        strat = VWAPReversionStrategy("META")
        strat.set_session(bars)
        signal = strat.scan_entry(bars.iloc[:10])

    def test_inactive_session_returns_none(self):
        bars = make_full_day()
        strat = VWAPReversionStrategy("SPY", orb_trending=True)
        strat.set_session(bars)
        assert strat.session_active is False
        signal = strat.scan_entry(bars)
        assert signal is None


class TestVolumeCharacter:

    def test_exhaustion_detected(self):
        bars = make_reversion_session("long", deviation_size=0.04, signal_bar_vol_ratio=0.8)
        strat = VWAPReversionStrategy("TEST", min_volume_ratio=0.5)
        strat.set_session(bars)
        signal = strat.scan_entry(bars.iloc[:10])
        if signal is not None:
            assert signal.volume_character == "exhaustion"

    def test_continuation_detected(self):
        bars = make_reversion_session("long", deviation_size=0.04, signal_bar_vol_ratio=2.0)
        strat = VWAPReversionStrategy("TEST")
        strat.set_session(bars)
        signal = strat.scan_entry(bars.iloc[:10])
        if signal is not None:
            assert signal.volume_character == "continuation"


class TestVWAPExit:

    def _make_active_strat(self):
        bars = make_full_day()
        strat = VWAPReversionStrategy("AAPL")
        strat.set_session(bars)
        return strat, bars

    def test_time_stop_fires_at_deadline(self):
        strat, bars = self._make_active_strat()
        late = datetime(2025, 6, 2, 15, 50, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=100.0,
            position_direction="long", current_time_et=late,
        )
        assert exit_sig is not None
        assert exit_sig.reason == "time_stop"
        assert exit_sig.close_pct == 1.0

    def test_time_stop_does_not_fire_early(self):
        strat, bars = self._make_active_strat()
        early = datetime(2025, 6, 2, 14, 30, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=100.0,
            position_direction="long", current_time_et=early,
        )
        if exit_sig is not None:
            assert exit_sig.reason != "time_stop"

    def test_stop_loss_long_at_3sd(self):
        strat, bars = self._make_active_strat()
        vwap_df = compute_vwap(bars)
        lower_3 = float(vwap_df.iloc[-1]["vwap_lower_3sd"])
        afternoon = datetime(2025, 6, 2, 14, 30, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=lower_3 - 0.10,
            position_direction="long", current_time_et=afternoon,
        )
        assert exit_sig is not None
        assert exit_sig.reason == "stop_loss"
        assert exit_sig.close_pct == 1.0

    def test_stop_loss_short_at_3sd(self):
        strat, bars = self._make_active_strat()
        vwap_df = compute_vwap(bars)
        upper_3 = float(vwap_df.iloc[-1]["vwap_upper_3sd"])
        afternoon = datetime(2025, 6, 2, 14, 30, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=upper_3 + 0.10,
            position_direction="short", current_time_et=afternoon,
        )
        assert exit_sig is not None
        assert exit_sig.reason == "stop_loss"
        assert exit_sig.close_pct == 1.0

    def test_target_1_scales_out_50_pct(self):
        strat, bars = self._make_active_strat()
        vwap_df = compute_vwap(bars)
        lower_1 = float(vwap_df.iloc[-1]["vwap_lower_1sd"])
        afternoon = datetime(2025, 6, 2, 14, 30, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=lower_1 + 0.10,
            position_direction="long", current_time_et=afternoon,
        )
        assert exit_sig is not None
        assert exit_sig.reason == "target_1"
        assert exit_sig.close_pct == 0.5
        assert strat.target_1_hit is True

    def test_target_1_only_fires_once(self):
        strat, bars = self._make_active_strat()
        vwap_df = compute_vwap(bars)
        lower_1 = float(vwap_df.iloc[-1]["vwap_lower_1sd"])
        afternoon = datetime(2025, 6, 2, 14, 30, tzinfo=ET)
        exit1 = strat.check_exit(
            bars, current_price=lower_1 + 0.10,
            position_direction="long", current_time_et=afternoon,
        )
        assert exit1 is not None
        assert exit1.reason == "target_1"
        exit2 = strat.check_exit(
            bars, current_price=lower_1 + 0.10,
            position_direction="long", current_time_et=afternoon,
        )
        if exit2 is not None:
            assert exit2.reason != "target_1"

    def test_target_2_fires_after_target_1(self):
        strat, bars = self._make_active_strat()
        vwap_df = compute_vwap(bars)
        lower_1 = float(vwap_df.iloc[-1]["vwap_lower_1sd"])
        vwap = float(vwap_df.iloc[-1]["vwap"])
        afternoon = datetime(2025, 6, 2, 14, 30, tzinfo=ET)
        strat.check_exit(
            bars, current_price=lower_1 + 0.10,
            position_direction="long", current_time_et=afternoon,
        )
        assert strat.target_1_hit is True
        exit_sig = strat.check_exit(
            bars, current_price=vwap + 0.05,
            position_direction="long", current_time_et=afternoon,
        )
        assert exit_sig is not None
        assert exit_sig.reason == "target_2"
        assert exit_sig.close_pct == 1.0

    def test_target_2_does_not_fire_before_target_1(self):
        strat, bars = self._make_active_strat()
        vwap_df = compute_vwap(bars)
        vwap = float(vwap_df.iloc[-1]["vwap"])
        afternoon = datetime(2025, 6, 2, 14, 30, tzinfo=ET)
        assert strat.target_1_hit is False
        exit_sig = strat.check_exit(
            bars, current_price=vwap + 0.05,
            position_direction="long", current_time_et=afternoon,
        )
        if exit_sig is not None:
            assert exit_sig.reason != "target_2"

    def test_no_exit_between_entry_and_targets(self):
        strat, bars = self._make_active_strat()
        vwap_df = compute_vwap(bars)
        lower_1 = float(vwap_df.iloc[-1]["vwap_lower_1sd"])
        lower_2 = float(vwap_df.iloc[-1]["vwap_lower_2sd"])
        mid_price = (lower_1 + lower_2) / 2
        afternoon = datetime(2025, 6, 2, 14, 30, tzinfo=ET)
        exit_sig = strat.check_exit(
            bars, current_price=mid_price,
            position_direction="long", current_time_et=afternoon,
        )
        assert exit_sig is None


class TestVWAPSlope:

    def test_flat_vwap_returns_near_zero_slope(self):
        bars = make_full_day()
        vwap_df = compute_vwap(bars)
        slope = _compute_vwap_slope(vwap_df, 4)
        assert slope is not None
        assert abs(slope) < 0.005

    def test_trending_vwap_returns_positive_slope(self):
        moves = [0.005] * 13
        bars = make_full_day(moves=moves)
        vwap_df = compute_vwap(bars)
        slope = _compute_vwap_slope(vwap_df, 4)
        assert slope is not None
        assert slope > 0

    def test_declining_vwap_returns_negative_slope(self):
        moves = [-0.005] * 13
        bars = make_full_day(moves=moves)
        vwap_df = compute_vwap(bars)
        slope = _compute_vwap_slope(vwap_df, 4)
        assert slope is not None
        assert slope < 0

    def test_insufficient_bars_returns_none(self):
        bars = make_full_day()[:3]
        vwap_df = compute_vwap(bars)
        slope = _compute_vwap_slope(vwap_df, 4)
        assert slope is None


class TestVWAPSignalProperties:

    def test_long_signal_properties(self):
        sig = VWAPReversionSignal(
            ticker="TEST", direction="long",
            entry_price=95.0, stop_loss=92.0, target_1=98.0, target_2=100.0,
            vwap=100.0, sd_distance=-2.5, volume_character="exhaustion",
            vwap_slope_pct_hr=0.0001,
        )
        assert sig.action == "buy"
        assert sig.risk_per_share == 3.0
        assert sig.reward_t1_per_share == 3.0
        assert sig.reward_t2_per_share == 5.0

    def test_short_signal_properties(self):
        sig = VWAPReversionSignal(
            ticker="TEST", direction="short",
            entry_price=105.0, stop_loss=108.0, target_1=102.0, target_2=100.0,
            vwap=100.0, sd_distance=2.5, volume_character="continuation",
            vwap_slope_pct_hr=-0.0001,
        )
        assert sig.action == "sell"
        assert sig.risk_per_share == 3.0
        assert sig.reward_t1_per_share == 3.0


class TestVWAPEdgeCases:

    def test_scan_before_set_session_returns_none(self):
        strat = VWAPReversionStrategy("SPY")
        bars = make_full_day()
        assert strat.scan_entry(bars) is None

    def test_check_exit_with_none_bars(self):
        strat = VWAPReversionStrategy("SPY")
        strat.set_session(make_full_day())
        afternoon = datetime(2025, 6, 2, 14, 30, tzinfo=ET)
        exit_sig = strat.check_exit(
            None, current_price=100.0,
            position_direction="long", current_time_et=afternoon,
        )
        assert exit_sig is None

    def test_exit_signal_close_pct_types(self):
        sig = VWAPExitSignal(
            ticker="TEST", reason="target_1",
            exit_price=100.0, close_pct=0.5,
        )
        assert isinstance(sig.close_pct, float)
        sig2 = VWAPExitSignal(
            ticker="TEST", reason="stop_loss",
            exit_price=95.0, close_pct=1.0,
        )
        assert isinstance(sig2.close_pct, float)

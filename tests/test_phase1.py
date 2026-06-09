"""
Phase 1 tests — VWAP calculator and Opening Range tracker.
Run with: pytest tests/test_phase1.py -v
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import zoneinfo

from engine.vwap import compute_vwap, vwap_band_position
from engine.opening_range import (
    identify_opening_range,
    detect_orb_breakout,
    compute_gap,
)

ET = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")


def make_intraday_bars(
    date_str: str = "2025-06-02",
    n_bars: int = 13,
    start_price: float = 100.0,
    moves: list[float] | None = None,
    volumes: list[float] | None = None,
    bar_minutes: int = 30,
) -> pd.DataFrame:
    if moves is None:
        rng = np.random.default_rng(42)
        moves = (rng.standard_normal(n_bars) * 0.003).tolist()
    if volumes is None:
        volumes = [500_000.0] * n_bars

    assert len(moves) == n_bars, f"moves length {len(moves)} != n_bars {n_bars}"
    assert len(volumes) == n_bars, f"volumes length {len(volumes)} != n_bars {n_bars}"

    base_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=9, minute=30, tzinfo=ET
    )
    timestamps = [base_dt + timedelta(minutes=bar_minutes * i) for i in range(n_bars)]

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
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(utc_timestamps))
    return df


class TestVWAP:

    def test_single_bar_vwap_equals_typical_price(self):
        df = make_intraday_bars(n_bars=1, moves=[0.01], volumes=[100_000])
        result = compute_vwap(df)
        bar = result.iloc[0]
        expected_tp = (bar["high"] + bar["low"] + bar["close"]) / 3
        assert bar["vwap"] == pytest.approx(expected_tp, rel=1e-6)

    def test_equal_volume_bars_vwap_is_simple_average(self):
        moves = [0.005, -0.003, 0.002]
        vol = 100_000
        df = make_intraday_bars(n_bars=3, moves=moves, volumes=[vol] * 3)
        result = compute_vwap(df)
        tps = (result["high"] + result["low"] + result["close"]) / 3
        expected_vwap_last = tps.mean()
        assert result.iloc[-1]["vwap"] == pytest.approx(expected_vwap_last, rel=1e-4)

    def test_high_volume_bar_dominates_vwap(self):
        moves = [0.01, -0.01]
        volumes = [100_000, 1_000_000]
        df = make_intraday_bars(n_bars=2, moves=moves, volumes=volumes)
        result = compute_vwap(df)
        tp_0 = (result.iloc[0]["high"] + result.iloc[0]["low"] + result.iloc[0]["close"]) / 3
        tp_1 = (result.iloc[1]["high"] + result.iloc[1]["low"] + result.iloc[1]["close"]) / 3
        final_vwap = result.iloc[-1]["vwap"]
        dist_to_0 = abs(final_vwap - tp_0)
        dist_to_1 = abs(final_vwap - tp_1)
        assert dist_to_1 < dist_to_0

    def test_sd_bands_are_symmetric_around_vwap(self):
        df = make_intraday_bars(n_bars=5)
        result = compute_vwap(df)
        for _, row in result.iterrows():
            vwap = row["vwap"]
            for n in [1, 2, 3]:
                upper = row[f"vwap_upper_{n}sd"]
                lower = row[f"vwap_lower_{n}sd"]
                assert upper - vwap == pytest.approx(vwap - lower, abs=1e-6)

    def test_first_bar_has_zero_sd(self):
        df = make_intraday_bars(n_bars=1, moves=[0.005])
        result = compute_vwap(df)
        bar = result.iloc[0]
        assert bar["vwap_upper_1sd"] == pytest.approx(bar["vwap"], abs=1e-6)
        assert bar["vwap_lower_1sd"] == pytest.approx(bar["vwap"], abs=1e-6)

    def test_bands_widen_with_price_dispersion(self):
        calm = make_intraday_bars(
            n_bars=5, moves=[0.001, -0.001, 0.001, -0.001, 0.001], volumes=[500_000] * 5,
        )
        calm_result = compute_vwap(calm)
        wild = make_intraday_bars(
            n_bars=5, moves=[0.02, -0.02, 0.02, -0.02, 0.02], volumes=[500_000] * 5,
        )
        wild_result = compute_vwap(wild)
        calm_width = calm_result.iloc[-1]["vwap_upper_2sd"] - calm_result.iloc[-1]["vwap_lower_2sd"]
        wild_width = wild_result.iloc[-1]["vwap_upper_2sd"] - wild_result.iloc[-1]["vwap_lower_2sd"]
        assert wild_width > calm_width * 2

    def test_vwap_deviation_sign(self):
        df = make_intraday_bars(
            n_bars=5, moves=[0.01, 0.01, 0.01, 0.01, 0.01], volumes=[500_000] * 5,
        )
        result = compute_vwap(df)
        assert result.iloc[-1]["vwap_deviation"] > 0

    def test_multi_day_vwap_resets(self):
        day1 = make_intraday_bars(date_str="2025-06-02", n_bars=3, start_price=100.0, moves=[0.05, 0.05, 0.05])
        day2 = make_intraday_bars(date_str="2025-06-03", n_bars=3, start_price=50.0, moves=[0.001, 0.001, 0.001])
        combined = pd.concat([day1, day2])
        result = compute_vwap(combined)
        day2_vwap = result.iloc[-1]["vwap"]
        assert day2_vwap < 55

    def test_requires_correct_columns(self):
        df = pd.DataFrame({"close": [100]}, index=pd.DatetimeIndex(["2025-01-01"]))
        with pytest.raises(ValueError, match="must have columns"):
            compute_vwap(df)

    def test_requires_datetime_index(self):
        df = pd.DataFrame({"open": [100], "high": [101], "low": [99], "close": [100.5], "volume": [1000]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_vwap(df)

    def test_empty_dataframe_passes_through(self):
        result = compute_vwap(pd.DataFrame())
        assert result.empty

    def test_none_passes_through(self):
        assert compute_vwap(None) is None


class TestVWAPBandPosition:

    def test_above_2sd_returns_short_signal(self):
        row = pd.Series({
            "close": 110, "vwap": 100,
            "vwap_upper_1sd": 103, "vwap_upper_2sd": 106, "vwap_upper_3sd": 109,
            "vwap_lower_1sd": 97, "vwap_lower_2sd": 94, "vwap_lower_3sd": 91,
        })
        result = vwap_band_position(row)
        assert result["zone"] == "above_3sd"
        assert result["reversion_signal"] == "short"

    def test_below_2sd_returns_long_signal(self):
        row = pd.Series({
            "close": 93, "vwap": 100,
            "vwap_upper_1sd": 103, "vwap_upper_2sd": 106, "vwap_upper_3sd": 109,
            "vwap_lower_1sd": 97, "vwap_lower_2sd": 94, "vwap_lower_3sd": 91,
        })
        result = vwap_band_position(row)
        assert result["zone"] == "below_2sd"
        assert result["reversion_signal"] == "long"

    def test_near_vwap_returns_no_signal(self):
        row = pd.Series({
            "close": 100.5, "vwap": 100,
            "vwap_upper_1sd": 103, "vwap_upper_2sd": 106, "vwap_upper_3sd": 109,
            "vwap_lower_1sd": 97, "vwap_lower_2sd": 94, "vwap_lower_3sd": 91,
        })
        result = vwap_band_position(row)
        assert result["zone"] == "near_vwap"
        assert result["reversion_signal"] == "none"


class TestOpeningRange:

    def test_identifies_first_bar_as_opening_range(self):
        df = make_intraday_bars(start_price=150.0, moves=[0.01] + [0.005] * 12)
        result = identify_opening_range(df)
        assert result is not None
        assert result["range_high"] == df.iloc[0]["high"]
        assert result["range_low"] == df.iloc[0]["low"]

    def test_range_width_calculation(self):
        df = make_intraday_bars(start_price=100.0, moves=[0.01] + [0.0] * 12)
        result = identify_opening_range(df)
        expected_width = result["range_high"] - result["range_low"]
        assert result["range_width"] == pytest.approx(expected_width, abs=1e-4)

    def test_narrow_range_rejected(self):
        df = make_intraday_bars(start_price=100.0, moves=[0.0001] + [0.0] * 12)
        result = identify_opening_range(df)
        assert result is not None
        assert result["is_valid"] is False
        assert "narrow" in result["rejection_reason"]

    def test_wide_range_rejected(self):
        df = make_intraday_bars(start_price=100.0, moves=[0.08] + [0.0] * 12)
        result = identify_opening_range(df)
        assert result is not None
        assert result["is_valid"] is False
        assert "wide" in result["rejection_reason"]

    def test_valid_range_accepted(self):
        df = make_intraday_bars(start_price=100.0, moves=[0.008] + [0.003] * 12)
        result = identify_opening_range(df)
        assert result is not None
        assert result["is_valid"] is True
        assert result["rejection_reason"] is None

    def test_returns_none_for_empty_df(self):
        assert identify_opening_range(pd.DataFrame()) is None

    def test_returns_none_for_none(self):
        assert identify_opening_range(None) is None


class TestORBBreakout:

    def _make_breakout_scenario(self, direction="long", rvol_multiplier=2.0):
        if direction == "long":
            moves = [0.008, 0.002, 0.012, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            moves = [0.008, -0.002, -0.015, -0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        base_vol = 500_000
        volumes = [base_vol] * 13
        volumes[2] = base_vol * rvol_multiplier
        df = make_intraday_bars(start_price=100.0, moves=moves, volumes=volumes)
        opening_range = identify_opening_range(df)
        return df, opening_range

    def test_long_breakout_detected(self):
        df, orng = self._make_breakout_scenario("long", rvol_multiplier=2.0)
        result = detect_orb_breakout(df, orng, rvol_threshold=1.5)
        assert result is not None
        assert result["direction"] == "long"
        assert result["breakout_price"] > orng["range_high"]
        assert result["rvol"] >= 1.5

    def test_short_breakout_detected(self):
        df, orng = self._make_breakout_scenario("short", rvol_multiplier=2.0)
        result = detect_orb_breakout(df, orng, rvol_threshold=1.5)
        assert result is not None
        assert result["direction"] == "short"
        assert result["breakout_price"] < orng["range_low"]

    def test_low_volume_breakout_rejected(self):
        df, orng = self._make_breakout_scenario("long", rvol_multiplier=0.8)
        result = detect_orb_breakout(df, orng, rvol_threshold=1.5)
        assert result is None

    def test_invalid_range_returns_none(self):
        df = make_intraday_bars(start_price=100.0, moves=[0.0001] + [0.05] * 12)
        orng = identify_opening_range(df)
        assert orng is not None and orng["is_valid"] is False
        result = detect_orb_breakout(df, orng)
        assert result is None

    def test_stop_loss_placement_long(self):
        df, orng = self._make_breakout_scenario("long")
        result = detect_orb_breakout(df, orng, rvol_threshold=1.5)
        assert result is not None
        assert result["stop_loss"] <= orng["range_high"]
        assert result["stop_loss"] >= orng["range_low"] - 0.01

    def test_take_profit_is_1_5x_range(self):
        df, orng = self._make_breakout_scenario("long")
        result = detect_orb_breakout(df, orng, rvol_threshold=1.5)
        if result is not None:
            expected_tp = result["entry_price"] + 1.5 * orng["range_width"]
            assert result["take_profit"] == pytest.approx(expected_tp, abs=0.02)


class TestGap:

    def test_gap_up(self):
        result = compute_gap(current_open=105.0, prev_close=100.0)
        assert result["gap_pct"] == pytest.approx(0.05, abs=1e-6)
        assert result["is_large_gap"] is True

    def test_gap_down(self):
        result = compute_gap(current_open=97.0, prev_close=100.0)
        assert result["gap_pct"] == pytest.approx(-0.03, abs=1e-6)
        assert result["is_large_gap"] is False

    def test_no_gap(self):
        result = compute_gap(current_open=100.0, prev_close=100.0)
        assert result["gap_pct"] == 0.0
        assert result["is_large_gap"] is False

    def test_small_gap_under_threshold(self):
        result = compute_gap(current_open=102.0, prev_close=100.0)
        assert result["gap_pct"] == pytest.approx(0.02, abs=1e-6)
        assert result["is_large_gap"] is False

    def test_zero_prev_close_safe(self):
        result = compute_gap(current_open=100.0, prev_close=0.0)
        assert result["gap_pct"] == 0.0

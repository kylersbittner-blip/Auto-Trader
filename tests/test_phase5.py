"""
Phase 5 tests — Regime Selector and Session Manager.
Run with: pytest tests/test_phase5.py -v
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import zoneinfo

from engine.regime import (
    classify_regime,
    get_session_phase,
    DayRegime,
    SessionPhase,
    RegimeConfig,
    RegimeClassification,
    SessionManager,
    BarAction,
)

ET = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")


def make_session_bars(date_str="2025-06-02", start_price=100.0, n_bars=13):
    base_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=9, minute=30, tzinfo=ET)
    timestamps = [base_dt + timedelta(minutes=30 * i) for i in range(n_bars)]
    utc_timestamps = [t.astimezone(UTC) for t in timestamps]
    price = start_price
    rows = []
    for i in range(n_bars):
        move = 0.003
        open_p = price
        close_p = price * (1 + move)
        high_p = max(open_p, close_p) * 1.001
        low_p = min(open_p, close_p) * 0.999
        rows.append({"open": round(open_p, 4), "high": round(high_p, 4), "low": round(low_p, 4), "close": round(close_p, 4), "volume": 500_000})
        price = close_p
    return pd.DataFrame(rows, index=pd.DatetimeIndex(utc_timestamps))


def make_et_time(hour, minute=0):
    return datetime(2025, 6, 2, hour, minute, tzinfo=ET)


class TestRegimeClassification:

    def test_trending_day(self):
        result = classify_regime(spy_range_width_pct=0.008, spy_gap_pct=0.012, spy_breakout=True, vix=20.0)
        assert result.regime == DayRegime.TRENDING
        assert result.suppress_vwap is True
        assert result.suppress_orb is False
        assert result.position_size_modifier == 1.0

    def test_ranging_day(self):
        result = classify_regime(spy_range_width_pct=0.002, spy_gap_pct=0.003, spy_breakout=False, vix=12.0)
        assert result.regime == DayRegime.RANGING
        assert result.suppress_orb is True
        assert result.suppress_vwap is False

    def test_chaotic_day_vix_extreme(self):
        result = classify_regime(spy_range_width_pct=0.008, spy_gap_pct=0.01, spy_breakout=True, vix=35.0)
        assert result.regime == DayRegime.CHAOTIC
        assert result.position_size_modifier == 0.5

    def test_chaotic_day_wide_range(self):
        result = classify_regime(spy_range_width_pct=0.015, vix=18.0)
        assert result.regime == DayRegime.CHAOTIC

    def test_chaotic_overrides_trending_signals(self):
        result = classify_regime(spy_range_width_pct=0.015, spy_gap_pct=0.02, spy_breakout=True, vix=32.0)
        assert result.regime == DayRegime.CHAOTIC

    def test_unknown_with_no_data(self):
        result = classify_regime()
        assert result.regime == DayRegime.UNKNOWN
        assert result.confidence == 0.0
        assert result.position_size_modifier == 0.75

    def test_tie_breaks_to_trending(self):
        result = classify_regime(spy_range_width_pct=0.006, spy_gap_pct=0.003)
        assert result.regime == DayRegime.TRENDING
        assert result.confidence == 0.5

    def test_position_size_chaotic_is_half(self):
        result = classify_regime(vix=35.0)
        assert result.position_size_modifier == 0.5

    def test_position_size_trending_is_full(self):
        result = classify_regime(spy_breakout=True, spy_gap_pct=0.015, vix=20.0)
        assert result.position_size_modifier == 1.0

    def test_custom_config_thresholds(self):
        config = RegimeConfig(vix_high=25.0)
        result = classify_regime(vix=27.0, config=config)
        assert result.regime == DayRegime.CHAOTIC

    def test_signals_list_populated(self):
        result = classify_regime(spy_range_width_pct=0.008, vix=20.0, spy_breakout=True)
        assert len(result.signals) > 0
        assert any("VIX" in s for s in result.signals)


class TestSessionPhase:

    def test_pre_market(self):
        assert get_session_phase(make_et_time(8, 0)) == SessionPhase.PRE_MARKET

    def test_opening_range(self):
        assert get_session_phase(make_et_time(9, 30)) == SessionPhase.OPENING_RANGE
        assert get_session_phase(make_et_time(9, 55)) == SessionPhase.OPENING_RANGE

    def test_morning(self):
        assert get_session_phase(make_et_time(10, 0)) == SessionPhase.MORNING
        assert get_session_phase(make_et_time(11, 30)) == SessionPhase.MORNING

    def test_lunch(self):
        assert get_session_phase(make_et_time(12, 0)) == SessionPhase.LUNCH
        assert get_session_phase(make_et_time(13, 0)) == SessionPhase.LUNCH

    def test_afternoon(self):
        assert get_session_phase(make_et_time(13, 30)) == SessionPhase.AFTERNOON
        assert get_session_phase(make_et_time(15, 0)) == SessionPhase.AFTERNOON

    def test_closing(self):
        assert get_session_phase(make_et_time(15, 30)) == SessionPhase.CLOSING
        assert get_session_phase(make_et_time(15, 55)) == SessionPhase.CLOSING

    def test_closed(self):
        assert get_session_phase(make_et_time(16, 0)) == SessionPhase.CLOSED
        assert get_session_phase(make_et_time(17, 0)) == SessionPhase.CLOSED

    def test_boundary_precision(self):
        assert get_session_phase(make_et_time(10, 0)) == SessionPhase.MORNING
        assert get_session_phase(make_et_time(12, 0)) == SessionPhase.LUNCH
        assert get_session_phase(make_et_time(13, 30)) == SessionPhase.AFTERNOON


class TestSessionManager:

    def _make_manager_with_ticker(self):
        mgr = SessionManager()
        bars = make_session_bars()
        prev_closes = {"TEST": 99.0}
        mgr.initialize_strategies(["TEST"], {"TEST": bars}, prev_closes)
        mgr.set_regime(spy_range_width_pct=0.008, spy_gap_pct=0.01, vix=20.0)
        return mgr, bars

    def test_initialize_creates_strategies(self):
        mgr = SessionManager()
        bars = make_session_bars()
        result = mgr.initialize_strategies(["NVDA", "TSLA"], {"NVDA": bars, "TSLA": bars}, {"NVDA": 99.0, "TSLA": 200.0})
        assert "NVDA" in result
        assert "TSLA" in result

    def test_morning_routes_to_orb(self):
        mgr, bars = self._make_manager_with_ticker()
        action = mgr.on_bar("TEST", bars, current_price=101.0, current_time_et=make_et_time(10, 30))
        assert action.phase == SessionPhase.MORNING
        assert action.action == "scan_orb"

    def test_afternoon_routes_to_vwap(self):
        mgr, bars = self._make_manager_with_ticker()
        mgr.set_regime(spy_range_width_pct=0.002, spy_gap_pct=0.003, vix=12.0)
        mgr.activate_afternoon_session({"TEST": bars})
        action = mgr.on_bar("TEST", bars, current_price=101.0, current_time_et=make_et_time(14, 0))
        assert action.phase == SessionPhase.AFTERNOON
        assert action.action in ("scan_vwap", "suppressed")

    def test_lunch_returns_no_action(self):
        mgr, bars = self._make_manager_with_ticker()
        action = mgr.on_bar("TEST", bars, current_price=101.0, current_time_et=make_et_time(12, 30))
        assert action.phase == SessionPhase.LUNCH
        assert action.action == "no_action"

    def test_closing_returns_no_action(self):
        mgr, bars = self._make_manager_with_ticker()
        action = mgr.on_bar("TEST", bars, current_price=101.0, current_time_et=make_et_time(15, 45))
        assert action.phase == SessionPhase.CLOSING
        assert action.action == "no_action"

    def test_regime_suppresses_orb_on_ranging(self):
        mgr, bars = self._make_manager_with_ticker()
        mgr.set_regime(spy_range_width_pct=0.002, spy_gap_pct=0.003, vix=12.0)
        action = mgr.on_bar("TEST", bars, current_price=101.0, current_time_et=make_et_time(10, 30))
        assert action.action == "suppressed"
        assert "ranging" in action.reason.lower()

    def test_regime_suppresses_vwap_on_trending(self):
        mgr, bars = self._make_manager_with_ticker()
        mgr.set_regime(spy_range_width_pct=0.008, spy_gap_pct=0.015, spy_breakout=True, vix=20.0)
        result = mgr.activate_afternoon_session({"TEST": bars})
        assert result["TEST"]["vwap_status"]["active"] is False

    def test_exit_check_works_during_lunch(self):
        mgr, bars = self._make_manager_with_ticker()
        action = mgr.on_bar(
            "TEST", bars, current_price=90.0, current_time_et=make_et_time(12, 30),
            position_direction="long", position_strategy="orb",
        )
        assert action.action in ("check_exit", "no_action")

    def test_mark_orb_exited(self):
        mgr, bars = self._make_manager_with_ticker()
        mgr._orb_active["TEST"] = True
        mgr.mark_orb_exited("TEST")
        assert mgr._orb_active["TEST"] is False

    def test_position_size_modifier_passed_through(self):
        mgr, bars = self._make_manager_with_ticker()
        mgr.set_regime(vix=35.0)
        action = mgr.on_bar("TEST", bars, current_price=101.0, current_time_et=make_et_time(10, 30))
        assert action.position_size_modifier == 0.5

    def test_reset_clears_everything(self):
        mgr, bars = self._make_manager_with_ticker()
        assert len(mgr._orb_strategies) > 0
        mgr.reset()
        assert len(mgr._orb_strategies) == 0
        assert mgr.regime is None

    def test_missing_ticker_returns_no_action(self):
        mgr = SessionManager()
        mgr.set_regime(vix=20.0)
        action = mgr.on_bar("UNKNOWN", make_session_bars(), current_price=100.0, current_time_et=make_et_time(10, 30))
        assert action.action in ("no_action", "suppressed")

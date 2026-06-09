"""
Time-Adaptive Regime Selector and Session Manager.

Orchestrates a single trading day by:
  1. Classifying the market regime at 10:00 ET (trending / ranging / chaotic)
  2. Routing bars to ORB (morning) or VWAP reversion (afternoon)
  3. Adjusting position sizing based on regime
  4. Handing off ORB trending state to VWAP strategy
  5. Re-evaluating regime at 13:00 ET before the afternoon session

Rules-based for v1. ML classifier earns its way in after real outcome data.
"""
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional, Union
import zoneinfo

import pandas as pd

from engine.strategies.orb_strategy import ORBStrategy, ORBSignal, ExitSignal as ORBExitSignal
from engine.strategies.vwap_strategy import (
    VWAPReversionStrategy,
    VWAPReversionSignal,
    VWAPExitSignal,
)
from engine.opening_range import identify_opening_range
from engine.vwap import compute_vwap

ET = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")


class DayRegime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    CHAOTIC = "chaotic"
    UNKNOWN = "unknown"


class SessionPhase(str, Enum):
    PRE_MARKET = "pre_market"
    OPENING_RANGE = "opening_range"
    MORNING = "morning"
    LUNCH = "lunch"
    AFTERNOON = "afternoon"
    CLOSING = "closing"
    CLOSED = "closed"


PHASE_TIMES = {
    SessionPhase.PRE_MARKET: time(0, 0),
    SessionPhase.OPENING_RANGE: time(9, 30),
    SessionPhase.MORNING: time(10, 0),
    SessionPhase.LUNCH: time(12, 0),
    SessionPhase.AFTERNOON: time(13, 30),
    SessionPhase.CLOSING: time(15, 30),
    SessionPhase.CLOSED: time(16, 0),
}


@dataclass
class RegimeConfig:
    spy_narrow_range_pct: float = 0.003
    spy_wide_range_pct: float = 0.012
    spy_small_gap_pct: float = 0.005
    vix_low: float = 15.0
    vix_high: float = 30.0
    size_trending: float = 1.0
    size_ranging: float = 1.0
    size_chaotic: float = 0.5


@dataclass
class RegimeClassification:
    regime: DayRegime
    confidence: float
    signals: list[str] = field(default_factory=list)
    position_size_modifier: float = 1.0
    suppress_orb: bool = False
    suppress_vwap: bool = False


def classify_regime(
    spy_range_width_pct: Optional[float] = None,
    spy_gap_pct: Optional[float] = None,
    spy_breakout: bool = False,
    vix: Optional[float] = None,
    config: Optional[RegimeConfig] = None,
) -> RegimeClassification:
    cfg = config or RegimeConfig()
    signals_trending = []
    signals_ranging = []
    signals_chaotic = []

    if vix is not None:
        if vix > cfg.vix_high:
            signals_chaotic.append(f"VIX {vix:.1f} > {cfg.vix_high} (extreme volatility)")
        elif vix > cfg.vix_low:
            signals_trending.append(f"VIX {vix:.1f} in {cfg.vix_low}–{cfg.vix_high} range (elevated)")
        else:
            signals_ranging.append(f"VIX {vix:.1f} < {cfg.vix_low} (complacent)")

    if spy_gap_pct is not None:
        abs_gap = abs(spy_gap_pct)
        if abs_gap < cfg.spy_small_gap_pct:
            signals_ranging.append(f"SPY gap {spy_gap_pct:+.2%} — small (no catalyst)")
        else:
            signals_trending.append(f"SPY gap {spy_gap_pct:+.2%} — meaningful catalyst")

    if spy_range_width_pct is not None:
        if spy_range_width_pct < cfg.spy_narrow_range_pct:
            signals_ranging.append(f"SPY range {spy_range_width_pct:.3%} — narrow (no conviction)")
        elif spy_range_width_pct > cfg.spy_wide_range_pct:
            signals_chaotic.append(f"SPY range {spy_range_width_pct:.3%} — extremely wide")
        else:
            signals_trending.append(f"SPY range {spy_range_width_pct:.3%} — healthy width")

    if spy_breakout:
        signals_trending.append("SPY broke opening range with volume")

    counts = {
        DayRegime.TRENDING: len(signals_trending),
        DayRegime.RANGING: len(signals_ranging),
        DayRegime.CHAOTIC: len(signals_chaotic),
    }

    if counts[DayRegime.CHAOTIC] > 0:
        regime = DayRegime.CHAOTIC
        all_signals = signals_chaotic + signals_trending + signals_ranging
        confidence = min(counts[DayRegime.CHAOTIC] / 3.0, 1.0)
        return RegimeClassification(
            regime=regime, confidence=round(confidence, 2),
            signals=all_signals, position_size_modifier=cfg.size_chaotic,
            suppress_orb=False, suppress_vwap=False,
        )

    total = counts[DayRegime.TRENDING] + counts[DayRegime.RANGING]
    if total == 0:
        return RegimeClassification(
            regime=DayRegime.UNKNOWN, confidence=0.0,
            signals=["No regime data available"],
            position_size_modifier=0.75,
            suppress_orb=False, suppress_vwap=False,
        )

    if counts[DayRegime.TRENDING] > counts[DayRegime.RANGING]:
        return RegimeClassification(
            regime=DayRegime.TRENDING,
            confidence=round(counts[DayRegime.TRENDING] / total, 2),
            signals=signals_trending + signals_ranging,
            position_size_modifier=cfg.size_trending,
            suppress_orb=False, suppress_vwap=True,
        )
    elif counts[DayRegime.RANGING] > counts[DayRegime.TRENDING]:
        return RegimeClassification(
            regime=DayRegime.RANGING,
            confidence=round(counts[DayRegime.RANGING] / total, 2),
            signals=signals_ranging + signals_trending,
            position_size_modifier=cfg.size_ranging,
            suppress_orb=True, suppress_vwap=False,
        )
    else:
        return RegimeClassification(
            regime=DayRegime.TRENDING, confidence=0.5,
            signals=signals_trending + signals_ranging,
            position_size_modifier=cfg.size_trending,
            suppress_orb=False, suppress_vwap=False,
        )


def get_session_phase(current_time_et: datetime) -> SessionPhase:
    t = current_time_et.time() if hasattr(current_time_et, 'time') else current_time_et

    if t >= time(16, 0):
        return SessionPhase.CLOSED
    elif t >= time(15, 30):
        return SessionPhase.CLOSING
    elif t >= time(13, 30):
        return SessionPhase.AFTERNOON
    elif t >= time(12, 0):
        return SessionPhase.LUNCH
    elif t >= time(10, 0):
        return SessionPhase.MORNING
    elif t >= time(9, 30):
        return SessionPhase.OPENING_RANGE
    else:
        return SessionPhase.PRE_MARKET


@dataclass
class BarAction:
    phase: SessionPhase
    regime: DayRegime
    action: str
    signal: Optional[Union[ORBSignal, VWAPReversionSignal]] = None
    exit_signal: Optional[Union[ORBExitSignal, VWAPExitSignal]] = None
    position_size_modifier: float = 1.0
    reason: Optional[str] = None


class SessionManager:
    def __init__(self, regime_config: Optional[RegimeConfig] = None):
        self.regime_config = regime_config or RegimeConfig()
        self.regime: Optional[RegimeClassification] = None
        self.phase: SessionPhase = SessionPhase.PRE_MARKET
        self._orb_strategies: dict[str, ORBStrategy] = {}
        self._vwap_strategies: dict[str, VWAPReversionStrategy] = {}
        self._orb_active: dict[str, bool] = {}

    def reset(self) -> None:
        self.regime = None
        self.phase = SessionPhase.PRE_MARKET
        self._orb_strategies.clear()
        self._vwap_strategies.clear()
        self._orb_active.clear()

    def initialize_strategies(
        self, tickers, bars_by_ticker, prev_closes,
    ) -> dict[str, dict]:
        results = {}
        for ticker in tickers:
            bars = bars_by_ticker.get(ticker)
            prev_close = prev_closes.get(ticker, 0.0)
            if bars is None or bars.empty:
                results[ticker] = {"orb_status": {"active": False, "skip_reason": "No data"}, "orb_active": False}
                continue
            orb = ORBStrategy(ticker)
            orb_status = orb.set_session(bars, prev_close=prev_close)
            self._orb_strategies[ticker] = orb
            self._orb_active[ticker] = orb_status["active"]
            vwap = VWAPReversionStrategy(ticker)
            self._vwap_strategies[ticker] = vwap
            results[ticker] = {"orb_status": orb_status, "orb_active": orb_status["active"]}
        return results

    def set_regime(
        self, spy_range_width_pct=None, spy_gap_pct=None,
        spy_breakout=False, vix=None,
    ) -> RegimeClassification:
        self.regime = classify_regime(
            spy_range_width_pct=spy_range_width_pct,
            spy_gap_pct=spy_gap_pct,
            spy_breakout=spy_breakout,
            vix=vix,
            config=self.regime_config,
        )
        return self.regime

    def activate_afternoon_session(self, bars_by_ticker) -> dict[str, dict]:
        results = {}
        suppress_vwap = self.regime.suppress_vwap if self.regime else False
        for ticker, vwap_strat in self._vwap_strategies.items():
            bars = bars_by_ticker.get(ticker)
            if bars is None or bars.empty:
                results[ticker] = {"vwap_status": {"active": False, "skip_reason": "No data"}}
                continue
            orb_trending = self._orb_active.get(ticker, False)
            if suppress_vwap:
                vwap_strat.reset()
                vwap_strat._skip_reason = f"VWAP suppressed by {self.regime.regime.value} regime"
                results[ticker] = {"vwap_status": {"active": False, "skip_reason": vwap_strat._skip_reason}}
                continue
            vwap_status = vwap_strat.set_session(bars, orb_trending=orb_trending)
            results[ticker] = {"vwap_status": vwap_status}
        return results

    def on_bar(
        self, ticker, bars, current_price, current_time_et,
        position_direction=None, position_strategy=None,
    ) -> BarAction:
        phase = get_session_phase(current_time_et)
        self.phase = phase
        regime = self.regime.regime if self.regime else DayRegime.UNKNOWN
        size_mod = self.regime.position_size_modifier if self.regime else 1.0

        if position_direction and position_strategy:
            exit_sig = self._check_exit(
                ticker, bars, current_price,
                position_direction, position_strategy, current_time_et,
            )
            if exit_sig:
                return BarAction(
                    phase=phase, regime=regime, action="check_exit",
                    exit_signal=exit_sig, position_size_modifier=size_mod,
                )

        if phase == SessionPhase.MORNING:
            return self._handle_morning(ticker, bars, phase, regime, size_mod)
        elif phase == SessionPhase.AFTERNOON:
            return self._handle_afternoon(ticker, bars, phase, regime, size_mod)
        elif phase in (SessionPhase.LUNCH, SessionPhase.OPENING_RANGE):
            return BarAction(
                phase=phase, regime=regime, action="no_action",
                position_size_modifier=size_mod,
                reason=f"Dead zone ({phase.value}) — no new entries",
            )
        elif phase in (SessionPhase.CLOSING, SessionPhase.CLOSED):
            return BarAction(
                phase=phase, regime=regime, action="no_action",
                position_size_modifier=size_mod,
                reason="Market closing — exits only",
            )
        return BarAction(phase=phase, regime=regime, action="no_action", position_size_modifier=size_mod)

    def mark_orb_exited(self, ticker: str) -> None:
        self._orb_active[ticker] = False

    def _handle_morning(self, ticker, bars, phase, regime, size_mod):
        orb = self._orb_strategies.get(ticker)
        if orb is None:
            return BarAction(
                phase=phase, regime=regime, action="no_action",
                reason="No ORB strategy for ticker", position_size_modifier=size_mod,
            )
        if self.regime and self.regime.suppress_orb:
            return BarAction(
                phase=phase, regime=regime, action="suppressed",
                reason=f"ORB suppressed by {regime.value} regime",
                position_size_modifier=size_mod,
            )
        signal = orb.scan_entry(bars)
        if signal:
            self._orb_active[ticker] = True
            return BarAction(
                phase=phase, regime=regime, action="scan_orb",
                signal=signal, position_size_modifier=size_mod,
            )
        return BarAction(
            phase=phase, regime=regime, action="scan_orb",
            reason="No ORB breakout detected", position_size_modifier=size_mod,
        )

    def _handle_afternoon(self, ticker, bars, phase, regime, size_mod):
        vwap = self._vwap_strategies.get(ticker)
        if vwap is None:
            return BarAction(
                phase=phase, regime=regime, action="no_action",
                reason="No VWAP strategy for ticker", position_size_modifier=size_mod,
            )
        if not vwap.session_active:
            return BarAction(
                phase=phase, regime=regime, action="suppressed",
                reason=vwap.skip_reason or "VWAP session not active",
                position_size_modifier=size_mod,
            )
        signal = vwap.scan_entry(bars)
        if signal:
            return BarAction(
                phase=phase, regime=regime, action="scan_vwap",
                signal=signal, position_size_modifier=size_mod,
            )
        return BarAction(
            phase=phase, regime=regime, action="scan_vwap",
            reason="No VWAP reversion signal", position_size_modifier=size_mod,
        )

    def _check_exit(self, ticker, bars, current_price, position_direction, position_strategy, current_time_et):
        if position_strategy == "orb":
            orb = self._orb_strategies.get(ticker)
            if orb:
                return orb.check_exit(bars, current_price, position_direction, current_time_et)
        elif position_strategy == "vwap":
            vwap = self._vwap_strategies.get(ticker)
            if vwap:
                return vwap.check_exit(bars, current_price, position_direction, current_time_et)
        return None

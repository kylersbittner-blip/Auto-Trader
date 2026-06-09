"""
Opening Range Breakout (ORB) Strategy.

Implements the full strategy from the Auto-Trader Blueprint:
  - Captures the 9:30-10:00 ET opening range
  - Scans for breakouts with volume confirmation (RVOL >= 1.5x)
  - Applies gap filter (>4% gap = skip day for ORB)
  - Applies range width filter (0.3%-2.0%)
  - No entries after 12:00 ET
  - VWAP trailing stop for exit management
  - Time stop: all positions flat by 15:30 ET
"""
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional
import zoneinfo

import pandas as pd

from engine.opening_range import (
    identify_opening_range,
    detect_orb_breakout,
    compute_gap,
)
from engine.vwap import compute_vwap

ET = zoneinfo.ZoneInfo("America/New_York")

ORB_ENTRY_OPEN = time(10, 0)
ORB_ENTRY_CLOSE = time(12, 0)
ORB_EXIT_DEADLINE = time(15, 30)


@dataclass
class ORBSignal:
    ticker: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    range_high: float
    range_low: float
    range_width: float
    rvol: float
    gap_pct: float
    confidence_factors: list[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    @property
    def action(self) -> str:
        return "buy" if self.direction == "long" else "sell"

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_per_share(self) -> float:
        return abs(self.take_profit - self.entry_price)

    @property
    def risk_reward_ratio(self) -> float:
        risk = self.risk_per_share
        return self.reward_per_share / risk if risk > 0 else 0.0


@dataclass
class ExitSignal:
    ticker: str
    reason: str
    exit_price: float
    timestamp: Optional[datetime] = None


class ORBStrategy:
    def __init__(
        self,
        ticker: str,
        rvol_threshold: float = 1.5,
        max_gap_pct: float = 0.04,
    ):
        self.ticker = ticker
        self.rvol_threshold = rvol_threshold
        self.max_gap_pct = max_gap_pct

        self._opening_range: Optional[dict] = None
        self._gap: Optional[dict] = None
        self._session_active: bool = False
        self._entry_taken: bool = False
        self._skip_reason: Optional[str] = None

    def reset(self) -> None:
        self._opening_range = None
        self._gap = None
        self._session_active = False
        self._entry_taken = False
        self._skip_reason = None

    def set_session(self, bars: pd.DataFrame, prev_close: float) -> dict:
        self.reset()

        self._opening_range = identify_opening_range(bars)

        if self._opening_range is None:
            self._skip_reason = "No opening range bar found in data"
            return self._session_status()

        current_open = float(bars.iloc[0]["open"])
        self._gap = compute_gap(current_open, prev_close)

        if self._gap["is_large_gap"]:
            self._skip_reason = (
                f"Gap too large ({self._gap['gap_pct']:.2%}) — "
                f"exceeds {self.max_gap_pct:.0%} threshold"
            )
            return self._session_status()

        if not self._opening_range["is_valid"]:
            self._skip_reason = self._opening_range["rejection_reason"]
            return self._session_status()

        self._session_active = True
        return self._session_status()

    def scan_entry(self, bars: pd.DataFrame) -> Optional[ORBSignal]:
        if not self._session_active:
            return None

        if self._entry_taken:
            return None

        if self._opening_range is None:
            return None

        breakout = detect_orb_breakout(
            bars,
            self._opening_range,
            rvol_threshold=self.rvol_threshold,
        )

        if breakout is None:
            return None

        self._entry_taken = True

        gap_pct = self._gap["gap_pct"] if self._gap else 0.0

        return ORBSignal(
            ticker=self.ticker,
            direction=breakout["direction"],
            entry_price=breakout["entry_price"],
            stop_loss=breakout["stop_loss"],
            take_profit=breakout["take_profit"],
            range_high=self._opening_range["range_high"],
            range_low=self._opening_range["range_low"],
            range_width=self._opening_range["range_width"],
            rvol=breakout["rvol"],
            gap_pct=gap_pct,
            confidence_factors=breakout["confidence_factors"],
            timestamp=datetime.now(tz=zoneinfo.ZoneInfo("UTC")),
        )

    def check_exit(
        self,
        bars: pd.DataFrame,
        current_price: float,
        position_direction: str,
        current_time_et: Optional[datetime] = None,
    ) -> Optional[ExitSignal]:
        if current_time_et is None:
            current_time_et = datetime.now(ET)

        now_t = current_time_et.time() if hasattr(current_time_et, 'time') else current_time_et

        if now_t >= ORB_EXIT_DEADLINE:
            return ExitSignal(
                ticker=self.ticker,
                reason="time_stop",
                exit_price=current_price,
                timestamp=datetime.now(tz=zoneinfo.ZoneInfo("UTC")),
            )

        if self._opening_range is None:
            return None

        range_high = self._opening_range["range_high"]
        range_low = self._opening_range["range_low"]
        range_width = self._opening_range["range_width"]
        range_midpoint = (range_high + range_low) / 2

        if position_direction == "long":
            stop = range_low if range_width / current_price <= 0.015 else range_midpoint
            target = range_high + 1.5 * range_width

            if current_price <= stop:
                return ExitSignal(
                    ticker=self.ticker,
                    reason="stop_loss",
                    exit_price=current_price,
                    timestamp=datetime.now(tz=zoneinfo.ZoneInfo("UTC")),
                )

            if current_price >= target:
                return ExitSignal(
                    ticker=self.ticker,
                    reason="take_profit",
                    exit_price=current_price,
                    timestamp=datetime.now(tz=zoneinfo.ZoneInfo("UTC")),
                )

        else:
            stop = range_high if range_width / current_price <= 0.015 else range_midpoint
            target = range_low - 1.5 * range_width

            if current_price >= stop:
                return ExitSignal(
                    ticker=self.ticker,
                    reason="stop_loss",
                    exit_price=current_price,
                    timestamp=datetime.now(tz=zoneinfo.ZoneInfo("UTC")),
                )

            if current_price <= target:
                return ExitSignal(
                    ticker=self.ticker,
                    reason="take_profit",
                    exit_price=current_price,
                    timestamp=datetime.now(tz=zoneinfo.ZoneInfo("UTC")),
                )

        vwap_exit = self._check_vwap_trail(bars, current_price, position_direction)
        if vwap_exit:
            return vwap_exit

        return None

    def _check_vwap_trail(
        self,
        bars: pd.DataFrame,
        current_price: float,
        position_direction: str,
    ) -> Optional[ExitSignal]:
        if bars is None or len(bars) < 3:
            return None

        try:
            vwap_df = compute_vwap(bars)
        except (ValueError, Exception):
            return None

        if vwap_df is None or vwap_df.empty:
            return None

        last_row = vwap_df.iloc[-1]
        vwap = last_row.get("vwap")

        if vwap is None or pd.isna(vwap):
            return None

        if self._opening_range is None:
            return None

        if position_direction == "long":
            if current_price <= self._opening_range["range_high"]:
                return None

            if current_price < vwap:
                return ExitSignal(
                    ticker=self.ticker,
                    reason="vwap_trail",
                    exit_price=current_price,
                    timestamp=datetime.now(tz=zoneinfo.ZoneInfo("UTC")),
                )

        else:
            if current_price >= self._opening_range["range_low"]:
                return None

            if current_price > vwap:
                return ExitSignal(
                    ticker=self.ticker,
                    reason="vwap_trail",
                    exit_price=current_price,
                    timestamp=datetime.now(tz=zoneinfo.ZoneInfo("UTC")),
                )

        return None

    def _session_status(self) -> dict:
        return {
            "active": self._session_active,
            "opening_range": self._opening_range,
            "gap": self._gap,
            "skip_reason": self._skip_reason,
        }

    @property
    def session_active(self) -> bool:
        return self._session_active

    @property
    def entry_taken(self) -> bool:
        return self._entry_taken

    @property
    def opening_range(self) -> Optional[dict]:
        return self._opening_range

    @property
    def skip_reason(self) -> Optional[str]:
        return self._skip_reason

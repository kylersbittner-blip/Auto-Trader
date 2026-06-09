"""
VWAP Mean Reversion Strategy.

Implements the full strategy from the Auto-Trader Blueprint:
  - Afternoon session only (13:30-15:30 ET)
  - Entry: price touches/exceeds 2 SD VWAP band
  - Trend filter: VWAP slope < 0.1%/hr (no mean reversion on trend days)
  - Volume filter: signal bar volume >= 1x 20-bar average
  - Scaled exit: 50% at 1 SD return, remaining 50% at VWAP return
  - Stop loss: price exceeds 3 SD band (thesis invalidated)
  - Time stop: flat by 15:50 ET
  - One entry per ticker per session

Based on:
  Zarattini & Aziz (2023), "VWAP: The Holy Grail for Day Trading Systems"
  QuantConnect 2022 backtest: 63% win rate at upper 2SD, 61% at lower 2SD
"""
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional
import zoneinfo

import numpy as np
import pandas as pd

from engine.vwap import compute_vwap, vwap_band_position

ET = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")

VWAP_ENTRY_OPEN = time(13, 30)
VWAP_ENTRY_CLOSE = time(15, 30)
VWAP_EXIT_DEADLINE = time(15, 50)

MAX_VWAP_SLOPE_PCT_PER_HR = 0.001
SLOPE_LOOKBACK_BARS = 4


@dataclass
class VWAPReversionSignal:
    ticker: str
    direction: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    vwap: float
    sd_distance: float
    volume_character: str
    vwap_slope_pct_hr: float
    confidence_factors: list[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    @property
    def action(self) -> str:
        return "buy" if self.direction == "long" else "sell"

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_t1_per_share(self) -> float:
        return abs(self.target_1 - self.entry_price)

    @property
    def reward_t2_per_share(self) -> float:
        return abs(self.target_2 - self.entry_price)


@dataclass
class VWAPExitSignal:
    ticker: str
    reason: str
    exit_price: float
    close_pct: float
    timestamp: Optional[datetime] = None


class VWAPReversionStrategy:
    def __init__(
        self,
        ticker: str,
        min_volume_ratio: float = 1.0,
        orb_trending: bool = False,
    ):
        self.ticker = ticker
        self.min_volume_ratio = min_volume_ratio
        self.orb_trending = orb_trending

        self._session_active: bool = False
        self._entry_taken: bool = False
        self._target_1_hit: bool = False
        self._skip_reason: Optional[str] = None
        self._entry_signal: Optional[VWAPReversionSignal] = None

    def reset(self) -> None:
        self._session_active = False
        self._entry_taken = False
        self._target_1_hit = False
        self._skip_reason = None
        self._entry_signal = None

    def set_session(
        self,
        bars: pd.DataFrame,
        orb_trending: Optional[bool] = None,
    ) -> dict:
        self.reset()

        if orb_trending is not None:
            self.orb_trending = orb_trending

        if bars is None or len(bars) < 3:
            self._skip_reason = "Insufficient bar data for VWAP calculation"
            return self._session_status()

        if self.orb_trending:
            self._skip_reason = "ORB breakout still trending — reversion suppressed"
            return self._session_status()

        try:
            vwap_df = compute_vwap(bars)
        except (ValueError, Exception) as e:
            self._skip_reason = f"VWAP computation failed: {e}"
            return self._session_status()

        last = vwap_df.iloc[-1]
        if pd.isna(last.get("vwap")) or last.get("vwap", 0) <= 0:
            self._skip_reason = "VWAP is invalid or zero"
            return self._session_status()

        band_width = last.get("vwap_upper_1sd", 0) - last.get("vwap", 0)
        if band_width <= 0:
            self._skip_reason = "VWAP bands have no width — too early in session"
            return self._session_status()

        self._session_active = True
        return self._session_status()

    def scan_entry(self, bars: pd.DataFrame) -> Optional[VWAPReversionSignal]:
        if not self._session_active or self._entry_taken:
            return None

        if bars is None or len(bars) < SLOPE_LOOKBACK_BARS + 1:
            return None

        try:
            vwap_df = compute_vwap(bars)
        except (ValueError, Exception):
            return None

        if vwap_df is None or vwap_df.empty:
            return None

        last = vwap_df.iloc[-1]

        idx_et = _to_et(vwap_df.index[-1])
        if idx_et is None:
            return None
        bar_time = idx_et.time()
        if bar_time < VWAP_ENTRY_OPEN or bar_time >= VWAP_ENTRY_CLOSE:
            return None

        position = vwap_band_position(last)
        if position["reversion_signal"] == "none":
            return None

        direction = position["reversion_signal"]
        sd_distance = position["sd_distance"]

        close = float(last["close"])
        vwap = float(last["vwap"])
        upper_1 = float(last["vwap_upper_1sd"])
        lower_1 = float(last["vwap_lower_1sd"])
        upper_3 = float(last["vwap_upper_3sd"])
        lower_3 = float(last["vwap_lower_3sd"])

        slope = _compute_vwap_slope(vwap_df, SLOPE_LOOKBACK_BARS)
        if slope is None:
            return None

        if abs(slope) > MAX_VWAP_SLOPE_PCT_PER_HR:
            return None

        volume = float(last["volume"])
        avg_vol = float(bars["volume"].rolling(20, min_periods=3).mean().iloc[-1])
        if avg_vol <= 0:
            avg_vol = float(bars["volume"].mean())

        vol_ratio = volume / avg_vol if avg_vol > 0 else 0.0
        if vol_ratio < self.min_volume_ratio:
            return None

        if len(bars) >= 2:
            prev_vol = float(bars["volume"].iloc[-2])
            vol_character = "exhaustion" if volume < prev_vol else "continuation"
        else:
            vol_character = "unknown"

        confidence = []

        if direction == "long":
            entry_price = close
            stop_loss = lower_3
            target_1 = lower_1
            target_2 = vwap
            confidence.append(f"Close ${close:.2f} at/below lower 2SD")
        else:
            entry_price = close
            stop_loss = upper_3
            target_1 = upper_1
            target_2 = vwap
            confidence.append(f"Close ${close:.2f} at/above upper 2SD")

        confidence.append(f"VWAP slope {slope:.5f}%/hr (threshold {MAX_VWAP_SLOPE_PCT_PER_HR:.4f})")
        confidence.append(f"Volume ratio {vol_ratio:.1f}x")
        if vol_character == "exhaustion":
            confidence.append("Volume declining (exhaustion pattern)")

        self._entry_taken = True

        signal = VWAPReversionSignal(
            ticker=self.ticker,
            direction=direction,
            entry_price=entry_price,
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            vwap=round(vwap, 2),
            sd_distance=sd_distance,
            volume_character=vol_character,
            vwap_slope_pct_hr=round(slope, 6),
            confidence_factors=confidence,
            timestamp=datetime.now(tz=UTC),
        )
        self._entry_signal = signal
        return signal

    def check_exit(
        self,
        bars: pd.DataFrame,
        current_price: float,
        position_direction: str,
        current_time_et: Optional[datetime] = None,
    ) -> Optional[VWAPExitSignal]:
        if current_time_et is None:
            current_time_et = datetime.now(ET)

        now_t = current_time_et.time() if hasattr(current_time_et, 'time') else current_time_et

        if now_t >= VWAP_EXIT_DEADLINE:
            return VWAPExitSignal(
                ticker=self.ticker,
                reason="time_stop",
                exit_price=current_price,
                close_pct=1.0,
                timestamp=datetime.now(tz=UTC),
            )

        if bars is None or len(bars) < 3:
            return None

        try:
            vwap_df = compute_vwap(bars)
        except (ValueError, Exception):
            return None

        if vwap_df is None or vwap_df.empty:
            return None

        last = vwap_df.iloc[-1]
        vwap = float(last.get("vwap", 0))

        if vwap <= 0:
            return None

        upper_1 = float(last.get("vwap_upper_1sd", vwap))
        lower_1 = float(last.get("vwap_lower_1sd", vwap))
        upper_3 = float(last.get("vwap_upper_3sd", vwap))
        lower_3 = float(last.get("vwap_lower_3sd", vwap))

        if position_direction == "long" and current_price <= lower_3:
            return VWAPExitSignal(
                ticker=self.ticker,
                reason="stop_loss",
                exit_price=current_price,
                close_pct=1.0,
                timestamp=datetime.now(tz=UTC),
            )
        if position_direction == "short" and current_price >= upper_3:
            return VWAPExitSignal(
                ticker=self.ticker,
                reason="stop_loss",
                exit_price=current_price,
                close_pct=1.0,
                timestamp=datetime.now(tz=UTC),
            )

        if not self._target_1_hit:
            t1_hit = False
            if position_direction == "long" and current_price >= lower_1:
                t1_hit = True
            elif position_direction == "short" and current_price <= upper_1:
                t1_hit = True

            if t1_hit:
                self._target_1_hit = True
                return VWAPExitSignal(
                    ticker=self.ticker,
                    reason="target_1",
                    exit_price=current_price,
                    close_pct=0.5,
                    timestamp=datetime.now(tz=UTC),
                )

        if self._target_1_hit:
            t2_hit = False
            if position_direction == "long" and current_price >= vwap:
                t2_hit = True
            elif position_direction == "short" and current_price <= vwap:
                t2_hit = True

            if t2_hit:
                return VWAPExitSignal(
                    ticker=self.ticker,
                    reason="target_2",
                    exit_price=current_price,
                    close_pct=1.0,
                    timestamp=datetime.now(tz=UTC),
                )

        return None

    @property
    def session_active(self) -> bool:
        return self._session_active

    @property
    def entry_taken(self) -> bool:
        return self._entry_taken

    @property
    def target_1_hit(self) -> bool:
        return self._target_1_hit

    @property
    def skip_reason(self) -> Optional[str]:
        return self._skip_reason

    def _session_status(self) -> dict:
        return {
            "active": self._session_active,
            "skip_reason": self._skip_reason,
        }


def _compute_vwap_slope(
    vwap_df: pd.DataFrame,
    lookback_bars: int,
) -> Optional[float]:
    if vwap_df is None or len(vwap_df) < lookback_bars + 1:
        return None

    recent = vwap_df.iloc[-(lookback_bars + 1):]
    vwap_start = float(recent.iloc[0].get("vwap", 0))
    vwap_end = float(recent.iloc[-1].get("vwap", 0))

    if vwap_start <= 0:
        return None

    total_pct_change = (vwap_end - vwap_start) / vwap_start
    hours = lookback_bars * 0.5
    slope_per_hr = total_pct_change / hours if hours > 0 else 0.0

    return slope_per_hr


def _to_et(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if hasattr(ts, 'tz') and ts.tz is not None:
        return ts.tz_convert(ET).to_pydatetime()
    try:
        return pd.Timestamp(ts).tz_localize("UTC").tz_convert(ET).to_pydatetime()
    except Exception:
        return None

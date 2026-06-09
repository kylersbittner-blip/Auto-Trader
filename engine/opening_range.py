"""
Opening Range Breakout (ORB) tracker.

Captures the high/low of the first 30-minute bar (9:30-10:00 ET)
and detects breakouts on subsequent bars with volume confirmation.

Based on:
  Zarattini, Barbon & Aziz (2024), "A Profitable Day Trading Strategy
  For The U.S. Equity Market," Swiss Finance Institute.
"""
import zoneinfo
from datetime import time, datetime
from typing import Optional

import numpy as np
import pandas as pd

ET = zoneinfo.ZoneInfo("America/New_York")

OR_START = time(9, 30)
OR_END = time(10, 0)
ORB_CUTOFF = time(12, 0)

MIN_RANGE_PCT = 0.003
MAX_RANGE_PCT = 0.020


def identify_opening_range(df: pd.DataFrame) -> Optional[dict]:
    if df is None or df.empty:
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        return None

    idx_et = df.index.tz_convert(ET) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(ET)

    mask = idx_et.time == OR_START
    if not mask.any():
        return None

    or_bar = df.loc[mask].iloc[-1]
    or_time = idx_et[mask][-1]

    range_high = float(or_bar["high"])
    range_low = float(or_bar["low"])
    range_close = float(or_bar["close"])
    range_width = range_high - range_low
    midpoint = (range_high + range_low) / 2
    range_width_pct = range_width / midpoint if midpoint > 0 else 0.0
    range_volume = float(or_bar["volume"])

    is_valid = True
    rejection_reason = None

    if range_width_pct < MIN_RANGE_PCT:
        is_valid = False
        rejection_reason = f"Range too narrow ({range_width_pct:.4f} < {MIN_RANGE_PCT})"
    elif range_width_pct > MAX_RANGE_PCT:
        is_valid = False
        rejection_reason = f"Range too wide ({range_width_pct:.4f} > {MAX_RANGE_PCT})"
    elif range_volume < 1:
        is_valid = False
        rejection_reason = "No volume in opening bar"

    return {
        "range_high": range_high,
        "range_low": range_low,
        "range_close": range_close,
        "range_width": round(range_width, 4),
        "range_width_pct": round(range_width_pct, 6),
        "range_volume": range_volume,
        "range_bar_time": or_time,
        "is_valid": is_valid,
        "rejection_reason": rejection_reason,
    }


def detect_orb_breakout(
    df: pd.DataFrame,
    opening_range: dict,
    rvol_threshold: float = 1.5,
    vol_lookback_days: int = 20,
) -> Optional[dict]:
    if opening_range is None or not opening_range["is_valid"]:
        return None

    if df is None or df.empty:
        return None

    range_high = opening_range["range_high"]
    range_low = opening_range["range_low"]
    range_width = opening_range["range_width"]
    range_midpoint = (range_high + range_low) / 2

    idx_et = df.index.tz_convert(ET) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(ET)

    or_date = opening_range["range_bar_time"].date()

    post_or_mask = (
        (idx_et.date == or_date)
        & (idx_et.time > OR_START)
        & (idx_et.time < ORB_CUTOFF)
    )

    post_or_bars = df.loc[post_or_mask]
    if post_or_bars.empty:
        return None

    avg_vol = df["volume"].rolling(vol_lookback_days * 13, min_periods=5).mean()

    for idx_pos, (bar_idx, bar) in enumerate(post_or_bars.iterrows()):
        close = float(bar["close"])
        volume = float(bar["volume"])
        bar_time_et = idx_et[df.index == bar_idx][0]

        avg_v = avg_vol.get(bar_idx)
        if avg_v is None or pd.isna(avg_v) or avg_v <= 0:
            avg_v = df["volume"].mean()
        rvol = volume / avg_v if avg_v > 0 else 0.0

        direction = None
        confidence_factors = []

        if close > range_high:
            direction = "long"
            confidence_factors.append(
                f"Close ${close:.2f} > range high ${range_high:.2f}"
            )
        elif close < range_low:
            direction = "short"
            confidence_factors.append(
                f"Close ${close:.2f} < range low ${range_low:.2f}"
            )

        if direction is None:
            continue

        if rvol < rvol_threshold:
            continue
        confidence_factors.append(f"RVOL {rvol:.1f}x (threshold {rvol_threshold}x)")

        if direction == "long":
            stop_loss = range_low if range_width_pct_ok(range_width, close) else range_midpoint
            take_profit = close + 1.5 * range_width
        else:
            stop_loss = range_high if range_width_pct_ok(range_width, close) else range_midpoint
            take_profit = close - 1.5 * range_width

        return {
            "direction": direction,
            "breakout_bar_time": bar_time_et,
            "breakout_price": close,
            "entry_price": close,
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "range_midpoint": round(range_midpoint, 2),
            "rvol": round(rvol, 2),
            "confidence_factors": confidence_factors,
        }

    return None


def range_width_pct_ok(width: float, price: float) -> bool:
    if price <= 0:
        return False
    pct = width / price
    return pct <= 0.015


def compute_gap(
    current_open: float,
    prev_close: float,
) -> dict:
    if prev_close <= 0:
        return {"gap_pct": 0.0, "gap_abs": 0.0, "is_large_gap": False}

    gap = current_open - prev_close
    gap_pct = gap / prev_close

    return {
        "gap_pct": round(gap_pct, 6),
        "gap_abs": round(abs(gap), 4),
        "is_large_gap": abs(gap_pct) > 0.04,
    }

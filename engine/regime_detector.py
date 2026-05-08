"""
Market regime detector.

Classifies current market conditions into one of three regimes:
  - trending       → strong directional move, use momentum strategy
  - ranging        → sideways / mean-reverting, use mean-reversion strategy
  - breakout_setup → compressed volatility with volume building, use breakout strategy

Algorithm:
  1. Trend strength  — R² of linear fit to recent closes
  2. Volatility compression — ratio of short-window ATR to long-window ATR
  3. Volume context  — recent bars vs rolling average
"""
import numpy as np
import pandas as pd


def detect_regime(df: pd.DataFrame, lookback: int = 20) -> str:
    if df is None or len(df) < 60:
        return "trending"

    recent = df.tail(lookback)
    longer = df.tail(60)

    # ── Trend strength via R² ─────────────────────────────────────────────────
    closes = recent["close"].values
    x = np.arange(len(closes))
    coeffs = np.polyfit(x, closes, 1)
    predicted = np.polyval(coeffs, x)
    ss_res = np.sum((closes - predicted) ** 2)
    ss_tot = np.sum((closes - closes.mean()) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    slope = coeffs[0]
    price_scale = closes.mean()
    normalized_slope = abs(slope * lookback) / price_scale if price_scale > 0 else 0.0

    # ── Volatility compression ────────────────────────────────────────────────
    def mean_atr(d: pd.DataFrame) -> float:
        hl = d["high"] - d["low"]
        hc = (d["high"] - d["close"].shift(1)).abs()
        lc = (d["low"]  - d["close"].shift(1)).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1).mean()

    recent_atr = mean_atr(recent)
    long_atr   = mean_atr(longer)
    vol_ratio  = recent_atr / long_atr if long_atr > 1e-10 else 1.0

    # ── Volume spike ──────────────────────────────────────────────────────────
    recent_vol = recent["volume"].iloc[-5:].mean()
    avg_vol    = longer["volume"].mean()
    vol_spike  = recent_vol / avg_vol if avg_vol > 1e-10 else 1.0

    # ── Decision ──────────────────────────────────────────────────────────────
    if r_squared > 0.65 and normalized_slope > 0.004:
        return "trending"
    if vol_ratio < 0.75 and vol_spike > 1.3:
        return "breakout_setup"
    return "ranging"

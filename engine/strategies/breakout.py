"""
Breakout strategy — best when volatility has compressed and price breaks range.

Buy  when close > N-bar high with volume surge.
Sell when close < N-bar low  with volume surge.
Score scales with breakout magnitude and volume confirmation.
"""
import numpy as np
import pandas as pd


def detect_breakout(df: pd.DataFrame, lookback: int = 20) -> dict:
    if df is None or len(df) < lookback + 5:
        return _empty("insufficient data")

    df = df.copy()

    # Use all bars except the current one to define the range
    range_window = df.iloc[-(lookback + 1):-1]
    current      = df.iloc[-1]

    range_high = float(range_window["high"].max())
    range_low  = float(range_window["low"].min())
    range_size = range_high - range_low

    avg_vol     = df["volume"].tail(lookback).mean()
    current_vol = float(current["volume"])
    vol_ratio   = current_vol / avg_vol if avg_vol > 1e-10 else 1.0

    close = float(current["close"])

    patterns   = []
    buy_score  = 0.0
    sell_score = 0.0

    if close > range_high:
        pct = (close - range_high) / range_high * 100
        patterns.append(f"Broke {lookback}-bar high (+{pct:.2f}%)")
        buy_score += min(40 + pct * 8, 65)
        if vol_ratio > 1.5:
            patterns.append(f"Volume surge {vol_ratio:.1f}x")
            buy_score += 20
        elif vol_ratio > 1.2:
            patterns.append(f"Volume uptick {vol_ratio:.1f}x")
            buy_score += 10

    elif close < range_low:
        pct = (range_low - close) / range_low * 100
        patterns.append(f"Broke {lookback}-bar low (-{pct:.2f}%)")
        sell_score += min(40 + pct * 8, 65)
        if vol_ratio > 1.5:
            patterns.append(f"Volume surge {vol_ratio:.1f}x")
            sell_score += 20
        elif vol_ratio > 1.2:
            patterns.append(f"Volume uptick {vol_ratio:.1f}x")
            sell_score += 10

    else:
        pos = (close - range_low) / range_size if range_size > 1e-10 else 0.5
        if pos > 0.85:
            patterns.append("Approaching range high")
            buy_score += 14
        elif pos < 0.15:
            patterns.append("Approaching range low")
            sell_score += 14

    net = buy_score - sell_score
    if net > 20:
        action = "buy"
        score  = 50 + min(net / 2, 45)
    elif net < -20:
        action = "sell"
        score  = 50 + min(abs(net) / 2, 45)
    else:
        action = "hold"
        score  = 35 + abs(net) * 0.2

    reasoning = (
        f"Breakout: {', '.join(patterns) if patterns else 'Inside range'} — net {net:+.0f}"
    )
    return {
        "score":    round(score, 1),
        "action":   action,
        "patterns": patterns,
        "reasoning": reasoning,
        "rsi":      None,
    }


def _empty(reason: str) -> dict:
    return {"score": 40.0, "action": "hold", "patterns": [], "reasoning": reason, "rsi": None}

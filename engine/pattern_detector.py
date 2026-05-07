"""
Detects technical patterns and generates a technical score (0–100).
All indicators computed with pure pandas/numpy — no extra dependencies.

Patterns supported:
  - MACD crossover (bullish/bearish)
  - RSI overbought / oversold
  - Bollinger Band breakout
  - Bull flag (price consolidation after strong move)
  - EMA trend alignment
  - Volume surge
"""
import pandas as pd
import numpy as np
import structlog

log = structlog.get_logger()


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=length - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series):
    ema12 = _ema(series, 12)
    ema26 = _ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bbands(series: pd.Series, length: int = 20, std: float = 2.0):
    mid = series.rolling(length).mean()
    dev = series.rolling(length).std()
    return mid + std * dev, mid - std * dev, mid


# ── Main entry point ──────────────────────────────────────────────────────────

def detect_patterns(df: pd.DataFrame, strategy: str = "momentum") -> dict:
    """
    Run pattern detection on OHLCV DataFrame.

    Returns:
        {
            "score": float (0-100),
            "action": "buy" | "sell" | "hold",
            "patterns": list[str],
            "reasoning": str,
            "rsi": float | None,
        }
    """
    if df is None or len(df) < 30:
        return _empty_result("insufficient data")

    df = df.copy()

    # Compute indicators
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["close"])
    df["rsi"]      = _rsi(df["close"], 14)
    df["bb_upper"], df["bb_lower"], df["bb_mid"] = _bbands(df["close"], 20)
    df["ema_9"]    = _ema(df["close"], 9)
    df["ema_21"]   = _ema(df["close"], 21)
    df["ema_50"]   = _ema(df["close"], 50)
    df["vol_avg"]  = df["volume"].rolling(20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    patterns   = []
    buy_score  = 0.0
    sell_score = 0.0

    # MACD crossover
    if _ok(last, "macd") and _ok(last, "macd_signal"):
        if last["macd"] > last["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
            patterns.append("MACD bullish cross")
            buy_score += 25
        elif last["macd"] < last["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
            patterns.append("MACD bearish cross")
            sell_score += 25

    # RSI
    rsi = last.get("rsi")
    if rsi is not None and not np.isnan(rsi):
        if rsi < 30:
            patterns.append(f"RSI oversold ({rsi:.0f})")
            buy_score += 20
        elif rsi > 70:
            patterns.append(f"RSI overbought ({rsi:.0f})")
            sell_score += 20
        elif 45 < rsi < 60:
            buy_score += 8
    else:
        rsi = None

    # EMA alignment
    if _ok(last, "ema_9") and _ok(last, "ema_21") and _ok(last, "ema_50"):
        if last["ema_9"] > last["ema_21"] > last["ema_50"]:
            patterns.append("EMA bullish alignment")
            buy_score += 20
        elif last["ema_9"] < last["ema_21"] < last["ema_50"]:
            patterns.append("EMA bearish alignment")
            sell_score += 20

    # Bollinger Band breakout
    if _ok(last, "bb_upper") and _ok(last, "bb_lower"):
        if last["close"] > last["bb_upper"]:
            if strategy == "momentum":
                patterns.append("BB upper breakout")
                buy_score += 15
            else:
                patterns.append("BB overbought")
                sell_score += 10
        elif last["close"] < last["bb_lower"]:
            if strategy == "mean_reversion":
                patterns.append("BB lower bounce")
                buy_score += 15
            else:
                patterns.append("BB oversold")
                sell_score += 10

    # Volume surge
    if _ok(last, "vol_avg") and last["vol_avg"] > 0:
        vol_ratio = last["volume"] / last["vol_avg"]
        if vol_ratio > 2.0:
            patterns.append(f"Volume surge {vol_ratio:.1f}x avg")
            if buy_score > sell_score:
                buy_score *= 1.15
            else:
                sell_score *= 1.15

    # Bull flag
    if len(df) >= 10:
        prior_move   = (df["close"].iloc[-6] - df["close"].iloc[-10]) / df["close"].iloc[-10]
        recent_range = (df["high"].iloc[-5:].max() - df["low"].iloc[-5:].min()) / df["close"].iloc[-6]
        if prior_move > 0.03 and recent_range < 0.015:
            patterns.append("Bull flag consolidation")
            buy_score += 18

    # Final score and action
    net       = buy_score - sell_score
    raw_score = min(100, abs(net))

    if net > 10:
        action = "buy"
        score  = 50 + min(raw_score / 2, 45)
    elif net < -10:
        action = "sell"
        score  = 50 + min(raw_score / 2, 45)
    else:
        action = "hold"
        score  = 40 + raw_score * 0.1

    reasoning = f"{', '.join(patterns) if patterns else 'No strong pattern'} — net signal {net:+.0f}"

    return {
        "score":    round(score, 1),
        "action":   action,
        "patterns": patterns,
        "reasoning": reasoning,
        "rsi":      round(rsi, 1) if rsi is not None else None,
    }


def _ok(row: pd.Series, col: str) -> bool:
    return col in row and pd.notna(row[col])


def _empty_result(reason: str) -> dict:
    return {"score": 0.0, "action": "hold", "patterns": [], "reasoning": reason, "rsi": None}

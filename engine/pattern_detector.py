"""
Detects technical patterns and generates a technical score (0–100).

Patterns supported:
  - MACD crossover (bullish/bearish)
  - RSI overbought / oversold
  - Bollinger Band breakout
  - Bull flag (price consolidation after strong move)
  - EMA trend alignment
  - Volume surge
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import structlog

log = structlog.get_logger()


def detect_patterns(df: pd.DataFrame, strategy: str = "momentum") -> dict:
    """
    Run pattern detection on OHLCV DataFrame.

    Returns:
        {
            "score": float (0-100),
            "action": "buy" | "sell" | "hold",
            "patterns": list[str],
            "reasoning": str
        }
    """
    if df is None or len(df) < 30:
        return _empty_result("insufficient data")

    df = df.copy()

    # --- Indicators ---
    macd = ta.macd(df["close"])
    if macd is not None:
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]

    df["rsi"] = ta.rsi(df["close"], length=14)
    bbands = ta.bbands(df["close"], length=20)
    if bbands is not None:
        df["bb_upper"] = bbands["BBU_20_2.0"]
        df["bb_lower"] = bbands["BBL_20_2.0"]
        df["bb_mid"] = bbands["BBM_20_2.0"]

    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["vol_avg"] = df["volume"].rolling(20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    patterns = []
    buy_score = 0.0
    sell_score = 0.0

    # --- MACD Crossover ---
    if _safe(last, "macd") and _safe(last, "macd_signal"):
        if last["macd"] > last["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
            patterns.append("MACD bullish cross")
            buy_score += 25
        elif last["macd"] < last["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
            patterns.append("MACD bearish cross")
            sell_score += 25

    # --- RSI ---
    rsi = last.get("rsi")
    if rsi is not None:
        if rsi < 30:
            patterns.append(f"RSI oversold ({rsi:.0f})")
            buy_score += 20
        elif rsi > 70:
            patterns.append(f"RSI overbought ({rsi:.0f})")
            sell_score += 20
        elif 45 < rsi < 60:
            buy_score += 8   # healthy momentum zone

    # --- EMA Trend Alignment ---
    if _safe(last, "ema_9") and _safe(last, "ema_21") and _safe(last, "ema_50"):
        if last["ema_9"] > last["ema_21"] > last["ema_50"]:
            patterns.append("EMA bullish alignment")
            buy_score += 20
        elif last["ema_9"] < last["ema_21"] < last["ema_50"]:
            patterns.append("EMA bearish alignment")
            sell_score += 20

    # --- Bollinger Band Breakout ---
    if _safe(last, "bb_upper") and _safe(last, "bb_lower"):
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

    # --- Volume Surge ---
    if _safe(last, "vol_avg") and last["vol_avg"] > 0:
        vol_ratio = last["volume"] / last["vol_avg"]
        if vol_ratio > 2.0:
            patterns.append(f"Volume surge {vol_ratio:.1f}x avg")
            # Amplifies whichever direction is dominant
            buy_score *= 1.15 if buy_score > sell_score else 1.0
            sell_score *= 1.15 if sell_score > buy_score else 1.0

    # --- Bull Flag (5-bar consolidation after strong move) ---
    if len(df) >= 10:
        prior_move = (df["close"].iloc[-6] - df["close"].iloc[-10]) / df["close"].iloc[-10]
        recent_range = (df["high"].iloc[-5:].max() - df["low"].iloc[-5:].min()) / df["close"].iloc[-6]
        if prior_move > 0.03 and recent_range < 0.015:
            patterns.append("Bull flag consolidation")
            buy_score += 18

    # --- Compute final score and action ---
    net = buy_score - sell_score
    raw_score = min(100, abs(net))

    if net > 10:
        action = "buy"
        score = 50 + min(raw_score / 2, 45)
    elif net < -10:
        action = "sell"
        score = 50 + min(raw_score / 2, 45)
    else:
        action = "hold"
        score = 40 + raw_score * 0.1

    reasoning = f"{', '.join(patterns) if patterns else 'No strong pattern'} — net signal {net:+.0f}"

    return {
        "score": round(score, 1),
        "action": action,
        "patterns": patterns,
        "reasoning": reasoning,
        "rsi": round(rsi, 1) if rsi else None,
    }


def _safe(row: pd.Series, col: str) -> bool:
    return col in row and pd.notna(row[col])


def _empty_result(reason: str) -> dict:
    return {"score": 0.0, "action": "hold", "patterns": [], "reasoning": reason, "rsi": None}

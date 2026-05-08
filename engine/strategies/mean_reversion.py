"""
Mean-reversion strategy — best in ranging/sideways markets.

Buy when price is deeply oversold (RSI < 30, near lower Bollinger Band).
Sell when price is deeply overbought (RSI > 70, near upper Bollinger Band).
Score scales with severity of the extreme.
"""
import numpy as np
import pandas as pd


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=length - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def detect_mean_reversion(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 30:
        return _empty("insufficient data")

    df = df.copy()
    df["rsi"]      = _rsi(df["close"])
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_std"]   = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    last    = df.iloc[-1]
    rsi     = last.get("rsi")
    bb_pct  = last.get("bb_pct")
    rsi_val = float(rsi) if (rsi is not None and not np.isnan(rsi)) else None
    bb_val  = float(bb_pct) if (bb_pct is not None and not np.isnan(bb_pct)) else None

    patterns   = []
    buy_score  = 0.0
    sell_score = 0.0

    # RSI extremes
    if rsi_val is not None:
        if rsi_val < 30:
            severity = (30 - rsi_val) / 30
            patterns.append(f"RSI oversold ({rsi_val:.0f})")
            buy_score += 40 + severity * 30
        elif rsi_val < 40:
            patterns.append(f"RSI near oversold ({rsi_val:.0f})")
            buy_score += 18
        elif rsi_val > 70:
            severity = (rsi_val - 70) / 30
            patterns.append(f"RSI overbought ({rsi_val:.0f})")
            sell_score += 40 + severity * 30
        elif rsi_val > 60:
            patterns.append(f"RSI near overbought ({rsi_val:.0f})")
            sell_score += 18

    # Bollinger Band position
    if bb_val is not None:
        if bb_val < 0.10:
            patterns.append("Price at lower BB")
            buy_score += 25
        elif bb_val < 0.20:
            patterns.append("Price near lower BB")
            buy_score += 12
        elif bb_val > 0.90:
            patterns.append("Price at upper BB")
            sell_score += 25
        elif bb_val > 0.80:
            patterns.append("Price near upper BB")
            sell_score += 12

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
        f"Mean-reversion: {', '.join(patterns) if patterns else 'No extreme'} — net {net:+.0f}"
    )
    return {
        "score":    round(score, 1),
        "action":   action,
        "patterns": patterns,
        "reasoning": reasoning,
        "rsi":      round(rsi_val, 1) if rsi_val is not None else None,
    }


def _empty(reason: str) -> dict:
    return {"score": 40.0, "action": "hold", "patterns": [], "reasoning": reason, "rsi": None}

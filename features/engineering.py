"""
Feature engineering — 20 lag-safe technical features for XGBoost.

Rules enforced here:
  - Every feature at index t uses only data from t and earlier.
  - No future data leaks into the feature matrix.
  - Features are normalized where needed so XGBoost doesn't have to worry about scale.
"""
import numpy as np
import pandas as pd


FEATURE_COLS = [
    # Returns
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    # Volume
    "vol_zscore", "vol_ratio",
    # Momentum
    "rsi", "macd_hist_norm", "roc_10",
    # Volatility
    "atr_norm", "bb_width", "realized_vol_10", "hl_range",
    # Trend
    "ema9_21_ratio", "price_ema50_dev",
    # Time-of-day (cyclic encoding so the model understands market open/close)
    "hour_sin", "hour_cos", "time_to_close",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features on a copy of the OHLCV DataFrame.
    Input must have columns: open, high, low, close, volume.
    Returns the DataFrame with feature columns appended.
    """
    if df is None or len(df) < 30:
        return df

    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── Lagged returns ────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{lag}"] = close.pct_change(lag)

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std().replace(0, np.nan)
    df["vol_zscore"] = (volume - vol_mean_20) / vol_std_20
    df["vol_ratio"]  = volume / volume.rolling(5).mean().replace(0, np.nan)

    # ── RSI (normalized to 0–1) ───────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)) / 100

    # ── ATR (normalized) ──────────────────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    df["atr_norm"] = atr / close.replace(0, np.nan)

    # ── MACD histogram (normalized by ATR to be scale-invariant) ─────────────
    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    sig_line  = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - sig_line
    df["macd_hist_norm"] = macd_hist / atr.replace(0, np.nan)

    # ── Rate of change ────────────────────────────────────────────────────────
    df["roc_10"] = close.pct_change(10)

    # ── Bollinger Band width (measures volatility regime) ────────────────────
    bb_mid      = close.rolling(20).mean()
    bb_std      = close.rolling(20).std()
    df["bb_width"] = (2 * bb_std) / bb_mid.replace(0, np.nan)

    # ── Realized volatility (10-bar rolling std of returns) ──────────────────
    df["realized_vol_10"] = close.pct_change().rolling(10).std()

    # ── High-low range (intrabar volatility) ─────────────────────────────────
    df["hl_range"] = (high - low) / close.replace(0, np.nan)

    # ── EMA trend features ────────────────────────────────────────────────────
    ema_9  = close.ewm(span=9,  adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    df["ema9_21_ratio"]  = ema_9 / ema_21.replace(0, np.nan) - 1
    df["price_ema50_dev"] = close / ema_50.replace(0, np.nan) - 1

    # ── Time-of-day (cyclic) ──────────────────────────────────────────────────
    if hasattr(df.index, "hour"):
        hour   = df.index.hour
        minute = df.index.minute
        df["hour_sin"]      = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"]      = np.cos(2 * np.pi * hour / 24)
        # Fraction of trading day remaining (0 = close, 1 = open)
        minutes_elapsed     = (hour - 9) * 60 + minute - 30
        df["time_to_close"] = np.clip(1 - minutes_elapsed / 390, 0, 1)
    else:
        df["hour_sin"]      = 0.0
        df["hour_cos"]      = 1.0
        df["time_to_close"] = 0.5

    return df

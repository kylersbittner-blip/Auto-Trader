"""
Cumulative intraday VWAP with standard deviation bands.

VWAP = Σ(Typical Price × Volume) / Σ(Volume)
Typical Price = (High + Low + Close) / 3

SD bands use the running population variance of (Typical Price - VWAP),
weighted by volume. This matches how institutional platforms compute
VWAP bands — each bar's deviation is weighted by how much volume
traded at that price.

Resets at the start of each trading day.
"""
import numpy as np
import pandas as pd


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    required = {"high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must have columns: {required}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    df = df.copy()
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tp_x_vol"] = df["typical_price"] * df["volume"]

    if df.index.tz is not None:
        dates = df.index.normalize()
    else:
        dates = df.index.normalize()
    df["_trading_day"] = dates

    df["cum_tp_vol"] = df.groupby("_trading_day")["tp_x_vol"].cumsum()
    df["cum_vol"] = df.groupby("_trading_day")["volume"].cumsum()

    df["vwap"] = np.where(
        df["cum_vol"] > 0,
        df["cum_tp_vol"] / df["cum_vol"],
        df["typical_price"],
    )

    df["sq_dev_x_vol"] = df["volume"] * (df["typical_price"] - df["vwap"]) ** 2
    df["cum_sq_dev_vol"] = df.groupby("_trading_day")["sq_dev_x_vol"].cumsum()

    variance = np.where(
        df["cum_vol"] > 0,
        df["cum_sq_dev_vol"] / df["cum_vol"],
        0.0,
    )
    sd = np.sqrt(np.maximum(variance, 0.0))

    df["vwap_upper_1sd"] = df["vwap"] + sd
    df["vwap_lower_1sd"] = df["vwap"] - sd
    df["vwap_upper_2sd"] = df["vwap"] + 2 * sd
    df["vwap_lower_2sd"] = df["vwap"] - 2 * sd
    df["vwap_upper_3sd"] = df["vwap"] + 3 * sd
    df["vwap_lower_3sd"] = df["vwap"] - 3 * sd

    df["vwap_deviation"] = np.where(
        df["vwap"] > 0,
        (df["close"] - df["vwap"]) / df["vwap"] * 100,
        0.0,
    )

    df.drop(
        columns=[
            "typical_price", "tp_x_vol", "_trading_day",
            "cum_tp_vol", "cum_vol", "sq_dev_x_vol", "cum_sq_dev_vol",
        ],
        inplace=True,
    )

    return df


def vwap_band_position(row: pd.Series) -> dict:
    close = row.get("close")
    vwap = row.get("vwap")

    if close is None or vwap is None or pd.isna(close) or pd.isna(vwap):
        return {"zone": "unknown", "sd_distance": 0.0, "reversion_signal": "none"}

    upper_1 = row.get("vwap_upper_1sd", vwap)
    upper_2 = row.get("vwap_upper_2sd", vwap)
    upper_3 = row.get("vwap_upper_3sd", vwap)
    lower_1 = row.get("vwap_lower_1sd", vwap)
    lower_2 = row.get("vwap_lower_2sd", vwap)
    lower_3 = row.get("vwap_lower_3sd", vwap)

    sd_width = upper_1 - vwap if upper_1 != vwap else 1e-10
    sd_distance = (close - vwap) / sd_width if sd_width > 1e-10 else 0.0

    if close >= upper_3:
        zone = "above_3sd"
        signal = "short"
    elif close >= upper_2:
        zone = "above_2sd"
        signal = "short"
    elif close >= upper_1:
        zone = "above_1sd"
        signal = "none"
    elif close <= lower_3:
        zone = "below_3sd"
        signal = "long"
    elif close <= lower_2:
        zone = "below_2sd"
        signal = "long"
    elif close <= lower_1:
        zone = "below_1sd"
        signal = "none"
    else:
        zone = "near_vwap"
        signal = "none"

    return {
        "zone": zone,
        "sd_distance": round(sd_distance, 2),
        "reversion_signal": signal,
    }

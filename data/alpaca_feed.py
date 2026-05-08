"""
Alpaca market data — real-time bars and historical data.
Replaces Polygon for OHLCV so we get real-time prices on the free IEX feed.
Polygon is kept for news only.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
import structlog

from config import get_settings

log = structlog.get_logger()
settings = get_settings()


def _get_client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )


def _fetch_bars_sync(
    ticker: str,
    days: int = 3,
    timeframe_minutes: int = 5,
    feed: str = "iex",
) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(timeframe_minutes, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=feed,
    )
    bars = client.get_stock_bars(req)
    df = bars.df

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _fetch_historical_sync(
    ticker: str,
    days: int = 365,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    """Pull up to `days` of history for ML training."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(timeframe_minutes, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed="iex",
        adjustment="all",      # split + dividend adjusted
    )
    bars = client.get_stock_bars(req)
    df = bars.df

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)

    # Keep only regular market hours (9:30–16:00 ET)
    df = df.between_time("13:30", "20:00")   # UTC equivalent of 9:30–16:00 ET
    return df.sort_index()


async def fetch_bars(ticker: str, days: int = 3, timeframe_minutes: int = 5) -> pd.DataFrame:
    """Async wrapper — returns recent bars for live signal generation."""
    try:
        return await asyncio.to_thread(_fetch_bars_sync, ticker, days, timeframe_minutes)
    except Exception as e:
        log.warning("alpaca_bars_failed", ticker=ticker, error=str(e))
        return pd.DataFrame()


async def fetch_bars_batch(tickers: list[str], days: int = 3, timeframe_minutes: int = 5) -> dict[str, pd.DataFrame]:
    """Fetch bars for multiple tickers concurrently."""
    results = await asyncio.gather(
        *[fetch_bars(t, days, timeframe_minutes) for t in tickers],
        return_exceptions=True,
    )
    return {
        t: r if isinstance(r, pd.DataFrame) else pd.DataFrame()
        for t, r in zip(tickers, results)
    }


async def fetch_historical(ticker: str, days: int = 365, timeframe_minutes: int = 5) -> pd.DataFrame:
    """Async wrapper — returns long history for ML training."""
    try:
        return await asyncio.to_thread(_fetch_historical_sync, ticker, days, timeframe_minutes)
    except Exception as e:
        log.warning("alpaca_historical_failed", ticker=ticker, error=str(e))
        return pd.DataFrame()

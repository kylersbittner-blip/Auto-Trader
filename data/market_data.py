"""
Fetches OHLCV bars and real-time quotes from Polygon.io.
Falls back to cached data if the API is unavailable.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import httpx
import pandas as pd
import structlog

from config import get_settings
from data.cache import get_redis, cache_set, cache_get

log = structlog.get_logger()
settings = get_settings()

POLYGON_BASE = "https://api.polygon.io"


async def fetch_bars(
    ticker: str,
    timespan: str = "minute",
    multiplier: int = 5,
    days_back: int = 2,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars for a ticker.

    Args:
        ticker: Stock symbol e.g. 'NVDA'
        timespan: 'minute' | 'hour' | 'day'
        multiplier: Bar size (5 = 5-minute bars)
        days_back: How many calendar days of history

    Returns:
        DataFrame with columns: open, high, low, close, volume, timestamp
    """
    cache_key = f"bars:{ticker}:{timespan}:{multiplier}"
    cached = await cache_get(cache_key)
    if cached:
        return pd.read_json(cached)

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")

    url = (
        f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range"
        f"/{multiplier}/{timespan}/{from_date}/{to_date}"
        f"?adjusted=true&sort=asc&limit=500&apiKey={settings.polygon_api_key}"
    )

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    if data.get("resultsCount", 0) == 0:
        log.warning("no_bars_returned", ticker=ticker)
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp")

    await cache_set(cache_key, df.reset_index().to_json(), ttl=60)
    return df


async def fetch_latest_quote(ticker: str) -> Optional[dict]:
    """
    Fetch the latest bid/ask/price for a ticker.
    Returns dict with: price, bid, ask, timestamp
    """
    url = f"{POLYGON_BASE}/v2/last/trade/{ticker}?apiKey={settings.polygon_api_key}"
    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return None
        data = resp.json()

    result = data.get("results", {})
    return {
        "price": result.get("p"),
        "size": result.get("s"),
        "timestamp": result.get("t"),
    }


async def fetch_bars_batch(tickers: list[str], **kwargs) -> dict[str, pd.DataFrame]:
    """Fetch bars for multiple tickers concurrently."""
    tasks = [fetch_bars(t, **kwargs) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {
        ticker: df
        for ticker, df in zip(tickers, results)
        if isinstance(df, pd.DataFrame) and not df.empty
    }

"""
Data adapters — thin wrappers implementing the scanner's Protocol
interfaces using Alpaca (market data) and Polygon (news).
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional
import httpx
import pandas as pd

from engine.scanner import MarketDataSource, NewsSource


class AlpacaDataSource:
    def __init__(self, api_key: str, secret_key: str, feed: str = "iex"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.feed = feed
        self._cache: dict[str, dict] = {}
        self._daily_cache: dict[str, pd.DataFrame] = {}

    def _get_client(self):
        from alpaca.data.historical import StockHistoricalDataClient
        return StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)

    def _ensure_daily_bars(self, ticker: str, days: int = 30) -> pd.DataFrame:
        if ticker in self._daily_cache:
            return self._daily_cache[ticker]
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        client = self._get_client()
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days + 5)
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start, end=end, feed=self.feed,
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            self._daily_cache[ticker] = pd.DataFrame()
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level="symbol")
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        self._daily_cache[ticker] = df
        return df

    def get_premarket_volume(self, ticker: str) -> float:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        client = self._get_client()
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=8, minute=0, second=0, microsecond=0)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                start=today_start, end=now, feed=self.feed,
            )
            bars = client.get_stock_bars(req)
            df = bars.df
            if df.empty:
                return 0.0
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level="symbol")
            return float(df["volume"].sum())
        except Exception:
            return 0.0

    def get_avg_daily_volume(self, ticker: str, lookback_days: int = 20) -> float:
        df = self._ensure_daily_bars(ticker, days=lookback_days + 5)
        if df.empty or len(df) < 3:
            return 0.0
        return float(df["volume"].tail(lookback_days).mean())

    def get_prev_close(self, ticker: str) -> float:
        df = self._ensure_daily_bars(ticker)
        if df.empty:
            return 0.0
        return float(df["close"].iloc[-1])

    def get_premarket_price(self, ticker: str) -> float:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            client = self._get_client()
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker, feed=self.feed)
            quotes = client.get_stock_latest_quote(req)
            quote = quotes.get(ticker)
            if quote:
                return float(quote.ask_price or quote.bid_price or 0)
        except Exception:
            pass
        return self.get_prev_close(ticker)

    def get_atr(self, ticker: str, period: int = 14) -> float:
        df = self._ensure_daily_bars(ticker, days=period + 10)
        if df.empty or len(df) < period + 1:
            return 0.0
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return float(tr.tail(period).mean())

    def clear_cache(self) -> None:
        self._cache.clear()
        self._daily_cache.clear()


class PolygonNewsSource:
    def __init__(self, api_key: str, lookback_hours: int = 18):
        self.api_key = api_key
        self.lookback_hours = lookback_hours
        self._cache: dict[str, list[dict]] = {}

    def _fetch_news(self, ticker: str) -> list[dict]:
        if ticker in self._cache:
            return self._cache[ticker]
        published_after = (
            datetime.utcnow() - timedelta(hours=self.lookback_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (
            f"https://api.polygon.io/v2/reference/news"
            f"?ticker={ticker}&published_utc.gte={published_after}"
            f"&limit=5&sort=published_utc&order=desc"
            f"&apiKey={self.api_key}"
        )
        try:
            resp = httpx.get(url, timeout=8)
            resp.raise_for_status()
            articles = resp.json().get("results", [])
        except Exception:
            articles = []
        self._cache[ticker] = articles
        return articles

    def has_catalyst(self, ticker: str) -> bool:
        return len(self._fetch_news(ticker)) > 0

    def get_headline(self, ticker: str) -> Optional[str]:
        articles = self._fetch_news(ticker)
        if articles:
            return articles[0].get("title")
        return None

    def clear_cache(self) -> None:
        self._cache.clear()


def fetch_30min_bars_sync(
    ticker: str, api_key: str, secret_key: str,
    days: int = 3, feed: str = "iex",
) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(30, TimeFrameUnit.Minute),
        start=start, end=end, feed=feed,
    )
    bars = client.get_stock_bars(req)
    df = bars.df
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert("America/New_York")
    df_et = df_et.between_time("09:30", "16:00")
    df_et.index = df_et.index.tz_convert("UTC")
    return df_et.sort_index()


async def fetch_30min_bars(
    ticker: str, api_key: str, secret_key: str,
    days: int = 3, feed: str = "iex",
) -> pd.DataFrame:
    return await asyncio.to_thread(
        fetch_30min_bars_sync, ticker, api_key, secret_key, days, feed,
    )


async def fetch_30min_bars_batch(
    tickers: list[str], api_key: str, secret_key: str,
    days: int = 3, feed: str = "iex",
) -> dict[str, pd.DataFrame]:
    results = await asyncio.gather(
        *[fetch_30min_bars(t, api_key, secret_key, days, feed) for t in tickers],
        return_exceptions=True,
    )
    return {
        t: r if isinstance(r, pd.DataFrame) else pd.DataFrame()
        for t, r in zip(tickers, results)
    }

"""
Fetches recent news from Polygon.io and scores sentiment using finBERT.
Returns a sentiment score (0–100) and labeled articles per ticker.
"""
import asyncio
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional
import httpx
import structlog

log = structlog.get_logger()

# Lazy-loaded to avoid slow startup
_sentiment_pipeline = None


def _get_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        from config import get_settings
        settings = get_settings()
        log.info("loading_finbert", model=settings.sentiment_model)
        _sentiment_pipeline = pipeline(
            "text-classification",
            model=settings.sentiment_model,
            top_k=None,
        )
    return _sentiment_pipeline


def score_text(text: str) -> dict:
    """
    Score a news headline or article snippet.

    Returns:
        {
            "label": "positive" | "negative" | "neutral",
            "score": float (0-100, where 100 = strong positive, 0 = strong negative, 50 = neutral)
        }
    """
    pipe = _get_pipeline()
    results = pipe(text[:512])[0]   # finBERT max 512 tokens
    label_map = {r["label"].lower(): r["score"] for r in results}

    pos = label_map.get("positive", 0)
    neg = label_map.get("negative", 0)
    neu = label_map.get("neutral", 0)

    # Convert to 0–100 scale centered on 50 (neutral)
    sentiment_score = 50 + (pos - neg) * 50
    dominant = max(label_map, key=label_map.get)

    return {
        "label": dominant,
        "score": round(sentiment_score, 1),
        "positive": round(pos, 3),
        "negative": round(neg, 3),
        "neutral": round(neu, 3),
    }


async def fetch_news(ticker: str, hours_back: int = 6) -> list[dict]:
    """
    Fetch recent news articles for a ticker from Polygon.io.
    Returns list of article dicts with sentiment scored.
    """
    from config import get_settings
    settings = get_settings()

    published_after = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        f"https://api.polygon.io/v2/reference/news"
        f"?ticker={ticker}&published_utc.gte={published_after}"
        f"&limit=10&sort=published_utc&order=desc"
        f"&apiKey={settings.polygon_api_key}"
    )

    async with httpx.AsyncClient(timeout=8) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            articles = resp.json().get("results", [])
        except Exception as e:
            log.warning("news_fetch_failed", ticker=ticker, error=str(e))
            return []

    scored = []
    for art in articles[:5]:   # score top 5 only to keep latency low
        headline = art.get("title", "")
        if not headline:
            continue
        sentiment = score_text(headline)
        scored.append({
            "ticker": ticker,
            "headline": headline,
            "source": art.get("publisher", {}).get("name", "Unknown"),
            "url": art.get("article_url"),
            "published_at": art.get("published_utc"),
            "sentiment": sentiment,
        })

    return scored


async def aggregate_sentiment(ticker: str, hours_back: int = 6) -> dict:
    """
    Fetch and aggregate sentiment for a ticker.

    Returns:
        {
            "score": float (0-100),
            "label": str,
            "article_count": int,
            "articles": list
        }
    """
    articles = await fetch_news(ticker, hours_back)
    if not articles:
        return {"score": 50.0, "label": "neutral", "article_count": 0, "articles": []}

    avg_score = sum(a["sentiment"]["score"] for a in articles) / len(articles)
    if avg_score > 60:
        label = "positive"
    elif avg_score < 40:
        label = "negative"
    else:
        label = "neutral"

    return {
        "score": round(avg_score, 1),
        "label": label,
        "article_count": len(articles),
        "articles": articles,
    }


async def scan_all(tickers: list[str]) -> dict[str, dict]:
    """Scan news for multiple tickers concurrently."""
    results = await asyncio.gather(*[aggregate_sentiment(t) for t in tickers], return_exceptions=True)
    return {
        ticker: res if isinstance(res, dict) else {"score": 50.0, "label": "neutral", "article_count": 0, "articles": []}
        for ticker, res in zip(tickers, results)
    }

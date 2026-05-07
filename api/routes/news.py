from fastapi import APIRouter, Query
from engine.news_scanner import fetch_news, scan_all
from config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("")
async def get_news(
    ticker: str | None = Query(None, description="Specific ticker, or all watchlist tickers"),
    hours_back: int = Query(6, ge=1, le=48),
):
    """Fetch and sentiment-score recent news."""
    if ticker:
        articles = await fetch_news(ticker.upper(), hours_back)
        return {"ticker": ticker.upper(), "articles": articles}

    all_news = await scan_all(settings.watchlist)
    return {"tickers": all_news}

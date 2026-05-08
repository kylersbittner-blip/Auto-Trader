"""
Training endpoints.
  POST /train          — fetch historical data, train XGBoost, run backtest, save model
  GET  /models         — list all saved models with metadata
  GET  /models/{ticker} — detailed stats + backtest for one ticker
"""
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional

from api.security import require_control_key
from config import get_settings
from activity import get_activity_logger

router = APIRouter()
settings = get_settings()

# Track in-progress training jobs so we don't run duplicates
_training_in_progress: set[str] = set()


class TrainRequest(BaseModel):
    tickers: Optional[list[str]] = None   # defaults to watchlist
    days:    int = 365                    # days of history to train on


async def _train_ticker(ticker: str, days: int) -> None:
    activity = get_activity_logger()
    _training_in_progress.add(ticker)
    try:
        from data.alpaca_feed import fetch_historical
        from models.trainer import walk_forward_train
        from models.backtester import run_backtest
        import models.registry as registry

        activity.info("scan", f"Fetching {days}d of history for {ticker}…", ticker=ticker)
        df = await fetch_historical(ticker, days=days)

        if df.empty or len(df) < 200:
            activity.failure("scan", f"Not enough historical data for {ticker} ({len(df)} bars)", ticker=ticker)
            return

        activity.info("scan", f"Training XGBoost on {len(df):,} bars for {ticker}…", ticker=ticker)

        train_result = await asyncio.to_thread(walk_forward_train, df)
        if "error" in train_result:
            activity.failure("scan", f"Training failed for {ticker}: {train_result['error']}", ticker=ticker)
            return

        bt_result = await asyncio.to_thread(run_backtest, df)

        registry.save(ticker, train_result, bt_result)

        summary = (
            f"{train_result['n_folds']} folds | "
            f"dir_acc={train_result['avg_dir_accuracy']:.1%} | "
            f"Sharpe={bt_result.get('sharpe_ratio','?')} | "
            f"WinRate={bt_result.get('win_rate_pct','?')}%"
        )
        activity.success("scan", f"Model trained for {ticker}", detail=summary, ticker=ticker)

    except Exception as e:
        get_activity_logger().failure("scan", f"Training error for {ticker}: {e}", ticker=ticker)
    finally:
        _training_in_progress.discard(ticker)


@router.post("", dependencies=[Depends(require_control_key)])
async def train_models(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Kick off model training in the background.
    Returns immediately — watch the Activity log for progress.
    """
    tickers = req.tickers or settings.watchlist
    queued = []
    skipped = []

    for ticker in tickers:
        if ticker in _training_in_progress:
            skipped.append(ticker)
        else:
            background_tasks.add_task(_train_ticker, ticker, req.days)
            queued.append(ticker)

    return {
        "status":  "training_started",
        "queued":  queued,
        "skipped": skipped,
        "message": "Training is running in the background. Watch the Activity log for progress.",
    }


@router.get("/models")
async def list_models():
    """List all saved models with their performance metadata."""
    import models.registry as registry
    models = registry.list_models()
    return {
        "count":  len(models),
        "models": models,
        "in_progress": list(_training_in_progress),
    }


@router.get("/models/{ticker}")
async def get_model(ticker: str):
    """Detailed model stats and backtest results for a specific ticker."""
    import models.registry as registry
    entry = registry.get(ticker.upper())
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No trained model found for {ticker.upper()}")
    return entry["meta"]

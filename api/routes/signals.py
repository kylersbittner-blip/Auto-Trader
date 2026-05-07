from fastapi import APIRouter, Query
from datetime import datetime
from engine.signal_engine import get_engine
from models.signal import SignalResponse

router = APIRouter()


@router.get("", response_model=SignalResponse)
async def get_signals(
    ticker: str | None = Query(None, description="Filter by ticker"),
    action: str | None = Query(None, description="Filter by action: buy|sell|hold"),
    min_confidence: float = Query(0, ge=0, le=100),
):
    """Return the latest AI-generated trade signals."""
    engine = get_engine()
    signals = engine.get_latest_signals()

    if ticker:
        signals = [s for s in signals if s.ticker.upper() == ticker.upper()]
    if action:
        signals = [s for s in signals if s.action.value == action.lower()]
    if min_confidence:
        signals = [s for s in signals if s.confidence >= min_confidence]

    return SignalResponse(
        signals=signals,
        generated_at=engine.last_scan_at or datetime.utcnow(),
        engine_running=engine.running,
    )

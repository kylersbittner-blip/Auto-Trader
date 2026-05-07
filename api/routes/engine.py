from fastapi import APIRouter, Depends, HTTPException
from api.security import require_control_key
from engine.signal_engine import get_engine
from models.signal import EngineConfig, EngineStatus

router = APIRouter()


@router.get("/status", response_model=EngineStatus)
async def engine_status():
    """Return current engine status and config."""
    return get_engine().get_status()


@router.post("/start", dependencies=[Depends(require_control_key)])
async def start_engine():
    """Start the signal engine loop."""
    engine = get_engine()
    await engine.start()
    return {"status": "started"}


@router.post("/stop", dependencies=[Depends(require_control_key)])
async def stop_engine():
    """Pause the signal engine."""
    engine = get_engine()
    await engine.stop()
    return {"status": "stopped"}


@router.put("/config", dependencies=[Depends(require_control_key)])
async def update_config(config: EngineConfig):
    """Update engine configuration (strategy, risk, watchlist, etc.)."""
    if config.auto_execute and config.max_daily_trades <= 0:
        raise HTTPException(
            status_code=400,
            detail="max_daily_trades must be greater than zero when auto_execute is enabled.",
        )
    engine = get_engine()
    engine.update_config(config)
    return {"status": "updated", "config": config}

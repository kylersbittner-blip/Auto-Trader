from fastapi import APIRouter, Depends, HTTPException
from api.security import require_control_key
from engine.signal_engine import get_engine

router = APIRouter(dependencies=[Depends(require_control_key)])


@router.get("")
async def get_portfolio():
    """Return live account summary and open positions from Alpaca."""
    engine = get_engine()
    try:
        account = engine.executor.get_account()
        positions = engine.executor.get_positions()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Broker unavailable: {e}")

    return {
        "account": account,
        "positions": positions,
        "position_count": len(positions),
    }

from fastapi import APIRouter, Depends, HTTPException
from api.security import require_control_key
from engine.signal_engine import get_engine

router = APIRouter(dependencies=[Depends(require_control_key)])


@router.get("")
async def get_portfolio():
    """Return live account summary and open positions from Alpaca."""
    engine = get_engine()
    try:
        account   = engine.executor.get_account()
        positions = engine.executor.get_positions()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Broker unavailable: {e}")

    return {
        "account":        account,
        "positions":      positions,
        "position_count": len(positions),
    }


@router.get("/equity-curve")
async def get_equity_curve():
    """Return historical equity snapshots for the P&L chart."""
    from data.equity_tracker import get_curve
    return {"curve": get_curve()}


@router.get("/performance")
async def get_performance():
    """Return strategy win-rates, outcomes summary, and trade history."""
    from engine.strategy_learner import get_summary
    from data.trade_outcomes import get_summary as outcome_summary, _load
    return {
        "strategy_performance": get_summary(),
        "overall":              outcome_summary(),
        "recent_trades":        _load()[-50:],  # last 50 outcomes
    }

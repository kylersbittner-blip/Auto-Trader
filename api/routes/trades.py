from fastapi import APIRouter, Depends, HTTPException
from api.security import require_control_key
from engine.signal_engine import get_engine
from engine.risk_manager import RiskViolation
from models.signal import ManualTradeRequest, Trade

router = APIRouter(dependencies=[Depends(require_control_key)])


@router.post("", response_model=Trade)
async def place_manual_trade(req: ManualTradeRequest):
    """
    Manually place a trade. Bypasses signal confidence check
    but still enforces position sizing and daily loss limit.
    """
    engine = get_engine()
    try:
        trade = await engine.executor.execute_signal(
            ticker=req.ticker,
            action=req.side.value,
            price=req.limit_price or 0,
            confidence=100.0,             # manual override — skip conf check
            daily_pnl=engine.daily_pnl,
            trades_today=engine.trades_today,
            requested_qty=req.qty,
        )
        engine.trades_today += 1
        return trade
    except RiskViolation as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

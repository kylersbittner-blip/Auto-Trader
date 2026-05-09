"""
Trade executor — places and tracks orders via the Alpaca brokerage API.
Supports market orders and bracket orders (with stop loss + take profit).
"""
from datetime import datetime
from typing import Optional
import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

from config import get_settings
from data.market_data import fetch_latest_quote
from models.signal import Trade, OrderStatus
from engine.risk_manager import RiskManager, RiskViolation

log = structlog.get_logger()
settings = get_settings()


def _get_client() -> TradingClient:
    if settings.trading_mode == "live" and not settings.allow_live_trading:
        raise RuntimeError(
            "Live trading is disabled. Set TRADING_MODE=paper, or set ALLOW_LIVE_TRADING=true deliberately."
        )

    return TradingClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        paper=(settings.trading_mode == "paper"),
    )


class TradeExecutor:
    def __init__(self, risk_manager: RiskManager):
        self.risk = risk_manager
        self.client = _get_client()

    async def execute_signal(
        self,
        ticker: str,
        action: str,           # "buy" | "sell"
        price: float,
        confidence: float,
        daily_pnl: float,
        trades_today: int = 0,
        requested_qty: Optional[float] = None,
    ) -> Trade:
        """
        Full execution flow:
          1. Risk check
          2. Size position
          3. Compute stop/target
          4. Place bracket order
          5. Return Trade record
        """
        from models.signal import Action, OrderSide as ModelSide
        action_enum = Action(action)

        # Risk gate — raises RiskViolation if blocked
        self.risk.check_signal(ticker, action_enum, confidence, daily_pnl, trades_today)

        if price <= 0:
            quote = await fetch_latest_quote(ticker)
            price = float(quote["price"]) if quote and quote.get("price") else 0.0

        if price <= 0:
            raise RiskViolation(f"{ticker}: no positive price available for execution")

        qty = self.risk.compute_qty(price, action, requested_qty=requested_qty)
        levels = self.risk.compute_stop_and_target(price, action)

        side = OrderSide.BUY if action == "buy" else OrderSide.SELL

        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=levels["take_profit"]),
            stop_loss=StopLossRequest(stop_price=levels["stop_loss"]),
        )

        log.info("placing_order", ticker=ticker, side=action, qty=qty, price=price)

        try:
            order = self.client.submit_order(order_request)
            broker_id = str(order.id)
            status = OrderStatus.PENDING
        except Exception as e:
            log.error("order_failed", ticker=ticker, error=str(e))
            raise

        return Trade(
            ticker=ticker,
            side=ModelSide(action),
            qty=qty,
            price=price,
            total_usd=round(qty * price, 2),
            status=status,
            signal_confidence=confidence,
            broker_order_id=broker_id,
            timestamp=datetime.utcnow(),
        )

    def get_positions(self) -> list[dict]:
        """Fetch all open positions from Alpaca."""
        positions = self.client.get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
                "side": p.side,
            }
            for p in positions
        ]

    def get_recent_closed_orders(self, limit: int = 50) -> list[dict]:
        """Fetch recently filled orders — used to detect bracket exits."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        try:
            req    = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=limit)
            orders = self.client.get_orders(req)
            return [
                {
                    "broker_id":   str(o.id),
                    "ticker":      str(o.symbol),
                    "side":        str(o.side.value),
                    "filled_qty":  float(o.filled_qty or 0),
                    "filled_avg":  float(o.filled_avg_price or 0),
                    "status":      str(o.status.value),
                    "filled_at":   str(o.filled_at) if o.filled_at else None,
                }
                for o in orders
                if o.filled_avg_price is not None
            ]
        except Exception as e:
            log.warning("get_closed_orders_failed", error=str(e))
            return []

    def get_account(self) -> dict:
        """Fetch account summary (equity, cash, buying power)."""
        acct = self.client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "day_pnl": float(acct.equity) - float(acct.last_equity),
        }

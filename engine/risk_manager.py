"""
Risk manager — enforces position sizing, exposure limits, and daily loss halt.
All checks are synchronous so they can be called inline before order placement.
"""
import structlog
from models.signal import EngineConfig, Action

log = structlog.get_logger()


class RiskViolation(Exception):
    """Raised when a trade would violate a risk rule."""
    pass


class RiskManager:
    def __init__(self, config: EngineConfig):
        self.config = config

    def check_signal(
        self,
        ticker: str,
        action: Action,
        confidence: float,
        daily_pnl: float,
        trades_today: int = 0,
    ) -> None:
        """
        Gate check before executing a signal.
        Raises RiskViolation with a human-readable reason if any rule fails.
        """
        # 1. Confidence threshold
        if confidence < self.config.min_confidence:
            raise RiskViolation(
                f"{ticker}: confidence {confidence:.0f}% < threshold {self.config.min_confidence:.0f}%"
            )

        # 2. Daily loss halt
        if daily_pnl < -abs(self.config.daily_loss_limit_usd):
            raise RiskViolation(
                f"Daily loss limit hit (${daily_pnl:,.0f}). Engine halted for today."
            )

        # 3. Daily trade cap
        if trades_today >= self.config.max_daily_trades:
            raise RiskViolation(
                f"Daily trade limit hit ({trades_today}/{self.config.max_daily_trades})."
            )

        # 4. Hold signals never execute
        if action == Action.HOLD:
            raise RiskViolation(f"{ticker}: HOLD signal — no execution")

        log.info("risk_check_passed", ticker=ticker, action=action, confidence=confidence)

    def compute_qty(self, price: float, side: str, requested_qty: float | None = None) -> float:
        """
        Calculate share quantity based on max position size.
        Applies a risk-level multiplier.
        """
        if price <= 0:
            raise RiskViolation("Cannot size trade without a positive reference price.")

        multiplier = {
            "conservative": 0.5,
            "moderate": 1.0,
            "aggressive": 1.5,
        }.get(self.config.risk_level, 1.0)

        max_usd = self.config.max_position_usd * multiplier

        if requested_qty is not None:
            if requested_qty <= 0:
                raise RiskViolation("Requested quantity must be greater than zero.")
            requested_notional = requested_qty * price
            if requested_notional > max_usd:
                raise RiskViolation(
                    f"Requested trade ${requested_notional:,.2f} exceeds max position ${max_usd:,.2f}."
                )
            return round(requested_qty, 2)

        qty = max_usd / price
        return max(1, round(qty, 2))

    def compute_stop_and_target(self, entry_price: float, side: str) -> dict:
        """
        Returns stop loss and take profit prices for an order.
        """
        if entry_price <= 0:
            raise RiskViolation("Cannot compute stop/target without a positive entry price.")

        stop_mult = (1 - self.config.stop_loss_pct / 100) if side == "buy" else (1 + self.config.stop_loss_pct / 100)
        tp_mult   = (1 + self.config.take_profit_pct / 100) if side == "buy" else (1 - self.config.take_profit_pct / 100)

        return {
            "stop_loss":   round(entry_price * stop_mult, 2),
            "take_profit": round(entry_price * tp_mult, 2),
        }

    def max_drawdown_ok(self, positions: list[dict]) -> bool:
        """Check whether open positions are within acceptable drawdown."""
        total_unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)
        return total_unrealized > -self.config.daily_loss_limit_usd

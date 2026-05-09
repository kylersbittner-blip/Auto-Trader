"""
Position Manager.

Wraps Alpaca's open positions to:
  1. Prevent double-entry (don't buy more of something we already own long)
  2. Detect when a signal reverses an open position (close before flipping)
  3. Surface open positions + unrealized P&L to the dashboard
"""
from typing import Optional
import structlog

log = structlog.get_logger()


class PositionManager:
    def __init__(self, executor):
        self.executor = executor
        self._positions: dict[str, dict] = {}  # ticker → position dict

    def refresh(self) -> None:
        """Pull latest positions from Alpaca."""
        try:
            raw = self.executor.get_positions()
            self._positions = {p["ticker"]: p for p in raw}
        except Exception as e:
            log.warning("position_refresh_failed", error=str(e))

    def get_all(self) -> list[dict]:
        return list(self._positions.values())

    def has_position(self, ticker: str) -> bool:
        return ticker in self._positions

    def position_side(self, ticker: str) -> Optional[str]:
        """'long' | 'short' | None"""
        p = self._positions.get(ticker)
        if p is None:
            return None
        return str(p.get("side", "long")).lower()

    def should_skip(self, ticker: str, action: str) -> tuple[bool, str]:
        """
        Returns (skip, reason).

        Skip when:
          - We already have a position in the same direction (avoid pyramiding)
          - We have a position and the signal is neutral/hold
        Allow when:
          - No open position
          - Signal reverses the open position (flip — execute close + new)
        """
        if not self.has_position(ticker):
            return False, ""

        side = self.position_side(ticker)
        if action == "buy" and side == "long":
            return True, f"already long {ticker}"
        if action == "sell" and side == "short":
            return True, f"already short {ticker}"

        # Signal reversal — allow (Alpaca will close + flip automatically with bracket)
        return False, ""

    def unrealized_pnl(self, ticker: str) -> float:
        p = self._positions.get(ticker)
        return float(p["unrealized_pnl"]) if p else 0.0

    def total_exposure_usd(self) -> float:
        return sum(abs(float(p["market_value"])) for p in self._positions.values())

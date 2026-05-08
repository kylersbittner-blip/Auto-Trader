from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Strategy(str, Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"


class RiskLevel(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# ---------- Signal ----------

class Signal(BaseModel):
    ticker: str
    action: Action
    confidence: float = Field(..., ge=0, le=100)
    technical_score: float
    sentiment_score: float
    patterns_detected: list[str] = Field(default_factory=list)
    reasoning: str
    price: float
    regime: Optional[str] = None          # trending / ranging / breakout_setup
    active_strategy: Optional[str] = None # momentum / mean_reversion / breakout
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SignalResponse(BaseModel):
    signals: list[Signal]
    generated_at: datetime
    engine_running: bool


# ---------- Trade ----------

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Trade(BaseModel):
    id: Optional[str] = None
    ticker: str
    side: OrderSide
    qty: float
    price: float
    total_usd: float
    status: OrderStatus = OrderStatus.PENDING
    signal_confidence: Optional[float] = None
    broker_order_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None


class ManualTradeRequest(BaseModel):
    ticker: str
    side: OrderSide
    qty: float = Field(..., gt=0)
    limit_price: Optional[float] = Field(None, gt=0)   # None = market order


# ---------- Engine Config ----------

class EngineConfig(BaseModel):
    strategy: Strategy = Strategy.MOMENTUM
    risk_level: RiskLevel = RiskLevel.MODERATE
    min_confidence: float = Field(72.0, ge=0, le=100)
    max_position_usd: float = Field(5000.0, gt=0)
    stop_loss_pct: float = Field(2.0, gt=0, le=50)
    take_profit_pct: float = Field(4.0, gt=0, le=100)
    daily_loss_limit_usd: float = Field(2000.0, gt=0)
    max_daily_trades: int = Field(5, ge=0)
    watchlist: list[str] = Field(default_factory=lambda: ["NVDA", "AAPL", "TSLA", "META", "AMD"])
    auto_execute: bool = False


class EngineStatus(BaseModel):
    running: bool
    config: EngineConfig
    trades_today: int
    daily_pnl: float
    last_scan_at: Optional[datetime] = None

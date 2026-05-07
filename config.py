from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # Alpaca
    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field("https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
    trading_mode: str = Field("paper", env="TRADING_MODE")
    allow_live_trading: bool = Field(False, env="ALLOW_LIVE_TRADING")

    # Polygon
    polygon_api_key: str = Field(..., env="POLYGON_API_KEY")

    # Infrastructure
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    database_url: str = Field(..., env="DATABASE_URL")

    # Engine defaults
    watchlist: list[str] = Field(default_factory=lambda: ["NVDA","AAPL","TSLA","META","AMD"])
    auto_execute: bool = Field(False, env="AUTO_EXECUTE")
    min_confidence: float = Field(72.0, env="MIN_CONFIDENCE")
    max_position_usd: float = Field(5000.0, env="MAX_POSITION_USD")
    max_daily_trades: int = Field(5, env="MAX_DAILY_TRADES")
    daily_loss_limit_usd: float = Field(2000.0, env="DAILY_LOSS_LIMIT_USD")
    stop_loss_pct: float = Field(2.0, env="STOP_LOSS_PCT")
    take_profit_pct: float = Field(4.0, env="TAKE_PROFIT_PCT")
    strategy: str = Field("momentum", env="STRATEGY")

    # API controls
    control_api_key: Optional[str] = Field(None, env="CONTROL_API_KEY")

    # Model
    sentiment_model: str = Field("ProsusAI/finbert", env="SENTIMENT_MODEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @field_validator("watchlist", mode="before")
    @classmethod
    def parse_watchlist(cls, value):
        if isinstance(value, str):
            return [item.strip().upper() for item in value.split(",") if item.strip()]
        return value

    @field_validator("trading_mode")
    @classmethod
    def validate_trading_mode(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"paper", "live"}:
            raise ValueError("TRADING_MODE must be 'paper' or 'live'")
        return normalized

    @property
    def watchlist_str(self) -> str:
        return ",".join(self.watchlist)


@lru_cache
def get_settings() -> Settings:
    return Settings()

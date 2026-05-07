"""
Tests for pattern detector and risk manager.
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from engine.pattern_detector import detect_patterns
from engine.risk_manager import RiskManager, RiskViolation
from models.signal import EngineConfig, Action


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_bars(n=60, trend="up", noise=0.005) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(42)
    base = 100.0
    closes = []
    for i in range(n):
        if trend == "up":
            base *= 1 + 0.002 + rng.normal(0, noise)
        elif trend == "down":
            base *= 1 - 0.002 + rng.normal(0, noise)
        else:
            base *= 1 + rng.normal(0, noise)
        closes.append(base)

    closes = np.array(closes)
    timestamps = [datetime(2025, 1, 2, 9, 30) + timedelta(minutes=5 * i) for i in range(n)]

    df = pd.DataFrame({
        "open":   closes * (1 - 0.001),
        "high":   closes * (1 + 0.003),
        "low":    closes * (1 - 0.003),
        "close":  closes,
        "volume": rng.integers(100_000, 500_000, n).astype(float),
    }, index=timestamps)
    return df


# ------------------------------------------------------------------ #
# Pattern detector tests                                               #
# ------------------------------------------------------------------ #

class TestPatternDetector:
    def test_uptrend_produces_buy_signal(self):
        df = make_bars(60, trend="up")
        result = detect_patterns(df)
        assert result["action"] in ("buy", "hold")
        assert 0 <= result["score"] <= 100

    def test_downtrend_produces_sell_signal(self):
        df = make_bars(60, trend="down")
        result = detect_patterns(df)
        assert result["action"] in ("sell", "hold")

    def test_insufficient_data_returns_hold(self):
        df = make_bars(10)   # not enough bars
        result = detect_patterns(df)
        assert result["action"] == "hold"
        assert result["score"] == 0.0

    def test_empty_dataframe_returns_hold(self):
        result = detect_patterns(pd.DataFrame())
        assert result["action"] == "hold"

    def test_patterns_list_is_list(self):
        df = make_bars(60)
        result = detect_patterns(df)
        assert isinstance(result["patterns"], list)

    def test_score_within_bounds(self):
        for trend in ("up", "down", "flat"):
            df = make_bars(60, trend=trend)
            result = detect_patterns(df)
            assert 0 <= result["score"] <= 100, f"score out of range for {trend}"

    def test_mean_reversion_strategy(self):
        df = make_bars(60, trend="down")
        result = detect_patterns(df, strategy="mean_reversion")
        assert isinstance(result, dict)


# ------------------------------------------------------------------ #
# Risk manager tests                                                   #
# ------------------------------------------------------------------ #

class TestRiskManager:
    @pytest.fixture
    def config(self):
        return EngineConfig(
            min_confidence=70.0,
            max_position_usd=5000.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            daily_loss_limit_usd=2000.0,
        )

    @pytest.fixture
    def rm(self, config):
        return RiskManager(config)

    def test_passes_valid_signal(self, rm):
        rm.check_signal("NVDA", Action.BUY, confidence=85.0, daily_pnl=-100.0)

    def test_blocks_low_confidence(self, rm):
        with pytest.raises(RiskViolation, match="confidence"):
            rm.check_signal("NVDA", Action.BUY, confidence=60.0, daily_pnl=0.0)

    def test_blocks_daily_loss_limit(self, rm):
        with pytest.raises(RiskViolation, match="Daily loss"):
            rm.check_signal("NVDA", Action.BUY, confidence=90.0, daily_pnl=-2500.0)

    def test_blocks_hold_action(self, rm):
        with pytest.raises(RiskViolation, match="HOLD"):
            rm.check_signal("NVDA", Action.HOLD, confidence=90.0, daily_pnl=0.0)

    def test_position_sizing(self, rm):
        qty = rm.compute_qty(price=500.0, side="buy")
        assert qty >= 1
        assert qty * 500 <= 5000 * 1.01   # within max + tiny float tolerance

    def test_blocks_zero_price_sizing(self, rm):
        with pytest.raises(RiskViolation, match="positive reference price"):
            rm.compute_qty(price=0.0, side="buy")

    def test_honors_manual_qty_within_limit(self, rm):
        qty = rm.compute_qty(price=100.0, side="buy", requested_qty=12.5)
        assert qty == pytest.approx(12.5)

    def test_blocks_manual_qty_over_limit(self, rm):
        with pytest.raises(RiskViolation, match="exceeds max position"):
            rm.compute_qty(price=100.0, side="buy", requested_qty=100.0)

    def test_blocks_daily_trade_limit(self, rm, config):
        with pytest.raises(RiskViolation, match="Daily trade limit"):
            rm.check_signal(
                "NVDA",
                Action.BUY,
                confidence=90.0,
                daily_pnl=0.0,
                trades_today=config.max_daily_trades,
            )

    def test_stop_and_target_buy(self, rm):
        levels = rm.compute_stop_and_target(100.0, "buy")
        assert levels["stop_loss"] == pytest.approx(98.0, rel=1e-3)
        assert levels["take_profit"] == pytest.approx(104.0, rel=1e-3)

    def test_stop_and_target_sell(self, rm):
        levels = rm.compute_stop_and_target(100.0, "sell")
        assert levels["stop_loss"] > 100.0
        assert levels["take_profit"] < 100.0

    def test_aggressive_risk_multiplier(self, config):
        config.risk_level = "aggressive"
        rm = RiskManager(config)
        qty = rm.compute_qty(price=100.0, side="buy")
        assert qty * 100 <= 5000 * 1.5 * 1.01

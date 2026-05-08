"""
Signal engine — main orchestration loop.
Runs every 60 seconds, scans all tickers, combines:
  - ML model prediction  (50% when model available)
  - Technical patterns   (30% when model available, 65% fallback)
  - News sentiment       (20% when model available, 35% fallback)
"""
import asyncio
from datetime import datetime
from typing import Optional
import structlog

from config import get_settings
from data.alpaca_feed import fetch_bars_batch
from data.cache import publish_signal, set_engine_state
from engine.pattern_detector import detect_patterns
from engine.news_scanner import scan_all
from engine.risk_manager import RiskManager, RiskViolation
from engine.trade_executor import TradeExecutor
from models.signal import Signal, Action, EngineConfig, EngineStatus
from activity import get_activity_logger

log = structlog.get_logger()
settings = get_settings()


class SignalEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig(
            strategy         = settings.strategy,
            min_confidence   = settings.min_confidence,
            max_position_usd = settings.max_position_usd,
            stop_loss_pct    = settings.stop_loss_pct,
            take_profit_pct  = settings.take_profit_pct,
            daily_loss_limit_usd = settings.daily_loss_limit_usd,
            max_daily_trades = settings.max_daily_trades,
            watchlist        = settings.watchlist,
            auto_execute     = settings.auto_execute,
        )
        self.running        = False
        self.last_scan_at:  Optional[datetime] = None
        self.trades_today   = 0
        self.daily_pnl      = 0.0
        self._task:         Optional[asyncio.Task] = None
        self._signals:      list[Signal] = []

        self.risk     = RiskManager(self.config)
        self.executor = TradeExecutor(self.risk)

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        await set_engine_state({"running": True})
        log.info("engine_started", strategy=self.config.strategy)
        get_activity_logger().info("engine", f"Engine started — strategy: {self.config.strategy}")

        # Load any already-trained models at startup
        import models.registry as registry
        registry.load_all(self.config.watchlist)

        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self.running = False
        await set_engine_state({"running": False})
        if self._task:
            self._task.cancel()
        log.info("engine_stopped")
        get_activity_logger().info("engine", "Engine stopped")

    def update_config(self, new_config: EngineConfig) -> None:
        self.config   = new_config
        self.risk     = RiskManager(new_config)
        self.executor = TradeExecutor(self.risk)
        log.info("config_updated", strategy=new_config.strategy)

    def get_status(self) -> EngineStatus:
        return EngineStatus(
            running       = self.running,
            config        = self.config,
            trades_today  = self.trades_today,
            daily_pnl     = self.daily_pnl,
            last_scan_at  = self.last_scan_at,
        )

    def get_latest_signals(self) -> list[Signal]:
        return self._signals

    # ── Internal loop ─────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self.running:
            try:
                await self._scan_and_emit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("engine_loop_error", error=str(e))
                get_activity_logger().failure("scan", f"Engine loop error: {e}")
            await asyncio.sleep(60)

    async def _scan_and_emit(self) -> None:
        tickers  = self.config.watchlist
        activity = get_activity_logger()
        log.info("scan_started", tickers=tickers)

        bars_map, sentiment_map = await asyncio.gather(
            fetch_bars_batch(tickers),
            scan_all(tickers),
        )

        try:
            account      = self.executor.get_account()
            self.daily_pnl = account["day_pnl"]
        except Exception:
            pass

        signals = []
        for ticker in tickers:
            df        = bars_map.get(ticker)
            sentiment = sentiment_map.get(ticker, {"score": 50.0, "label": "neutral"})

            if df is None or df.empty:
                activity.failure("data", f"No market data for {ticker}", ticker=ticker)
                continue

            tech_result = detect_patterns(df, strategy=self.config.strategy)
            tech_score  = tech_result["score"]
            sent_score  = sentiment["score"]

            # Try ML model — use it if available and trained
            ml_pred    = self._ml_predict(ticker, df)
            ml_score   = ml_pred["confidence"] if ml_pred else None

            # Ensemble weights depend on whether a trained model exists
            if ml_score is not None:
                ml_action   = ml_pred["action"]
                confidence  = (ml_score * 0.50 + tech_score * 0.30 + sent_score * 0.20)
                action      = ml_action   # ML action takes lead
            else:
                confidence  = tech_score * 0.65 + sent_score * 0.35
                action      = tech_result["action"]

            # Conflict resolution: if sentiment strongly disagrees with direction, hold
            if sent_score < 35 and action == "buy":
                action     = "hold"
                confidence *= 0.70
            elif sent_score > 65 and action == "sell":
                action     = "hold"
                confidence *= 0.70

            price  = float(df["close"].iloc[-1])
            signal = Signal(
                ticker             = ticker,
                action             = Action(action),
                confidence         = round(confidence, 1),
                technical_score    = round(tech_score, 1),
                sentiment_score    = round(sent_score, 1),
                patterns_detected  = tech_result["patterns"],
                reasoning          = tech_result["reasoning"],
                price              = price,
            )
            signals.append(signal)
            await publish_signal(signal.model_dump(mode="json"))

            if self.config.auto_execute and action != "hold":
                await self._try_execute(signal)

        self._signals    = signals
        self.last_scan_at = datetime.utcnow()
        summary = ", ".join(f"{s.ticker}:{s.action.value}@{s.confidence:.0f}%" for s in signals)
        log.info("scan_complete", signals=len(signals))
        activity.success("scan", f"Scan complete — {len(signals)} signals", detail=summary)

    def _ml_predict(self, ticker: str, df) -> Optional[dict]:
        """Return ML prediction dict or None if no model exists yet."""
        try:
            import models.registry as registry
            from models.trainer import predict_latest
            entry = registry.get(ticker)
            if entry is None:
                return None
            return predict_latest(
                entry["model"],
                entry["label_encoder"],
                entry["feature_cols"],
                df,
            )
        except Exception as e:
            log.warning("ml_predict_failed", ticker=ticker, error=str(e))
            return None

    async def _try_execute(self, signal: Signal) -> None:
        activity = get_activity_logger()
        try:
            trade = await self.executor.execute_signal(
                ticker       = signal.ticker,
                action       = signal.action.value,
                price        = signal.price,
                confidence   = signal.confidence,
                daily_pnl    = self.daily_pnl,
                trades_today = self.trades_today,
            )
            self.trades_today += 1
            log.info("trade_executed", ticker=trade.ticker, side=trade.side, qty=trade.qty)
            activity.success(
                "trade",
                f"{trade.ticker} {trade.side.upper()} {trade.qty:.0f} shares @ ${trade.price:.2f}",
                detail   = f"Total: ${trade.total_usd:.2f} | Confidence: {signal.confidence:.0f}%",
                ticker   = signal.ticker,
            )
        except RiskViolation as e:
            log.info("trade_blocked_by_risk", reason=str(e))
            activity.warning("risk", f"Trade blocked for {signal.ticker}: {e}", ticker=signal.ticker)
        except Exception as e:
            log.error("trade_execution_failed", ticker=signal.ticker, error=str(e))
            activity.failure("trade", f"Trade failed for {signal.ticker}: {e}", ticker=signal.ticker)


_engine: Optional[SignalEngine] = None


def get_engine() -> SignalEngine:
    global _engine
    if _engine is None:
        _engine = SignalEngine()
    return _engine

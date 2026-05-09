"""
Signal engine — main orchestration loop.

Phase 4 additions:
  - Market hours filter (only trades in high-liquidity windows)
  - Position manager (prevents double-entry, surfaces open positions)
  - Strategy learner (adjusts weights from real win/loss history)
  - Kelly-criterion position sizing (scales with edge, not fixed $)
  - Equity curve snapshots every scan
"""
import asyncio
from datetime import datetime
from typing import Optional
import structlog

from config import get_settings
from data.alpaca_feed import fetch_bars_batch
from data.cache import publish_signal, set_engine_state
from data.trade_outcomes import record_entry, record_exit, get_open_entries, should_retrain, mark_used_in_training
from data.equity_tracker import record_snapshot
from engine.pattern_detector import detect_patterns
from engine.regime_detector import detect_regime
from engine.strategies import detect_mean_reversion, detect_breakout
from engine.news_scanner import scan_all
from engine.risk_manager import RiskManager, RiskViolation
from engine.trade_executor import TradeExecutor
from engine.position_manager import PositionManager
from engine.strategy_learner import get_weight, get_best_strategy, record_outcome
from engine.position_sizer import kelly_size
from engine.market_hours import is_trading_window
from models.signal import Signal, Action, EngineConfig, EngineStatus
from activity import get_activity_logger

log = structlog.get_logger()
settings = get_settings()

REGIME_STRATEGY = {
    "trending":       "momentum",
    "ranging":        "mean_reversion",
    "breakout_setup": "breakout",
}


class SignalEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig(
            strategy             = settings.strategy,
            min_confidence       = settings.min_confidence,
            max_position_usd     = settings.max_position_usd,
            stop_loss_pct        = settings.stop_loss_pct,
            take_profit_pct      = settings.take_profit_pct,
            daily_loss_limit_usd = settings.daily_loss_limit_usd,
            max_daily_trades     = settings.max_daily_trades,
            watchlist            = settings.watchlist,
            auto_execute         = settings.auto_execute,
        )
        self.running       = False
        self.last_scan_at: Optional[datetime] = None
        self.trades_today  = 0
        self.daily_pnl     = 0.0
        self._task:        Optional[asyncio.Task] = None
        self._signals:     list[Signal] = []
        self._account:     dict = {}

        self.risk     = RiskManager(self.config)
        self.executor = TradeExecutor(self.risk)
        self.positions = PositionManager(self.executor)

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        await set_engine_state({"running": True})
        log.info("engine_started", strategy=self.config.strategy)
        get_activity_logger().info("engine", f"Engine started — strategy: {self.config.strategy}")

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
            running      = self.running,
            config       = self.config,
            trades_today = self.trades_today,
            daily_pnl    = self.daily_pnl,
            last_scan_at = self.last_scan_at,
        )

    def get_latest_signals(self) -> list[Signal]:
        return self._signals

    def get_open_positions(self) -> list[dict]:
        return self.positions.get_all()

    def get_account(self) -> dict:
        return self._account

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

        # ── Market hours gate ─────────────────────────────────────────────────
        allowed, window_reason = is_trading_window()
        if not allowed:
            log.info("outside_trading_window", reason=window_reason)
            # Still scan for signals but suppress execution
        else:
            log.info("scan_started", tickers=tickers, window=window_reason)

        # ── Reconcile closed trades ───────────────────────────────────────────
        self._reconcile_closed_trades()

        # ── Refresh positions + account ────────────────────────────────────────
        self.positions.refresh()
        try:
            acct = self.executor.get_account()
            self._account  = acct
            self.daily_pnl = acct["day_pnl"]
            record_snapshot(acct["equity"], acct["cash"])
        except Exception:
            pass

        bars_map, sentiment_map = await asyncio.gather(
            fetch_bars_batch(tickers),
            scan_all(tickers),
        )

        signals = []
        for ticker in tickers:
            df        = bars_map.get(ticker)
            sentiment = sentiment_map.get(ticker, {"score": 50.0, "label": "neutral"})

            if df is None or df.empty:
                activity.failure("data", f"No market data for {ticker}", ticker=ticker)
                continue

            # ── Regime detection ──────────────────────────────────────────────
            regime = detect_regime(df)

            # ── Strategy selection (learner can override regime default) ──────
            learned_strategy = get_best_strategy(ticker, regime)
            active_strategy  = learned_strategy or REGIME_STRATEGY.get(regime, "momentum")

            if active_strategy == "mean_reversion":
                tech_result = detect_mean_reversion(df)
            elif active_strategy == "breakout":
                tech_result = detect_breakout(df)
            else:
                tech_result = detect_patterns(df, strategy="momentum")

            # Apply learned weight to tech score
            weight     = get_weight(ticker, active_strategy, regime)
            tech_score = min(100.0, tech_result["score"] * weight)
            sent_score = sentiment["score"]

            # ── ML ensemble ───────────────────────────────────────────────────
            ml_pred  = self._ml_predict(ticker, df)
            ml_score = ml_pred["confidence"] if ml_pred else None

            if ml_score is not None:
                confidence = ml_score * 0.50 + tech_score * 0.30 + sent_score * 0.20
                action     = ml_pred["action"]
            else:
                confidence = tech_score * 0.65 + sent_score * 0.35
                action     = tech_result["action"]

            # Sentiment conflict resolution
            if sent_score < 35 and action == "buy":
                action      = "hold"
                confidence *= 0.70
            elif sent_score > 65 and action == "sell":
                action      = "hold"
                confidence *= 0.70

            price  = float(df["close"].iloc[-1])
            signal = Signal(
                ticker            = ticker,
                action            = Action(action),
                confidence        = round(confidence, 1),
                technical_score   = round(tech_score, 1),
                sentiment_score   = round(sent_score, 1),
                patterns_detected = tech_result["patterns"],
                reasoning         = tech_result["reasoning"],
                price             = price,
                regime            = regime,
                active_strategy   = active_strategy,
            )
            signals.append(signal)
            await publish_signal(signal.model_dump(mode="json"))

            if self.config.auto_execute and action != "hold" and allowed:
                await self._try_execute(signal, regime, active_strategy)
            elif self.config.auto_execute and action != "hold" and not allowed:
                activity.info(
                    "scan",
                    f"Signal {ticker} {action.upper()} skipped — {window_reason}",
                    ticker=ticker,
                )

            # ── Incremental retrain check ─────────────────────────────────────
            if should_retrain(ticker):
                activity.info("scan", f"20 new outcomes for {ticker} — triggering retrain…", ticker=ticker)
                mark_used_in_training(ticker)
                asyncio.create_task(self._retrain_ticker(ticker))

        self._signals     = signals
        self.last_scan_at = datetime.utcnow()
        summary = ", ".join(
            f"{s.ticker}:{s.action.value}@{s.confidence:.0f}%[{s.regime}]"
            for s in signals
        )
        log.info("scan_complete", signals=len(signals))
        activity.success("scan", f"Scan complete — {len(signals)} signals", detail=summary)

    # ── ML prediction ─────────────────────────────────────────────────────────

    def _ml_predict(self, ticker: str, df) -> Optional[dict]:
        try:
            import models.registry as registry
            from models.trainer import predict_latest
            entry = registry.get(ticker)
            if entry is None:
                return None
            return predict_latest(entry["model"], entry["label_encoder"], entry["feature_cols"], df)
        except Exception as e:
            log.warning("ml_predict_failed", ticker=ticker, error=str(e))
            return None

    # ── Closed-trade reconciliation ───────────────────────────────────────────

    def _reconcile_closed_trades(self) -> None:
        """
        Match Alpaca's recently filled orders against our open trade entries.
        For each match: record the exit price, compute P&L, and feed the
        strategy learner so it can adjust future weights.
        """
        open_entries = get_open_entries()
        if not open_entries:
            return

        try:
            closed_orders = self.executor.get_recent_closed_orders(limit=100)
        except Exception:
            return

        # Build a lookup: broker_id → filled_avg for fast matching
        closed_by_id = {o["broker_id"]: o for o in closed_orders if o.get("filled_avg")}

        for entry in open_entries:
            broker_id = entry.get("broker_id")
            if not broker_id or broker_id not in closed_by_id:
                continue

            filled = closed_by_id[broker_id]
            exit_price = filled["filled_avg"]
            if not exit_price:
                continue

            actual_return = record_exit(entry["id"], exit_price)
            if actual_return is None:
                continue

            won = actual_return > 0
            record_outcome(
                ticker     = entry["ticker"],
                strategy   = entry.get("strategy", "momentum"),
                regime     = entry.get("regime", "ranging"),
                won        = won,
                return_pct = actual_return * 100,
            )

            activity = get_activity_logger()
            pnl_str = f"+{actual_return*100:.2f}%" if won else f"{actual_return*100:.2f}%"
            activity.success(
                "trade",
                f"{entry['ticker']} closed {pnl_str} — strategy learner updated",
                ticker=entry["ticker"],
            )
            log.info(
                "trade_reconciled",
                trade_id=entry["id"],
                ticker=entry["ticker"],
                actual_return=round(actual_return, 4),
                won=won,
            )

    # ── Trade execution ───────────────────────────────────────────────────────

    async def _try_execute(self, signal: Signal, regime: str, strategy: str) -> None:
        activity = get_activity_logger()

        # Position guard — don't pyramid
        skip, skip_reason = self.positions.should_skip(signal.ticker, signal.action.value)
        if skip:
            log.info("position_skip", ticker=signal.ticker, reason=skip_reason)
            return

        # Kelly sizing
        account_equity = self._account.get("equity", self.config.max_position_usd * 5)
        position_usd   = kelly_size(
            ticker         = signal.ticker,
            strategy       = strategy,
            regime         = regime,
            account_equity = account_equity,
            max_position   = self.config.max_position_usd,
            confidence     = signal.confidence,
        )

        try:
            trade = await self.executor.execute_signal(
                ticker        = signal.ticker,
                action        = signal.action.value,
                price         = signal.price,
                confidence    = signal.confidence,
                daily_pnl     = self.daily_pnl,
                trades_today  = self.trades_today,
                requested_qty = position_usd / signal.price if signal.price > 0 else None,
            )
            self.trades_today += 1
            broker_id = trade.broker_order_id

            record_entry(
                ticker    = signal.ticker,
                action    = signal.action.value,
                price     = signal.price,
                strategy  = strategy,
                regime    = regime,
                broker_id = broker_id,
            )

            log.info("trade_executed", ticker=trade.ticker, side=trade.side, qty=trade.qty)
            activity.success(
                "trade",
                f"{trade.ticker} {trade.side.upper()} {trade.qty:.0f} @ ${trade.price:.2f}"
                f" [{strategy}/{regime}] ${position_usd:.0f}",
                detail = f"Total: ${trade.total_usd:.2f} | Conf: {signal.confidence:.0f}% | Kelly: ${position_usd:.0f}",
                ticker = signal.ticker,
            )
        except RiskViolation as e:
            log.info("trade_blocked_by_risk", reason=str(e))
            activity.warning("risk", f"Trade blocked for {signal.ticker}: {e}", ticker=signal.ticker)
        except Exception as e:
            log.error("trade_execution_failed", ticker=signal.ticker, error=str(e))
            activity.failure("trade", f"Trade failed for {signal.ticker}: {e}", ticker=signal.ticker)

    # ── Incremental retrain ───────────────────────────────────────────────────

    async def _retrain_ticker(self, ticker: str) -> None:
        activity = get_activity_logger()
        try:
            from data.alpaca_feed import fetch_historical
            from models.trainer import walk_forward_train
            from models.backtester import run_backtest
            import models.registry as registry
            import gc

            activity.info("scan", f"Retrain: fetching fresh data for {ticker}…", ticker=ticker)
            df = await fetch_historical(ticker, days=365)
            if df.empty or len(df) < 200:
                activity.failure("scan", f"Not enough data for retrain of {ticker}", ticker=ticker)
                return

            activity.info("scan", f"Retrain: training on {len(df):,} bars for {ticker}…", ticker=ticker)
            train_result = await asyncio.to_thread(walk_forward_train, df)
            if "error" in train_result:
                activity.failure("scan", f"Retrain failed for {ticker}: {train_result['error']}", ticker=ticker)
                return

            bt_result = await asyncio.to_thread(run_backtest, df)
            del df
            gc.collect()

            registry.save(ticker, train_result, bt_result)
            summary = (
                f"{train_result['n_folds']} folds | "
                f"dir_acc={train_result['avg_dir_accuracy']:.1%} | "
                f"Sharpe={bt_result.get('sharpe_ratio', '?')}"
            )
            activity.success("scan", f"Retrain complete for {ticker}", detail=summary, ticker=ticker)
            registry.load_all([ticker])
        except Exception as e:
            get_activity_logger().failure("scan", f"Retrain error for {ticker}: {e}", ticker=ticker)


_engine: Optional[SignalEngine] = None


def get_engine() -> SignalEngine:
    global _engine
    if _engine is None:
        _engine = SignalEngine()
    return _engine

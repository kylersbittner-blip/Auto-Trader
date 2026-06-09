"""
Session Runner — orchestrates a complete trading day.

Ties together scanner, regime classifier, session manager,
position sizing (1% risk per trade), and trade logging.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional, Protocol
import zoneinfo

import pandas as pd

from engine.regime import (
    SessionManager, RegimeConfig, DayRegime, SessionPhase, get_session_phase,
)
from engine.scanner import (
    StocksInPlayScanner, ScannerConfig, ScanResult, MarketDataSource, NewsSource,
)
from engine.opening_range import identify_opening_range, compute_gap

ET = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")


def compute_position_size(
    account_equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.01,
    max_position_pct: float = 0.10,
    regime_modifier: float = 1.0,
) -> dict:
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0 or entry_price <= 0 or account_equity <= 0:
        return {"qty": 0, "risk_per_share": 0.0, "total_risk_usd": 0.0, "position_value_usd": 0.0, "risk_pct_of_equity": 0.0}

    risk_budget = account_equity * risk_pct * regime_modifier
    qty_from_risk = risk_budget / risk_per_share
    max_qty_from_position = (account_equity * max_position_pct) / entry_price
    qty = int(min(qty_from_risk, max_qty_from_position))
    qty = max(qty, 0)

    return {
        "qty": qty,
        "risk_per_share": round(risk_per_share, 4),
        "total_risk_usd": round(qty * risk_per_share, 2),
        "position_value_usd": round(qty * entry_price, 2),
        "risk_pct_of_equity": round((qty * risk_per_share) / account_equity * 100, 2) if account_equity > 0 else 0.0,
    }


@dataclass
class TradeRecord:
    timestamp: datetime
    ticker: str
    strategy: str
    action: str
    direction: Optional[str] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    qty: int = 0
    risk_usd: float = 0.0
    regime: str = ""
    reason: Optional[str] = None


@dataclass
class SessionSummary:
    date: str
    regime: str
    candidates_scanned: int
    candidates_passed: int
    trades_entered: int
    trades_exited: int
    trades_skipped: int
    trade_log: list[TradeRecord] = field(default_factory=list)


class SessionRunner:
    def __init__(
        self,
        data_source: MarketDataSource,
        news_source: NewsSource,
        account_equity: float = 10_000.0,
        risk_pct: float = 0.01,
        scanner_config: Optional[ScannerConfig] = None,
        regime_config: Optional[RegimeConfig] = None,
        universe: Optional[list[str]] = None,
    ):
        self.data_source = data_source
        self.news_source = news_source
        self.account_equity = account_equity
        self.risk_pct = risk_pct
        self.scanner = StocksInPlayScanner(data_source, news_source, scanner_config)
        self.session_mgr = SessionManager(regime_config)
        self.universe = universe or [
            "NVDA", "TSLA", "AAPL", "META", "AMD",
            "MSFT", "AMZN", "GOOGL", "NFLX", "AVGO",
        ]
        self.scan_result: Optional[ScanResult] = None
        self.active_tickers: list[str] = []
        self.positions: dict[str, dict] = {}
        self.trade_log: list[TradeRecord] = []

    def run_premarket_scan(self) -> ScanResult:
        self.scan_result = self.scanner.scan(self.universe)
        self.active_tickers = [c.ticker for c in self.scan_result.candidates]
        return self.scan_result

    def initialize_session(
        self, bars_by_ticker, prev_closes,
        spy_bars=None, spy_prev_close=0.0, vix=None,
    ) -> dict:
        spy_range_width_pct = None
        spy_gap_pct = None
        spy_breakout = False

        if spy_bars is not None and not spy_bars.empty:
            spy_or = identify_opening_range(spy_bars)
            if spy_or is not None:
                spy_range_width_pct = spy_or["range_width_pct"]
            spy_gap = compute_gap(float(spy_bars.iloc[0]["open"]), spy_prev_close) if spy_prev_close > 0 else None
            if spy_gap:
                spy_gap_pct = spy_gap["gap_pct"]

        regime = self.session_mgr.set_regime(
            spy_range_width_pct=spy_range_width_pct, spy_gap_pct=spy_gap_pct,
            spy_breakout=spy_breakout, vix=vix,
        )
        init_results = self.session_mgr.initialize_strategies(
            self.active_tickers, bars_by_ticker, prev_closes,
        )
        return {"regime": regime, "init_results": init_results, "active_tickers": self.active_tickers}

    def activate_afternoon(self, bars_by_ticker) -> dict:
        return self.session_mgr.activate_afternoon_session(bars_by_ticker)

    def process_bar(self, ticker, bars, current_price, current_time_et) -> Optional[TradeRecord]:
        pos = self.positions.get(ticker)
        pos_dir = pos["direction"] if pos else None
        pos_strat = pos["strategy"] if pos else None

        action = self.session_mgr.on_bar(
            ticker, bars, current_price, current_time_et,
            position_direction=pos_dir, position_strategy=pos_strat,
        )
        regime_str = action.regime.value if isinstance(action.regime, DayRegime) else str(action.regime)

        if action.exit_signal and pos:
            exit_sig = action.exit_signal
            close_pct = getattr(exit_sig, 'close_pct', 1.0)
            record = TradeRecord(
                timestamp=current_time_et, ticker=ticker, strategy=pos["strategy"],
                action="exit", direction=pos["direction"], entry_price=pos["entry_price"],
                exit_price=exit_sig.exit_price, exit_reason=exit_sig.reason,
                qty=int(pos["qty"] * close_pct), regime=regime_str,
            )
            self.trade_log.append(record)
            if close_pct >= 1.0:
                del self.positions[ticker]
                self.session_mgr.mark_orb_exited(ticker)
            else:
                self.positions[ticker]["qty"] = int(pos["qty"] * (1 - close_pct))
            return record

        if action.signal and not pos:
            signal = action.signal
            strategy = "orb" if action.action == "scan_orb" else "vwap"
            sizing = compute_position_size(
                account_equity=self.account_equity, entry_price=signal.entry_price,
                stop_price=signal.stop_loss, risk_pct=self.risk_pct,
                regime_modifier=action.position_size_modifier,
            )
            if sizing["qty"] <= 0:
                record = TradeRecord(
                    timestamp=current_time_et, ticker=ticker, strategy=strategy,
                    action="skip", direction=signal.direction, entry_price=signal.entry_price,
                    stop_price=signal.stop_loss, regime=regime_str,
                    reason="Position size computed to 0 shares",
                )
                self.trade_log.append(record)
                return record

            target = signal.take_profit if hasattr(signal, 'take_profit') else getattr(signal, 'target_2', None)
            record = TradeRecord(
                timestamp=current_time_et, ticker=ticker, strategy=strategy,
                action="entry", direction=signal.direction, entry_price=signal.entry_price,
                stop_price=signal.stop_loss, target_price=target, qty=sizing["qty"],
                risk_usd=sizing["total_risk_usd"], regime=regime_str,
            )
            self.trade_log.append(record)
            self.positions[ticker] = {
                "direction": signal.direction, "strategy": strategy,
                "qty": sizing["qty"], "entry_price": signal.entry_price,
            }
            return record

        return None

    def get_summary(self, date_str: str = "") -> SessionSummary:
        entries = [r for r in self.trade_log if r.action == "entry"]
        exits = [r for r in self.trade_log if r.action == "exit"]
        skips = [r for r in self.trade_log if r.action == "skip"]
        return SessionSummary(
            date=date_str or datetime.now(ET).strftime("%Y-%m-%d"),
            regime=self.session_mgr.regime.regime.value if self.session_mgr.regime else "unknown",
            candidates_scanned=self.scan_result.total_scanned if self.scan_result else 0,
            candidates_passed=self.scan_result.total_passed if self.scan_result else 0,
            trades_entered=len(entries), trades_exited=len(exits),
            trades_skipped=len(skips), trade_log=self.trade_log,
        )

    def reset(self) -> None:
        self.session_mgr.reset()
        self.scan_result = None
        self.active_tickers = []
        self.positions.clear()
        self.trade_log.clear()

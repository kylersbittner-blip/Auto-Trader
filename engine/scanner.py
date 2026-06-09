"""
Pre-Market Scanner — Stocks in Play.

Identifies the best ORB candidates before market open by scoring
tickers against the criteria from Zarattini, Barbon & Aziz (2024):
  - Pre-market volume > 100K shares
  - Average daily volume > 500K shares (liquidity floor)
  - ATR(14) > $0.50 (enough range to trade)
  - Pre-market gap >= 2% (catalyst-driven attention)
  - News catalyst present
  - Relative volume >= 2x average

Protocol-based data interfaces for testability. In production,
plug in Alpaca for market data and Polygon for news.
"""
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
import math


@runtime_checkable
class MarketDataSource(Protocol):
    def get_premarket_volume(self, ticker: str) -> float: ...
    def get_avg_daily_volume(self, ticker: str, lookback_days: int = 20) -> float: ...
    def get_prev_close(self, ticker: str) -> float: ...
    def get_premarket_price(self, ticker: str) -> float: ...
    def get_atr(self, ticker: str, period: int = 14) -> float: ...


@runtime_checkable
class NewsSource(Protocol):
    def has_catalyst(self, ticker: str) -> bool: ...
    def get_headline(self, ticker: str) -> Optional[str]: ...


@dataclass
class ScannerConfig:
    min_premarket_volume: float = 100_000
    min_avg_daily_volume: float = 500_000
    min_atr: float = 0.50
    min_gap_pct: float = 0.02
    max_gap_pct: float = 0.04
    min_relative_volume: float = 2.0
    max_candidates: int = 5

    weight_rvol: float = 0.40
    weight_gap: float = 0.25
    weight_atr: float = 0.20
    weight_news: float = 0.15


@dataclass
class StockCandidate:
    ticker: str
    score: float
    rank: int = 0

    premarket_volume: float = 0.0
    avg_daily_volume: float = 0.0
    relative_volume: float = 0.0
    gap_pct: float = 0.0
    atr: float = 0.0
    premarket_price: float = 0.0
    prev_close: float = 0.0
    has_news: bool = False
    news_headline: Optional[str] = None

    passed_filters: list[str] = field(default_factory=list)
    failed_filters: list[str] = field(default_factory=list)

    @property
    def passed_all(self) -> bool:
        return len(self.failed_filters) == 0


@dataclass
class ScanResult:
    candidates: list[StockCandidate]
    total_scanned: int
    total_passed: int
    total_failed: int


class StocksInPlayScanner:
    def __init__(
        self,
        data_source: MarketDataSource,
        news_source: NewsSource,
        config: Optional[ScannerConfig] = None,
    ):
        self.data = data_source
        self.news = news_source
        self.config = config or ScannerConfig()

    def scan(self, universe: list[str]) -> ScanResult:
        if not universe:
            return ScanResult(candidates=[], total_scanned=0, total_passed=0, total_failed=0)

        all_candidates = []

        for ticker in universe:
            candidate = self._evaluate_ticker(ticker)
            all_candidates.append(candidate)

        passed = [c for c in all_candidates if c.passed_all]
        failed = [c for c in all_candidates if not c.passed_all]

        for c in passed:
            c.score = self._compute_score(c)

        passed.sort(key=lambda c: c.score, reverse=True)

        top = passed[: self.config.max_candidates]
        for i, c in enumerate(top):
            c.rank = i + 1

        return ScanResult(
            candidates=top,
            total_scanned=len(universe),
            total_passed=len(passed),
            total_failed=len(failed),
        )

    def _evaluate_ticker(self, ticker: str) -> StockCandidate:
        passed = []
        failed = []

        try:
            pm_vol = self.data.get_premarket_volume(ticker)
            adv = self.data.get_avg_daily_volume(ticker)
            prev_close = self.data.get_prev_close(ticker)
            pm_price = self.data.get_premarket_price(ticker)
            atr = self.data.get_atr(ticker)
            has_news = self.news.has_catalyst(ticker)
            headline = self.news.get_headline(ticker)
        except Exception as e:
            return StockCandidate(
                ticker=ticker, score=0.0,
                failed_filters=[f"Data fetch error: {e}"],
            )

        rvol = pm_vol / adv if adv > 0 else 0.0
        gap_pct = (pm_price - prev_close) / prev_close if prev_close > 0 else 0.0

        if pm_vol >= self.config.min_premarket_volume:
            passed.append(f"PM volume {pm_vol:,.0f} >= {self.config.min_premarket_volume:,.0f}")
        else:
            failed.append(f"PM volume {pm_vol:,.0f} < {self.config.min_premarket_volume:,.0f}")

        if adv >= self.config.min_avg_daily_volume:
            passed.append(f"ADV {adv:,.0f} >= {self.config.min_avg_daily_volume:,.0f}")
        else:
            failed.append(f"ADV {adv:,.0f} < {self.config.min_avg_daily_volume:,.0f}")

        if atr >= self.config.min_atr:
            passed.append(f"ATR ${atr:.2f} >= ${self.config.min_atr:.2f}")
        else:
            failed.append(f"ATR ${atr:.2f} < ${self.config.min_atr:.2f}")

        abs_gap = abs(gap_pct)
        if abs_gap >= self.config.min_gap_pct:
            if abs_gap <= self.config.max_gap_pct:
                passed.append(f"Gap {gap_pct:+.2%} in range [{self.config.min_gap_pct:.0%}, {self.config.max_gap_pct:.0%}]")
            else:
                failed.append(f"Gap {gap_pct:+.2%} exceeds {self.config.max_gap_pct:.0%} — too large for ORB")
        else:
            failed.append(f"Gap {gap_pct:+.2%} below {self.config.min_gap_pct:.0%} minimum")

        if has_news:
            passed.append(f"News catalyst: {headline or 'yes'}")
        else:
            failed.append("No news catalyst found")

        if rvol >= self.config.min_relative_volume:
            passed.append(f"RVOL {rvol:.1f}x >= {self.config.min_relative_volume:.1f}x")
        else:
            failed.append(f"RVOL {rvol:.1f}x < {self.config.min_relative_volume:.1f}x")

        return StockCandidate(
            ticker=ticker,
            score=0.0,
            premarket_volume=pm_vol,
            avg_daily_volume=adv,
            relative_volume=round(rvol, 2),
            gap_pct=round(gap_pct, 6),
            atr=atr,
            premarket_price=pm_price,
            prev_close=prev_close,
            has_news=has_news,
            news_headline=headline,
            passed_filters=passed,
            failed_filters=failed,
        )

    def _compute_score(self, candidate: StockCandidate) -> float:
        cfg = self.config

        rvol_raw = candidate.relative_volume
        if rvol_raw <= cfg.min_relative_volume:
            rvol_score = 0.0
        else:
            rvol_score = min(
                math.log(rvol_raw / cfg.min_relative_volume)
                / math.log(10 / cfg.min_relative_volume),
                1.0,
            )

        abs_gap = abs(candidate.gap_pct)
        if abs_gap <= 0.02:
            gap_score = 0.3
        elif abs_gap <= 0.03:
            gap_score = 0.3 + 0.5 * ((abs_gap - 0.02) / 0.01)
        elif abs_gap <= 0.035:
            gap_score = 0.8 + 0.2 * ((abs_gap - 0.03) / 0.005)
        elif abs_gap <= 0.04:
            gap_score = 1.0 - 0.3 * ((abs_gap - 0.035) / 0.005)
        else:
            gap_score = 0.5

        price = candidate.premarket_price if candidate.premarket_price > 0 else candidate.prev_close
        if price > 0:
            atr_pct = candidate.atr / price
            atr_score = min(atr_pct / 0.03, 1.0)
        else:
            atr_score = 0.0

        news_score = 1.0 if candidate.has_news else 0.0

        composite = (
            cfg.weight_rvol * rvol_score
            + cfg.weight_gap * gap_score
            + cfg.weight_atr * atr_score
            + cfg.weight_news * news_score
        )

        return round(composite * 100, 1)

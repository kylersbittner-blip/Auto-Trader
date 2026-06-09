"""
Phase 4 tests — Pre-Market Scanner (Stocks in Play).
Run with: pytest tests/test_phase4.py -v
"""
import pytest
import math
from typing import Optional

from engine.scanner import (
    StocksInPlayScanner,
    ScannerConfig,
    StockCandidate,
    ScanResult,
    MarketDataSource,
    NewsSource,
)


class MockMarketData:
    def __init__(self, stocks: dict[str, dict]):
        self.stocks = stocks

    def get_premarket_volume(self, ticker: str) -> float:
        return self.stocks[ticker]["pm_vol"]

    def get_avg_daily_volume(self, ticker: str, lookback_days: int = 20) -> float:
        return self.stocks[ticker]["adv"]

    def get_prev_close(self, ticker: str) -> float:
        return self.stocks[ticker]["prev_close"]

    def get_premarket_price(self, ticker: str) -> float:
        return self.stocks[ticker]["pm_price"]

    def get_atr(self, ticker: str, period: int = 14) -> float:
        return self.stocks[ticker]["atr"]


class MockNews:
    def __init__(self, catalysts: dict[str, Optional[str]]):
        self.catalysts = catalysts

    def has_catalyst(self, ticker: str) -> bool:
        return self.catalysts.get(ticker) is not None

    def get_headline(self, ticker: str) -> Optional[str]:
        return self.catalysts.get(ticker)


def perfect_stock() -> dict:
    return {
        "pm_vol": 2_500_000,
        "adv": 1_000_000,
        "prev_close": 100.0,
        "pm_price": 103.0,
        "atr": 2.50,
    }


def weak_stock() -> dict:
    return {
        "pm_vol": 50_000,
        "adv": 200_000,
        "prev_close": 10.0,
        "pm_price": 10.05,
        "atr": 0.20,
    }


def make_scanner(stocks, catalysts, config=None):
    return StocksInPlayScanner(
        data_source=MockMarketData(stocks),
        news_source=MockNews(catalysts),
        config=config,
    )


class TestProtocols:

    def test_mock_market_data_satisfies_protocol(self):
        mock = MockMarketData({"X": perfect_stock()})
        assert isinstance(mock, MarketDataSource)

    def test_mock_news_satisfies_protocol(self):
        mock = MockNews({"X": "headline"})
        assert isinstance(mock, NewsSource)


class TestFilters:

    def test_perfect_stock_passes_all_filters(self):
        scanner = make_scanner(
            {"NVDA": perfect_stock()},
            {"NVDA": "NVIDIA beats earnings"},
        )
        result = scanner.scan(["NVDA"])
        assert result.total_passed == 1
        assert result.total_failed == 0
        assert len(result.candidates) == 1
        assert result.candidates[0].passed_all is True

    def test_low_premarket_volume_fails(self):
        stock = perfect_stock()
        stock["pm_vol"] = 50_000
        scanner = make_scanner({"X": stock}, {"X": "news"})
        result = scanner.scan(["X"])
        assert result.total_passed == 0
        c = scanner._evaluate_ticker("X")
        assert any("PM volume" in f for f in c.failed_filters)

    def test_low_adv_fails(self):
        stock = perfect_stock()
        stock["adv"] = 200_000
        scanner = make_scanner({"X": stock}, {"X": "news"})
        result = scanner.scan(["X"])
        assert result.total_passed == 0
        c = scanner._evaluate_ticker("X")
        assert any("ADV" in f for f in c.failed_filters)

    def test_low_atr_fails(self):
        stock = perfect_stock()
        stock["atr"] = 0.30
        scanner = make_scanner({"X": stock}, {"X": "news"})
        result = scanner.scan(["X"])
        assert result.total_passed == 0
        c = scanner._evaluate_ticker("X")
        assert any("ATR" in f for f in c.failed_filters)

    def test_small_gap_fails(self):
        stock = perfect_stock()
        stock["pm_price"] = 100.50
        scanner = make_scanner({"X": stock}, {"X": "news"})
        result = scanner.scan(["X"])
        assert result.total_passed == 0
        c = scanner._evaluate_ticker("X")
        assert any("Gap" in f and "below" in f for f in c.failed_filters)

    def test_large_gap_fails(self):
        stock = perfect_stock()
        stock["pm_price"] = 105.50
        scanner = make_scanner({"X": stock}, {"X": "news"})
        result = scanner.scan(["X"])
        assert result.total_passed == 0
        c = scanner._evaluate_ticker("X")
        assert any("too large" in f for f in c.failed_filters)

    def test_no_news_fails(self):
        scanner = make_scanner({"X": perfect_stock()}, {"X": None})
        result = scanner.scan(["X"])
        assert result.total_passed == 0
        c = scanner._evaluate_ticker("X")
        assert any("No news" in f for f in c.failed_filters)

    def test_low_relative_volume_fails(self):
        stock = perfect_stock()
        stock["pm_vol"] = 150_000
        stock["adv"] = 1_000_000
        scanner = make_scanner({"X": stock}, {"X": "news"})
        result = scanner.scan(["X"])
        assert result.total_passed == 0
        c = scanner._evaluate_ticker("X")
        assert any("RVOL" in f for f in c.failed_filters)

    def test_gap_down_also_passes(self):
        stock = perfect_stock()
        stock["pm_price"] = 97.0
        scanner = make_scanner({"X": stock}, {"X": "news"})
        result = scanner.scan(["X"])
        assert result.total_passed == 1
        assert result.candidates[0].gap_pct < 0

    def test_data_fetch_error_handled(self):
        class BrokenData:
            def get_premarket_volume(self, t): raise ConnectionError("API down")
            def get_avg_daily_volume(self, t, **kw): return 0
            def get_prev_close(self, t): return 0
            def get_premarket_price(self, t): return 0
            def get_atr(self, t, **kw): return 0

        scanner = StocksInPlayScanner(BrokenData(), MockNews({"X": "news"}))
        result = scanner.scan(["X"])
        assert result.total_passed == 0
        assert result.total_failed == 1


class TestScoring:

    def test_higher_rvol_scores_higher(self):
        high_rvol = perfect_stock()
        high_rvol["pm_vol"] = 8_000_000
        high_rvol["adv"] = 1_000_000
        low_rvol = perfect_stock()
        low_rvol["pm_vol"] = 3_000_000
        low_rvol["adv"] = 1_000_000
        scanner = make_scanner(
            {"HIGH": high_rvol, "LOW": low_rvol},
            {"HIGH": "news", "LOW": "news"},
        )
        result = scanner.scan(["HIGH", "LOW"])
        assert len(result.candidates) == 2
        assert result.candidates[0].ticker == "HIGH"
        assert result.candidates[0].score > result.candidates[1].score

    def test_optimal_gap_scores_higher_than_edge_gap(self):
        optimal = perfect_stock()
        optimal["pm_price"] = 103.0
        optimal["pm_vol"] = 2_000_000
        optimal["adv"] = 1_000_000
        edge = perfect_stock()
        edge["pm_price"] = 102.0
        edge["pm_vol"] = 2_000_000
        edge["adv"] = 1_000_000
        scanner = make_scanner(
            {"OPT": optimal, "EDGE": edge},
            {"OPT": "news", "EDGE": "news"},
        )
        result = scanner.scan(["OPT", "EDGE"])
        assert len(result.candidates) == 2
        opt_c = next(c for c in result.candidates if c.ticker == "OPT")
        edge_c = next(c for c in result.candidates if c.ticker == "EDGE")
        assert opt_c.score > edge_c.score

    def test_score_between_0_and_100(self):
        scanner = make_scanner({"X": perfect_stock()}, {"X": "news"})
        result = scanner.scan(["X"])
        assert len(result.candidates) == 1
        assert 0 <= result.candidates[0].score <= 100

    def test_higher_atr_scores_higher(self):
        high_atr = perfect_stock()
        high_atr["atr"] = 4.00
        low_atr = perfect_stock()
        low_atr["atr"] = 0.80
        for s in [high_atr, low_atr]:
            s["pm_vol"] = 2_000_000
            s["adv"] = 1_000_000
            s["pm_price"] = 103.0
        scanner = make_scanner(
            {"HIGH_ATR": high_atr, "LOW_ATR": low_atr},
            {"HIGH_ATR": "news", "LOW_ATR": "news"},
        )
        result = scanner.scan(["HIGH_ATR", "LOW_ATR"])
        high_c = next(c for c in result.candidates if c.ticker == "HIGH_ATR")
        low_c = next(c for c in result.candidates if c.ticker == "LOW_ATR")
        assert high_c.score > low_c.score


class TestRanking:

    def test_candidates_ranked_by_score_descending(self):
        stocks = {}
        catalysts = {}
        for ticker, rvol_mult in [("A", 2.0), ("B", 5.0), ("C", 8.0), ("D", 3.0), ("E", 10.0)]:
            s = perfect_stock()
            s["pm_vol"] = int(1_000_000 * rvol_mult)
            s["adv"] = 1_000_000
            stocks[ticker] = s
            catalysts[ticker] = f"news for {ticker}"
        scanner = make_scanner(stocks, catalysts)
        result = scanner.scan(list(stocks.keys()))
        scores = [c.score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_rank_numbers_are_sequential(self):
        stocks = {}
        catalysts = {}
        for ticker in ["A", "B", "C"]:
            s = perfect_stock()
            s["pm_vol"] = 2_000_000
            s["adv"] = 1_000_000
            stocks[ticker] = s
            catalysts[ticker] = "news"
        scanner = make_scanner(stocks, catalysts)
        result = scanner.scan(list(stocks.keys()))
        ranks = [c.rank for c in result.candidates]
        assert ranks == [1, 2, 3]

    def test_max_candidates_respected(self):
        stocks = {}
        catalysts = {}
        for i in range(10):
            ticker = f"T{i}"
            s = perfect_stock()
            s["pm_vol"] = 2_000_000 + i * 100_000
            s["adv"] = 1_000_000
            stocks[ticker] = s
            catalysts[ticker] = "news"
        config = ScannerConfig(max_candidates=3)
        scanner = make_scanner(stocks, catalysts, config)
        result = scanner.scan(list(stocks.keys()))
        assert len(result.candidates) == 3
        assert result.total_passed == 10
        assert result.total_scanned == 10

    def test_mixed_pass_fail_universe(self):
        stocks = {
            "GOOD1": perfect_stock(),
            "GOOD2": perfect_stock(),
            "BAD": weak_stock(),
        }
        catalysts = {"GOOD1": "earnings beat", "GOOD2": "FDA approval", "BAD": "some news"}
        scanner = make_scanner(stocks, catalysts)
        result = scanner.scan(["GOOD1", "GOOD2", "BAD"])
        assert result.total_passed == 2
        assert result.total_failed == 1
        tickers = [c.ticker for c in result.candidates]
        assert "BAD" not in tickers


class TestScannerEdgeCases:

    def test_empty_universe(self):
        scanner = make_scanner({}, {})
        result = scanner.scan([])
        assert result.total_scanned == 0
        assert len(result.candidates) == 0

    def test_all_fail(self):
        scanner = make_scanner({"X": weak_stock()}, {"X": None})
        result = scanner.scan(["X"])
        assert result.total_passed == 0
        assert result.total_failed == 1
        assert len(result.candidates) == 0

    def test_zero_prev_close_handled(self):
        stock = perfect_stock()
        stock["prev_close"] = 0.0
        scanner = make_scanner({"X": stock}, {"X": "news"})
        result = scanner.scan(["X"])
        assert result.total_scanned == 1

    def test_custom_config_thresholds(self):
        stock = perfect_stock()
        stock["pm_vol"] = 80_000
        config = ScannerConfig(min_premarket_volume=50_000)
        scanner = make_scanner({"X": stock}, {"X": "news"}, config)
        c = scanner._evaluate_ticker("X")
        assert any("PM volume" in f for f in c.passed_filters)

    def test_scan_result_counts_accurate(self):
        stocks = {
            "A": perfect_stock(), "B": perfect_stock(),
            "C": weak_stock(), "D": weak_stock(),
            "E": perfect_stock(),
        }
        catalysts = {"A": "news", "B": "news", "C": None, "D": None, "E": "news"}
        scanner = make_scanner(stocks, catalysts)
        result = scanner.scan(["A", "B", "C", "D", "E"])
        assert result.total_scanned == 5
        assert result.total_passed == 3
        assert result.total_failed == 2

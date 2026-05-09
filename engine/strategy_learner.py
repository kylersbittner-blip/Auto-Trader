"""
Strategy Performance Learner.

Tracks actual win/loss outcomes per (ticker, strategy, regime) combination
and derives signal weight multipliers. The engine uses these multipliers to
boost strategies that are working and suppress ones that are not.

Storage: strategy_performance.json at project root.

Weight logic:
  - Fewer than MIN_SAMPLES trades → neutral weight (1.0)
  - win_rate > 60%  → boost up to 1.5×
  - win_rate < 40%  → suppress down to 0.5×
  - 40–60%          → linear interpolation around 1.0
"""
import json
from pathlib import Path
from typing import Optional
import structlog

log = structlog.get_logger()

PERF_FILE   = Path("strategy_performance.json")
MIN_SAMPLES = 10    # minimum trades before adjusting weight
WINDOW      = 50    # rolling window — only use the last N outcomes per key


def _load() -> dict:
    if not PERF_FILE.exists():
        return {}
    try:
        return json.loads(PERF_FILE.read_text())
    except Exception:
        return {}


def _save(data: dict) -> None:
    try:
        PERF_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.error("strategy_perf_save_failed", error=str(e))


def _key(strategy: str, regime: str) -> str:
    return f"{strategy}/{regime}"


# ── Public API ────────────────────────────────────────────────────────────────

def record_outcome(
    ticker:      str,
    strategy:    str,
    regime:      str,
    won:         bool,
    return_pct:  float,
) -> None:
    """Record a closed trade outcome."""
    data = _load()
    k    = _key(strategy, regime)
    if ticker not in data:
        data[ticker] = {}
    if k not in data[ticker]:
        data[ticker][k] = {"outcomes": []}

    # Append and keep only rolling window
    data[ticker][k]["outcomes"].append({
        "won":        won,
        "return_pct": round(return_pct, 4),
    })
    data[ticker][k]["outcomes"] = data[ticker][k]["outcomes"][-WINDOW:]
    _save(data)
    log.info("strategy_outcome_recorded", ticker=ticker, key=k, won=won)


def get_weight(ticker: str, strategy: str, regime: str) -> float:
    """
    Return a multiplier (0.5 – 1.5) to scale the technical score.
    Neutral = 1.0 when we lack sufficient data.
    """
    data     = _load()
    outcomes = data.get(ticker, {}).get(_key(strategy, regime), {}).get("outcomes", [])
    if len(outcomes) < MIN_SAMPLES:
        return 1.0

    win_rate = sum(1 for o in outcomes if o["won"]) / len(outcomes)
    # Linear: 0% → 0.5, 50% → 1.0, 100% → 1.5
    return round(max(0.5, min(1.5, win_rate * 2)), 3)


def get_best_strategy(ticker: str, regime: str) -> Optional[str]:
    """
    Return the strategy with the highest win rate for this (ticker, regime),
    or None if no strategy has MIN_SAMPLES trades yet.
    """
    data      = _load().get(ticker, {})
    strategies = ["momentum", "mean_reversion", "breakout"]
    best_wr   = -1.0
    best_strat: Optional[str] = None

    for strat in strategies:
        outcomes = data.get(_key(strat, regime), {}).get("outcomes", [])
        if len(outcomes) < MIN_SAMPLES:
            continue
        wr = sum(1 for o in outcomes if o["won"]) / len(outcomes)
        if wr > best_wr:
            best_wr   = wr
            best_strat = strat

    return best_strat


def get_summary() -> dict:
    """Full performance table for dashboard display."""
    data    = _load()
    summary = {}
    for ticker, keys in data.items():
        summary[ticker] = {}
        for k, v in keys.items():
            outcomes = v.get("outcomes", [])
            if not outcomes:
                continue
            wins = [o for o in outcomes if o["won"]]
            summary[ticker][k] = {
                "n_trades":   len(outcomes),
                "win_rate":   round(len(wins) / len(outcomes) * 100, 1),
                "avg_return": round(
                    sum(o["return_pct"] for o in outcomes) / len(outcomes) * 100, 3
                ),
                "weight":     get_weight(ticker, *k.split("/")),
            }
    return summary

"""
Trade outcome tracker.

Records every executed trade with the market context at entry time.
When enough new outcomes accumulate (RETRAIN_THRESHOLD), signals the engine
to trigger a fresh model retrain so the ML adapts to recent price behavior.

Storage: JSON file at project root (trade_outcomes.json).
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import structlog

log = structlog.get_logger()

OUTCOMES_FILE    = Path("trade_outcomes.json")
RETRAIN_THRESHOLD = 20   # retrain after this many new closed outcomes per ticker


# ── Internal I/O ──────────────────────────────────────────────────────────────

def _load() -> list[dict]:
    if not OUTCOMES_FILE.exists():
        return []
    try:
        return json.loads(OUTCOMES_FILE.read_text())
    except Exception:
        return []


def _save(outcomes: list[dict]) -> None:
    try:
        OUTCOMES_FILE.write_text(json.dumps(outcomes, indent=2, default=str))
    except Exception as e:
        log.error("trade_outcomes_save_failed", error=str(e))


# ── Public API ────────────────────────────────────────────────────────────────

def record_entry(
    ticker:     str,
    action:     str,
    price:      float,
    strategy:   str,
    regime:     str,
    broker_id:  Optional[str] = None,
    features:   Optional[dict] = None,
) -> str:
    """Record a trade entry. Returns a short trade_id for later exit recording."""
    outcomes = _load()
    trade_id = str(uuid.uuid4())[:8]
    outcomes.append({
        "id":               trade_id,
        "ticker":           ticker,
        "action":           action,
        "entry_price":      price,
        "entry_time":       datetime.now(timezone.utc).isoformat(),
        "strategy":         strategy,
        "regime":           regime,
        "broker_id":        broker_id,
        "features":         features or {},
        "exit_price":       None,
        "exit_time":        None,
        "actual_return":    None,
        "used_in_training": False,
    })
    _save(outcomes)
    log.info("trade_entry_recorded", trade_id=trade_id, ticker=ticker, action=action)
    return trade_id


def get_open_entries() -> list[dict]:
    """Return all entries that have no exit recorded yet."""
    return [o for o in _load() if o["actual_return"] is None]


def record_exit(trade_id: str, exit_price: float) -> Optional[float]:
    """
    Record a trade exit, compute the actual return, and persist.
    Returns the actual_return or None if trade_id not found.
    """
    outcomes = _load()
    for o in outcomes:
        if o["id"] == trade_id and o["exit_price"] is None:
            ret = (exit_price - o["entry_price"]) / o["entry_price"]
            if o["action"] == "sell":
                ret = -ret
            o["exit_price"]    = exit_price
            o["exit_time"]     = datetime.now(timezone.utc).isoformat()
            o["actual_return"] = round(ret, 6)
            _save(outcomes)
            log.info("trade_exit_recorded", trade_id=trade_id, actual_return=round(ret, 4))
            return ret
    return None


def should_retrain(ticker: str) -> bool:
    """True when there are enough new closed outcomes to justify retraining."""
    return _pending_count(ticker) >= RETRAIN_THRESHOLD


def mark_used_in_training(ticker: str) -> None:
    """Mark all current closed outcomes for this ticker as consumed."""
    outcomes = _load()
    for o in outcomes:
        if o["ticker"] == ticker and o["actual_return"] is not None:
            o["used_in_training"] = True
    _save(outcomes)


def get_summary(ticker: Optional[str] = None) -> dict:
    """Return win-rate, avg-return, and counts for monitoring."""
    outcomes = _load()
    if ticker:
        outcomes = [o for o in outcomes if o["ticker"] == ticker]
    closed = [o for o in outcomes if o["actual_return"] is not None]
    if not closed:
        return {"n_trades": 0, "win_rate": None, "avg_return": None}
    wins    = [o for o in closed if o["actual_return"] > 0]
    returns = [o["actual_return"] for o in closed]
    return {
        "n_trades":   len(closed),
        "win_rate":   round(len(wins) / len(closed) * 100, 1),
        "avg_return": round(sum(returns) / len(returns) * 100, 3),
    }


def _pending_count(ticker: str) -> int:
    return sum(
        1 for o in _load()
        if o["ticker"] == ticker
        and o["actual_return"] is not None
        and not o["used_in_training"]
    )

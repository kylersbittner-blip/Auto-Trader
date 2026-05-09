"""
Dynamic position sizer using half-Kelly criterion.

Kelly formula: f* = (b·p - q) / b
  b = avg_win / avg_loss  (odds ratio)
  p = win rate
  q = 1 - p

We use half-Kelly (f*/2) to reduce variance while still capturing the edge.
Falls back to fixed sizing (max_position_usd) when trade history is thin.

Hard caps:
  - Never exceed MAX_POSITION_USD from config
  - Never allocate more than 20% of account equity to a single trade
  - Minimum position: $500
"""
from typing import Optional
import structlog

log = structlog.get_logger()

MIN_TRADES_FOR_KELLY = 15
MIN_POSITION_USD     = 500.0
MAX_ACCOUNT_PCT      = 0.20   # cap at 20% of account per trade


def kelly_size(
    ticker:          str,
    strategy:        str,
    regime:          str,
    account_equity:  float,
    max_position:    float,
    confidence:      float,   # 0–100 from signal
) -> float:
    """
    Return optimal position size in USD.
    Falls back to confidence-scaled fixed size if Kelly data is unavailable.
    """
    from engine.strategy_learner import _load, _key

    data     = _load()
    outcomes = data.get(ticker, {}).get(_key(strategy, regime), {}).get("outcomes", [])

    hard_cap = min(max_position, account_equity * MAX_ACCOUNT_PCT)

    if len(outcomes) < MIN_TRADES_FOR_KELLY:
        # Not enough data — scale by confidence
        conf_scale = max(0.3, (confidence - 50) / 50) if confidence > 50 else 0.3
        size = hard_cap * conf_scale
        return max(MIN_POSITION_USD, round(size, 2))

    wins   = [o for o in outcomes if o["won"]]
    losses = [o for o in outcomes if not o["won"]]
    p      = len(wins) / len(outcomes)
    q      = 1 - p

    avg_win  = sum(abs(o["return_pct"]) for o in wins)  / max(len(wins), 1)
    avg_loss = sum(abs(o["return_pct"]) for o in losses) / max(len(losses), 1)

    if avg_loss < 1e-6:
        return hard_cap  # no losses recorded — full size

    b = avg_win / avg_loss
    f = (b * p - q) / b   # full Kelly fraction
    half_f = max(0.0, f / 2)   # half-Kelly

    size = account_equity * half_f
    size = max(MIN_POSITION_USD, min(hard_cap, round(size, 2)))

    log.debug(
        "kelly_size",
        ticker=ticker, p=round(p, 2), b=round(b, 2),
        half_f=round(half_f, 3), size=size,
    )
    return size

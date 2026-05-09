"""
Market hours filter.

Only allows trading during high-liquidity windows in ET:
  - Morning session: 09:35 – 11:30  (momentum, high volume)
  - Afternoon session: 13:30 – 15:50 (institutional activity resumes)

Avoids:
  - First 5 min (09:30–09:35): wild open prints, wide spreads
  - Lunch (11:30–13:30): low volume, choppy / mean-reverting
  - Last 10 min (15:50–16:00): closing auction manipulation
  - Weekends and US market holidays
"""
from datetime import datetime, time, timezone
import zoneinfo

ET = zoneinfo.ZoneInfo("America/New_York")

MORNING_OPEN  = time(9,  35)
MORNING_CLOSE = time(11, 30)
AFTERNOON_OPEN  = time(13, 30)
AFTERNOON_CLOSE = time(15, 50)


def is_trading_window(dt: datetime | None = None) -> tuple[bool, str]:
    """
    Returns (allowed, reason).
    Pass dt for testing; defaults to now.
    """
    now_et = (dt or datetime.now(timezone.utc)).astimezone(ET)

    if now_et.weekday() >= 5:
        return False, "weekend"

    t = now_et.time()

    if MORNING_OPEN <= t <= MORNING_CLOSE:
        return True, "morning session"
    if AFTERNOON_OPEN <= t <= AFTERNOON_CLOSE:
        return True, "afternoon session"
    if t < MORNING_OPEN:
        return False, "pre-market"
    if MORNING_CLOSE < t < AFTERNOON_OPEN:
        return False, "lunch lull — low liquidity"
    return False, "post-market / closing auction"

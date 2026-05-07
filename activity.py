"""
Activity logger — tracks successes, failures, and warnings across the engine.
Lives at the top level to avoid circular imports between api/ and engine/.
"""
from collections import deque
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class ActivityEntry(BaseModel):
    id: int
    timestamp: datetime
    type: Literal["success", "failure", "warning", "info"]
    category: str   # scan | trade | engine | data | risk | signal
    message: str
    detail: Optional[str] = None
    ticker: Optional[str] = None


class ActivityLogger:
    def __init__(self, max_entries: int = 500):
        self._log: deque[ActivityEntry] = deque(maxlen=max_entries)
        self._counter = 0

    def _add(
        self,
        type_: str,
        category: str,
        message: str,
        detail: str | None = None,
        ticker: str | None = None,
    ) -> ActivityEntry:
        self._counter += 1
        entry = ActivityEntry(
            id=self._counter,
            timestamp=datetime.utcnow(),
            type=type_,
            category=category,
            message=message,
            detail=detail,
            ticker=ticker,
        )
        self._log.appendleft(entry)
        return entry

    def success(self, category: str, message: str, detail: str | None = None, ticker: str | None = None):
        return self._add("success", category, message, detail, ticker)

    def failure(self, category: str, message: str, detail: str | None = None, ticker: str | None = None):
        return self._add("failure", category, message, detail, ticker)

    def warning(self, category: str, message: str, detail: str | None = None, ticker: str | None = None):
        return self._add("warning", category, message, detail, ticker)

    def info(self, category: str, message: str, detail: str | None = None, ticker: str | None = None):
        return self._add("info", category, message, detail, ticker)

    def get_recent(
        self,
        limit: int = 100,
        type_filter: str | None = None,
        category_filter: str | None = None,
    ) -> list[ActivityEntry]:
        entries = list(self._log)
        if type_filter:
            entries = [e for e in entries if e.type == type_filter]
        if category_filter:
            entries = [e for e in entries if e.category == category_filter]
        return entries[:limit]

    def get_stats(self) -> dict:
        entries = list(self._log)
        return {
            "total": len(entries),
            "successes": sum(1 for e in entries if e.type == "success"),
            "failures": sum(1 for e in entries if e.type == "failure"),
            "warnings": sum(1 for e in entries if e.type == "warning"),
            "infos": sum(1 for e in entries if e.type == "info"),
        }


_logger: ActivityLogger | None = None


def get_activity_logger() -> ActivityLogger:
    global _logger
    if _logger is None:
        _logger = ActivityLogger()
    return _logger

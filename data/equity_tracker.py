"""
Equity curve tracker.

Snapshots account equity every scan and persists to equity_curve.json.
Keeps the last 390 snapshots (one full trading day at 1-min resolution,
or ~5 days at 60-second engine intervals).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

EQUITY_FILE   = Path("equity_curve.json")
MAX_SNAPSHOTS = 500


def record_snapshot(equity: float, cash: float) -> None:
    data = _load()
    data.append({
        "t": datetime.now(timezone.utc).isoformat(),
        "equity": round(equity, 2),
        "cash":   round(cash, 2),
    })
    data = data[-MAX_SNAPSHOTS:]
    _save(data)


def get_curve() -> list[dict]:
    return _load()


def _load() -> list:
    if not EQUITY_FILE.exists():
        return []
    try:
        return json.loads(EQUITY_FILE.read_text())
    except Exception:
        return []


def _save(data: list) -> None:
    try:
        EQUITY_FILE.write_text(json.dumps(data))
    except Exception:
        pass

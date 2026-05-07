from fastapi import APIRouter, Query
from activity import get_activity_logger

router = APIRouter()


@router.get("")
async def get_activity(
    limit: int = Query(100, ge=1, le=500),
    type: str | None = Query(None, description="Filter: success | failure | warning | info"),
    category: str | None = Query(None, description="Filter: scan | trade | engine | data | risk"),
):
    """Return recent activity log entries, newest first."""
    logger = get_activity_logger()
    entries = logger.get_recent(limit=limit, type_filter=type, category_filter=category)
    return [e.model_dump(mode="json") for e in entries]


@router.get("/stats")
async def get_activity_stats():
    """Return counts of successes, failures, warnings, and infos."""
    return get_activity_logger().get_stats()

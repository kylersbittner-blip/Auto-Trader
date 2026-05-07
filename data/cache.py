"""
Redis helpers for signal caching and pub/sub.
"""
import json
from typing import Any, Optional
import redis.asyncio as aioredis
import structlog

from config import get_settings

log = structlog.get_logger()
settings = get_settings()

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def cache_set(key: str, value: Any, ttl: int = 300) -> None:
    r = await get_redis()
    if not isinstance(value, str):
        value = json.dumps(value)
    await r.setex(key, ttl, value)


async def cache_get(key: str) -> Optional[str]:
    r = await get_redis()
    return await r.get(key)


async def cache_delete(key: str) -> None:
    r = await get_redis()
    await r.delete(key)


async def publish_signal(signal_dict: dict) -> None:
    """Publish a signal to the 'signals' channel for WebSocket broadcast."""
    r = await get_redis()
    await r.publish("signals", json.dumps(signal_dict))


async def get_engine_state() -> dict:
    r = await get_redis()
    raw = await r.get("engine:state")
    return json.loads(raw) if raw else {"running": False}


async def set_engine_state(state: dict) -> None:
    r = await get_redis()
    await r.set("engine:state", json.dumps(state))

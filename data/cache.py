"""
Redis helpers for signal caching and pub/sub.
All operations fail silently if Redis is unavailable — the bot runs without it.
"""
import json
from typing import Any, Optional
import redis.asyncio as aioredis
import structlog

from config import get_settings

log = structlog.get_logger()
settings = get_settings()

_redis: Optional[aioredis.Redis] = None
_redis_available: bool = True


async def get_redis() -> Optional[aioredis.Redis]:
    global _redis, _redis_available
    if not _redis_available:
        return None
    try:
        if _redis is None:
            _redis = await aioredis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2)
            await _redis.ping()
    except Exception:
        _redis_available = False
        _redis = None
        log.warning("redis_unavailable", url=settings.redis_url, note="running without cache")
        return None
    return _redis


async def cache_set(key: str, value: Any, ttl: int = 300) -> None:
    r = await get_redis()
    if r is None:
        return
    try:
        if not isinstance(value, str):
            value = json.dumps(value)
        await r.setex(key, ttl, value)
    except Exception:
        pass


async def cache_get(key: str) -> Optional[str]:
    r = await get_redis()
    if r is None:
        return None
    try:
        return await r.get(key)
    except Exception:
        return None


async def cache_delete(key: str) -> None:
    r = await get_redis()
    if r is None:
        return
    try:
        await r.delete(key)
    except Exception:
        pass


async def publish_signal(signal_dict: dict) -> None:
    r = await get_redis()
    if r is None:
        return
    try:
        await r.publish("signals", json.dumps(signal_dict))
    except Exception:
        pass


async def get_engine_state() -> dict:
    r = await get_redis()
    if r is None:
        return {"running": False}
    try:
        raw = await r.get("engine:state")
        return json.loads(raw) if raw else {"running": False}
    except Exception:
        return {"running": False}


async def set_engine_state(state: dict) -> None:
    r = await get_redis()
    if r is None:
        return
    try:
        await r.set("engine:state", json.dumps(state))
    except Exception:
        pass

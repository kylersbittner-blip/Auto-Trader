"""
WebSocket endpoint — streams signals in real-time via Redis pub/sub.
Connect at: ws://localhost:8000/ws/live

Each message is a JSON Signal object emitted whenever the engine
completes a scan and publishes to the 'signals' Redis channel.
"""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
import structlog

from config import get_settings

log = structlog.get_logger()
router = APIRouter()
settings = get_settings()


@router.websocket("/ws/live")
async def live_signals(websocket: WebSocket):
    await websocket.accept()
    log.info("ws_client_connected", client=websocket.client)

    redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe("signals")

    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    await websocket.send_json(data)
                except Exception as e:
                    log.warning("ws_send_error", error=str(e))
    except WebSocketDisconnect:
        log.info("ws_client_disconnected")
    finally:
        await pubsub.unsubscribe("signals")
        await redis.aclose()

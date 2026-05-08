"""
AutoTrader Pro — FastAPI application entry point.
"""
from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from api.routes import signals, portfolio, trades, news, engine as engine_routes
from api.routes import activity as activity_routes
from api.routes import training as training_routes
from api.websocket import router as ws_router

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("autotrader_starting")
    # Engine does NOT auto-start — user starts it via POST /engine/start
    yield
    log.info("autotrader_shutdown")
    from engine.signal_engine import get_engine
    eng = get_engine()
    if eng.running:
        await eng.stop()


app = FastAPI(
    title="AutoTrader Pro API",
    description="AI-powered day trading engine — signals, execution, portfolio.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router,        prefix="/signals",   tags=["Signals"])
app.include_router(portfolio.router,      prefix="/portfolio", tags=["Portfolio"])
app.include_router(trades.router,         prefix="/trades",    tags=["Trades"])
app.include_router(news.router,           prefix="/news",      tags=["News"])
app.include_router(engine_routes.router,  prefix="/engine",    tags=["Engine"])
app.include_router(activity_routes.router,  prefix="/activity", tags=["Activity"])
app.include_router(training_routes.router,  prefix="/train",    tags=["Training"])
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")
    return FileResponse(path)

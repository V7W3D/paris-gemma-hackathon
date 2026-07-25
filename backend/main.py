from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.dependencies import AppContainer
from backend.routers import chats, verify

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container = AppContainer(get_settings())
    app.state.container = container
    await container.startup()
    logger.info("Claim verifier ready: %s", container.status())
    try:
        yield
    finally:
        await container.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Claim Verifier",
        version="0.1.0",
        description="Dual-agent claim verification with context engineering.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chats.router, prefix="/api")
    app.include_router(verify.router, prefix="/api")

    @app.get("/api/status", tags=["system"])
    async def status() -> dict[str, object]:
        return app.state.container.status()

    return app


app = create_app()

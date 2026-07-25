from __future__ import annotations

import logging

from backend.config import Settings
from backend.db.store import MemoryStore, MongoStore, Store

logger = logging.getLogger(__name__)


class Database:
    """Owns the Mongo connection and hands out the active store.

    If the server is unreachable at startup we degrade to an in-memory store
    instead of failing the whole app, which keeps the demo usable offline.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        self._store: Store = MemoryStore()
        self.connected = False

    @property
    def store(self) -> Store:
        return self._store

    async def connect(self) -> Store:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            client = AsyncIOMotorClient(
                self._settings.mongodb_uri,
                serverSelectionTimeoutMS=self._settings.mongodb_timeout_ms,
                uuidRepresentation="standard",
            )
            await client.admin.command("ping")
            store = MongoStore(client[self._settings.mongodb_db])
            await store.ensure_indexes()
        except Exception as exc:  # noqa: BLE001 - any driver/network failure degrades the same way
            logger.warning(
                "MongoDB unavailable at %s (%s); falling back to in-memory storage",
                self._settings.mongodb_uri,
                exc,
            )
            self._store = MemoryStore()
            self.connected = False
            return self._store

        self._client = client
        self._store = store
        self.connected = True
        logger.info("Connected to MongoDB database %s", self._settings.mongodb_db)
        return self._store

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self.connected = False

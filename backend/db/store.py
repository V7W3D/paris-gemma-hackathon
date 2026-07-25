from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from backend.models.schemas.chat import Conversation, ConversationSummary, Message
from backend.models.schemas.context import ContextObject


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _summary(conversation: Conversation) -> ConversationSummary:
    return ConversationSummary(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(conversation.messages),
    )


class Store(ABC):
    """Persistence for conversations and the context objects agent 2 writes."""

    @abstractmethod
    async def list_conversations(self) -> list[ConversationSummary]: ...

    @abstractmethod
    async def create_conversation(self, title: str | None = None) -> Conversation: ...

    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Conversation | None: ...

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> bool: ...

    @abstractmethod
    async def rename_conversation(self, conversation_id: str, title: str) -> Conversation | None: ...

    @abstractmethod
    async def append_message(self, conversation_id: str, message: Message) -> Conversation | None: ...

    @abstractmethod
    async def save_context(self, context: ContextObject) -> None: ...

    @abstractmethod
    async def latest_context(self, conversation_id: str) -> ContextObject | None: ...

    @abstractmethod
    async def list_contexts(self, conversation_id: str) -> list[ContextObject]: ...


class MemoryStore(Store):
    """In-process fallback so the app and tests run without a MongoDB server."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._contexts: dict[str, list[ContextObject]] = {}

    async def list_conversations(self) -> list[ConversationSummary]:
        ordered = sorted(
            self._conversations.values(), key=lambda c: c.updated_at, reverse=True
        )
        return [_summary(c) for c in ordered]

    async def create_conversation(self, title: str | None = None) -> Conversation:
        conversation = Conversation(title=title or "New verification")
        self._conversations[conversation.id] = conversation
        return conversation.model_copy(deep=True)

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        conversation = self._conversations.get(conversation_id)
        return conversation.model_copy(deep=True) if conversation else None

    async def delete_conversation(self, conversation_id: str) -> bool:
        self._contexts.pop(conversation_id, None)
        return self._conversations.pop(conversation_id, None) is not None

    async def rename_conversation(self, conversation_id: str, title: str) -> Conversation | None:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return None
        conversation.title = title
        conversation.updated_at = _now()
        return conversation.model_copy(deep=True)

    async def append_message(self, conversation_id: str, message: Message) -> Conversation | None:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return None
        conversation.messages.append(message.model_copy(deep=True))
        conversation.updated_at = _now()
        return conversation.model_copy(deep=True)

    async def save_context(self, context: ContextObject) -> None:
        self._contexts.setdefault(context.conversation_id, []).append(
            context.model_copy(deep=True)
        )

    async def latest_context(self, conversation_id: str) -> ContextObject | None:
        contexts = self._contexts.get(conversation_id)
        return contexts[-1].model_copy(deep=True) if contexts else None

    async def list_contexts(self, conversation_id: str) -> list[ContextObject]:
        return [c.model_copy(deep=True) for c in self._contexts.get(conversation_id, [])]


class MongoStore(Store):
    """MongoDB-backed store. Documents are stored as JSON-mode dumps so that
    dates round-trip as sortable ISO strings and enums as plain values."""

    def __init__(self, database: Any) -> None:
        self.conversations = database["conversations"]
        self.contexts = database["contexts"]

    async def ensure_indexes(self) -> None:
        await self.conversations.create_index("updated_at")
        await self.contexts.create_index([("conversation_id", 1), ("created_at", 1)])

    @staticmethod
    def _clean(document: dict[str, Any]) -> dict[str, Any]:
        document = copy.deepcopy(document)
        document.pop("_id", None)
        return document

    async def list_conversations(self) -> list[ConversationSummary]:
        cursor = self.conversations.find(
            {}, {"messages": 0}
        ).sort("updated_at", -1)
        summaries: list[ConversationSummary] = []
        async for document in cursor:
            document = self._clean(document)
            summaries.append(
                ConversationSummary(
                    id=document["id"],
                    title=document.get("title", "New verification"),
                    created_at=document["created_at"],
                    updated_at=document["updated_at"],
                    message_count=document.get("message_count", 0),
                )
            )
        return summaries

    async def create_conversation(self, title: str | None = None) -> Conversation:
        conversation = Conversation(title=title or "New verification")
        document = conversation.model_dump(mode="json")
        document["message_count"] = 0
        document["_id"] = conversation.id
        await self.conversations.insert_one(document)
        return conversation

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        document = await self.conversations.find_one({"id": conversation_id})
        return Conversation.model_validate(self._clean(document)) if document else None

    async def delete_conversation(self, conversation_id: str) -> bool:
        await self.contexts.delete_many({"conversation_id": conversation_id})
        result = await self.conversations.delete_one({"id": conversation_id})
        return result.deleted_count > 0

    async def rename_conversation(self, conversation_id: str, title: str) -> Conversation | None:
        await self.conversations.update_one(
            {"id": conversation_id},
            {"$set": {"title": title, "updated_at": _now().isoformat()}},
        )
        return await self.get_conversation(conversation_id)

    async def append_message(self, conversation_id: str, message: Message) -> Conversation | None:
        result = await self.conversations.update_one(
            {"id": conversation_id},
            {
                "$push": {"messages": message.model_dump(mode="json")},
                "$inc": {"message_count": 1},
                "$set": {"updated_at": _now().isoformat()},
            },
        )
        if result.matched_count == 0:
            return None
        return await self.get_conversation(conversation_id)

    async def save_context(self, context: ContextObject) -> None:
        document = context.model_dump(mode="json")
        document["_id"] = context.id
        await self.contexts.replace_one({"_id": context.id}, document, upsert=True)

    async def latest_context(self, conversation_id: str) -> ContextObject | None:
        cursor = (
            self.contexts.find({"conversation_id": conversation_id})
            .sort([("created_at", -1), ("revision", -1)])
            .limit(1)
        )
        async for document in cursor:
            return ContextObject.model_validate(self._clean(document))
        return None

    async def list_contexts(self, conversation_id: str) -> list[ContextObject]:
        cursor = self.contexts.find({"conversation_id": conversation_id}).sort(
            [("created_at", 1), ("revision", 1)]
        )
        return [ContextObject.model_validate(self._clean(d)) async for d in cursor]

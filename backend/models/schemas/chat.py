from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.models.schemas.context import Stage, Verdict


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return uuid4().hex[:12]


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class TraceStep(BaseModel):
    """One decision point, flattened for display in the frontend trace panel."""

    stage: Stage
    summary: str = ""
    retrievals: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class Message(BaseModel):
    id: str = Field(default_factory=_uid)
    role: Role
    content: str
    created_at: datetime = Field(default_factory=_now)
    turn_id: str | None = None
    verdict: Verdict | None = None
    trace: list[TraceStep] = Field(default_factory=list)


class Conversation(BaseModel):
    id: str = Field(default_factory=_uid)
    title: str = "New verification"
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class CreateConversationRequest(BaseModel):
    title: str | None = None


class RenameConversationRequest(BaseModel):
    title: str


class VerifyRequest(BaseModel):
    content: str
    use_verifier: bool = Field(
        default=True,
        description="False answers in one plain model call, with no context, sources or verdict.",
    )

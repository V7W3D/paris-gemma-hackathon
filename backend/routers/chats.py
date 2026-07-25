from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import AppContainer, get_container
from backend.models.schemas.chat import (
    Conversation,
    ConversationSummary,
    CreateConversationRequest,
    RenameConversationRequest,
)
from backend.models.schemas.context import ContextObject

router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("", response_model=list[ConversationSummary])
async def list_chats(container: AppContainer = Depends(get_container)) -> list[ConversationSummary]:
    return await container.store.list_conversations()


@router.post("", response_model=Conversation, status_code=status.HTTP_201_CREATED)
async def create_chat(
    payload: CreateConversationRequest | None = None,
    container: AppContainer = Depends(get_container),
) -> Conversation:
    return await container.store.create_conversation(payload.title if payload else None)


@router.get("/{chat_id}", response_model=Conversation)
async def get_chat(chat_id: str, container: AppContainer = Depends(get_container)) -> Conversation:
    conversation = await container.store.get_conversation(chat_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conversation


@router.patch("/{chat_id}", response_model=Conversation)
async def rename_chat(
    chat_id: str,
    payload: RenameConversationRequest,
    container: AppContainer = Depends(get_container),
) -> Conversation:
    conversation = await container.store.rename_conversation(chat_id, payload.title.strip())
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conversation


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: str, container: AppContainer = Depends(get_container)) -> None:
    if not await container.store.delete_conversation(chat_id):
        raise HTTPException(status_code=404, detail="conversation not found")


@router.get("/{chat_id}/contexts", response_model=list[ContextObject])
async def list_chat_contexts(
    chat_id: str, container: AppContainer = Depends(get_container)
) -> list[ContextObject]:
    """Every context revision agent 2 wrote for this conversation."""
    if await container.store.get_conversation(chat_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return await container.context_agent.list_contexts(chat_id)

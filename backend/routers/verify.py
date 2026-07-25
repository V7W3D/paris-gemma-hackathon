from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.dependencies import AppContainer, get_container
from backend.models.schemas.chat import VerifyRequest
from backend.services.workflow.claim_verification import ConversationNotFound

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chats", tags=["verify"])

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/{chat_id}/messages")
async def post_message(
    chat_id: str,
    payload: VerifyRequest,
    stream: bool = Query(default=True, description="Stream decision points as SSE"),
    container: AppContainer = Depends(get_container),
) -> Any:
    """Run one turn: the five decision points over both agents, or a direct answer."""
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="message content is empty")
    if await container.store.get_conversation(chat_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    events = container.workflow.run(
        conversation_id=chat_id, content=content, use_verifier=payload.use_verifier
    )

    if not stream:
        collected: list[dict[str, Any]] = []
        try:
            async for event in events:
                collected.append(event)
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="conversation not found") from None
        message = next(
            (e["message"] for e in reversed(collected) if e["type"] == "message"), None
        )
        return {"message": message, "events": [e for e in collected if e["type"] != "token"]}

    return StreamingResponse(
        _to_sse(events), media_type="text/event-stream", headers=SSE_HEADERS
    )


async def _to_sse(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    try:
        async for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except ConversationNotFound:
        yield f"data: {json.dumps({'type': 'error', 'error': 'conversation not found'})}\n\n"
    except Exception as exc:  # noqa: BLE001 - the stream must close with an event, not a trace
        logger.exception("Streaming turn failed")
        yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

from __future__ import annotations

import pytest

from backend.db.store import MemoryStore
from backend.models.schemas.context import Stage
from backend.services.retrieval.alien_client import RetrievalError
from backend.services.workflow.claim_verification import (
    ClaimVerificationWorkflow,
    ConversationNotFound,
)

pytestmark = pytest.mark.asyncio

CLAIM = "The Eiffel Tower is 330 metres tall including its antennas."


async def _run(workflow: ClaimVerificationWorkflow, chat_id: str, content: str) -> list[dict]:
    return [event async for event in workflow.run(conversation_id=chat_id, content=content)]


async def test_a_claim_walks_every_decision_point(
    workflow: ClaimVerificationWorkflow, store: MemoryStore
):
    conversation = await store.create_conversation()
    events = await _run(workflow, conversation.id, CLAIM)

    assert [e["type"] for e in events][0] == "turn_started"
    assert events[-1]["type"] == "done"
    assert not [e for e in events if e["type"] == "error"]

    started = [e["stage"] for e in events if e["type"] == "stage" and e["status"] == "started"]
    assert set(started) == {stage.value for stage in Stage}
    assert started.index("decompose") < started.index("gather") < started.index("verdict")


async def test_the_turn_is_persisted_with_a_verdict_and_a_trace(
    workflow: ClaimVerificationWorkflow, store: MemoryStore
):
    conversation = await store.create_conversation()
    await _run(workflow, conversation.id, CLAIM)

    saved = await store.get_conversation(conversation.id)
    assert [m.role.value for m in saved.messages] == ["user", "assistant"]

    answer = saved.messages[-1]
    assert answer.verdict is not None
    assert answer.verdict.label.value in {"true", "false", "mixed", "unverified"}
    assert "Sources" in answer.content
    assert [step.stage for step in answer.trace][0] is Stage.DECOMPOSE
    assert saved.title.startswith("The Eiffel Tower")


async def test_searches_run_against_the_corpus_and_become_evidence(
    workflow: ClaimVerificationWorkflow, store: MemoryStore
):
    conversation = await store.create_conversation()
    events = await _run(workflow, conversation.id, CLAIM)

    searches = [e for e in events if e["type"] == "retrieval"]
    assert searches, "the verifier never searched"
    assert searches[0]["stage"] == "gather"
    assert searches[0]["query"] == CLAIM
    assert searches[0]["ok"] is True
    assert searches[0]["evidence"]

    contexts = await store.list_contexts(conversation.id)
    assert any(context.evidence for context in contexts)
    assert contexts[-1].running_summary


async def test_gathering_stops_at_the_search_budget(
    workflow: ClaimVerificationWorkflow, store: MemoryStore, settings
):
    conversation = await store.create_conversation()
    await _run(workflow, conversation.id, CLAIM)

    contexts = await store.list_contexts(conversation.id)
    assert contexts[-1].searches_used <= settings.max_gather_steps


async def test_a_failing_search_does_not_break_the_turn(
    workflow: ClaimVerificationWorkflow, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    conversation = await store.create_conversation()

    async def boom(*_args, **_kwargs):
        raise RetrievalError("cluster offline")

    monkeypatch.setattr(workflow._retriever, "search", boom)
    events = await _run(workflow, conversation.id, CLAIM)

    searches = [e for e in events if e["type"] == "retrieval"]
    assert searches and searches[0]["ok"] is False
    assert events[-1]["type"] == "done"

    saved = await store.get_conversation(conversation.id)
    assert saved.messages[-1].verdict is not None


async def test_context_revisions_are_written_for_every_decision(
    workflow: ClaimVerificationWorkflow, store: MemoryStore
):
    conversation = await store.create_conversation()
    await _run(workflow, conversation.id, CLAIM)

    contexts = await store.list_contexts(conversation.id)
    stages = [context.stage for context in contexts]
    assert stages[0] is Stage.DECOMPOSE
    assert Stage.VERDICT in stages
    assert len(contexts) > len(Stage)
    assert [c.revision for c in contexts] == sorted(c.revision for c in contexts)


async def test_a_message_without_a_claim_answers_directly(
    workflow: ClaimVerificationWorkflow, store: MemoryStore
):
    conversation = await store.create_conversation()
    events = await _run(workflow, conversation.id, "hi")

    stages = {e["stage"] for e in events if e["type"] == "stage"}
    assert stages == {"decompose"}

    saved = await store.get_conversation(conversation.id)
    assert saved.messages[-1].verdict is None
    assert "factual statement" in saved.messages[-1].content


async def test_an_unknown_conversation_is_reported(workflow: ClaimVerificationWorkflow):
    with pytest.raises(ConversationNotFound):
        await _run(workflow, "missing", CLAIM)


async def test_a_failing_verifier_still_closes_the_turn(
    workflow: ClaimVerificationWorkflow, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    conversation = await store.create_conversation()

    async def boom(**_kwargs):
        raise RuntimeError("gpu melted")

    monkeypatch.setattr(workflow._verifier, "run", boom)
    events = await _run(workflow, conversation.id, CLAIM)

    assert [e for e in events if e["type"] == "error"]
    assert events[-1]["type"] == "done"

    saved = await store.get_conversation(conversation.id)
    assert "could not be completed" in saved.messages[-1].content

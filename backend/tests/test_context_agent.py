from __future__ import annotations

import pytest

from backend.db.store import MemoryStore
from backend.models.schemas.context import (
    Claim,
    ClaimStatus,
    ContextObject,
    Decision,
    Stage,
    Stance,
)
from backend.services.agent.context_agent import ContextAgent

pytestmark = pytest.mark.asyncio


def _context(stage: Stage, **kwargs) -> ContextObject:
    return ContextObject(conversation_id="c1", turn_id="t1", stage=stage, **kwargs)


async def test_first_turn_context_is_empty(context_agent: ContextAgent, store: MemoryStore):
    conversation = await store.create_conversation()
    context = await context_agent.assemble(conversation=conversation, question="Is the sky blue?")

    assert context.stage is Stage.DECOMPOSE
    assert context.claims == []
    assert context.evidence == []
    assert context.decisions == []
    assert context.running_summary == ""
    assert await store.latest_context(conversation.id) is not None


async def test_second_turn_inherits_summary_and_sources(
    context_agent: ContextAgent, store: MemoryStore
):
    conversation = await store.create_conversation()
    first = await context_agent.assemble(conversation=conversation, question="first claim")
    first.running_summary = "previously established facts"
    first.evidence = [_evidence()]
    await store.save_context(first)

    second = await context_agent.assemble(conversation=conversation, question="follow up")
    assert second.running_summary == "previously established facts"
    assert [e.url for e in second.evidence] == ["https://example.org/a"]
    assert second.claims == []
    assert second.revision > first.revision


def _evidence():
    from backend.models.schemas.context import Evidence

    return Evidence(title="a source", url="https://example.org/a", snippet="text")


async def test_commit_applies_assessments_to_claims_and_evidence(context_agent: ContextAgent):
    claim = Claim(text="a claim")
    evidence = _evidence()
    context = _context(Stage.ASSESS, claims=[claim], evidence=[evidence])
    decision = Decision(
        stage=Stage.ASSESS,
        output={
            "assessments": [
                {
                    "claim_id": claim.id,
                    "status": "refuted",
                    "rationale": "the source says otherwise",
                    "evidence_ids": [evidence.id, "does-not-exist"],
                }
            ]
        },
    )

    committed = await context_agent.commit(context=context, decision=decision)

    assert committed.claims[0].status is ClaimStatus.REFUTED
    assert committed.claims[0].evidence_ids == [evidence.id]
    assert committed.evidence[0].stance is Stance.REFUTES


async def test_compact_writes_a_summary_without_calling_the_model(context_agent: ContextAgent):
    context = _context(
        Stage.VERDICT,
        question="Is the sky blue?",
        claims=[Claim(text="the sky is blue", status=ClaimStatus.SUPPORTED)],
    )

    compacted = await context_agent.compact(context)

    assert "Is the sky blue?" in compacted.running_summary
    assert "supported" in compacted.running_summary

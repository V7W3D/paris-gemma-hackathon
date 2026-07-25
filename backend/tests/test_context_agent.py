from __future__ import annotations

import pytest

from backend.db.store import MemoryStore
from backend.models.schemas.context import (
    Chunk,
    Claim,
    ClaimStatus,
    ContextObject,
    Decision,
    Retrieval,
    Stage,
    Stance,
)
from backend.services.agent.context_agent import ContextAgent

pytestmark = pytest.mark.asyncio


def _context(stage: Stage, **kwargs) -> ContextObject:
    return ContextObject(conversation_id="c1", turn_id="t1", stage=stage, search_budget=2, **kwargs)


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


async def test_commit_turns_chunks_into_evidence(context_agent: ContextAgent):
    context = _context(Stage.GATHER, claims=[Claim(text="a claim")])
    decision = Decision(stage=Stage.GATHER, summary="searching", queries=["a claim"])
    retrieval = Retrieval(
        query="a claim",
        chunks=[
            Chunk(title="One", url="https://a.example/1", text="passage one", score=0.8),
            Chunk(title="One", url="https://a.example/1", text="passage one", score=0.8),
            Chunk(title="Two", url="https://b.example/2", text="passage two", score=0.6),
        ],
    )

    committed = await context_agent.commit(
        context=context, decision=decision, retrievals=[retrieval]
    )

    assert [e.snippet for e in committed.evidence] == ["passage one", "passage two"]
    assert committed.evidence[0].claim_id == context.claims[0].id
    assert committed.evidence[0].query == "a claim"
    assert committed.evidence[0].credibility == 0.8
    assert committed.searches_used == 1
    assert retrieval.evidence_ids == [e.id for e in committed.evidence]


async def test_commit_keeps_distinct_passages_from_one_document(context_agent: ContextAgent):
    context = _context(Stage.GATHER)
    retrieval = Retrieval(
        query="a claim",
        chunks=[
            Chunk(title="Report", url="https://a.example/1", text="first passage"),
            Chunk(title="Report", url="https://a.example/1", text="second passage"),
        ],
    )

    committed = await context_agent.commit(
        context=context, decision=Decision(stage=Stage.GATHER), retrievals=[retrieval]
    )

    assert len(committed.evidence) == 2


async def test_commit_records_a_failed_search_as_an_open_question(context_agent: ContextAgent):
    context = _context(Stage.GATHER)
    retrieval = Retrieval(query="a claim", ok=False, error="cluster offline")

    committed = await context_agent.commit(
        context=context,
        decision=Decision(stage=Stage.GATHER),
        retrievals=[retrieval],
    )

    assert committed.open_questions == ["search for 'a claim' failed: cluster offline"]
    assert committed.evidence == []


async def test_commit_records_the_queries_already_run(context_agent: ContextAgent):
    context = _context(Stage.GATHER)

    committed = await context_agent.commit(
        context=context,
        decision=Decision(stage=Stage.GATHER),
        retrievals=[Retrieval(query="a claim")],
    )

    assert committed.prompt_view()["queries_already_run"] == ["a claim"]
    assert committed.prompt_view()["searches_left"] == 1


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

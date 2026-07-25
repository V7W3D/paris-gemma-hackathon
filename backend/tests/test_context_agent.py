from __future__ import annotations

import pytest

from backend.config import Settings
from backend.db.store import MemoryStore
from backend.models.schemas.context import (
    Claim,
    ClaimStatus,
    ContextObject,
    Decision,
    Stage,
    Stance,
    ToolResult,
    ToolSpec,
)
from backend.services.agent.context_agent import ContextAgent
from backend.services.agent.llm_client import Inference, LLMError

pytestmark = pytest.mark.asyncio

TOOLS = [
    ToolSpec(name="web_search", description="search the web"),
    ToolSpec(name="fetch_url", description="read a page"),
]


def _context(stage: Stage, **kwargs) -> ContextObject:
    return ContextObject(conversation_id="c1", turn_id="t1", stage=stage, tool_budget=2, **kwargs)


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


async def test_curation_exposes_no_tools_outside_gathering(context_agent: ContextAgent):
    for stage in (Stage.DECOMPOSE, Stage.PLAN, Stage.ASSESS, Stage.VERDICT):
        assert await context_agent.curate(context=_context(stage), available_tools=TOOLS) == []


async def test_curation_starts_with_search_only(context_agent: ContextAgent):
    curated = await context_agent.curate(context=_context(Stage.GATHER), available_tools=TOOLS)
    assert [tool.name for tool in curated] == ["web_search"]


async def test_curation_stops_at_the_tool_budget(context_agent: ContextAgent):
    context = _context(Stage.GATHER)
    context.tools_used = context.tool_budget
    assert await context_agent.curate(context=context, available_tools=TOOLS) == []


async def test_curation_falls_back_to_policy_when_inference_fails(
    store: MemoryStore, settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    inference = Inference(settings)

    async def boom(*_args, **_kwargs):
        raise LLMError("endpoint down")

    monkeypatch.setattr(inference, "run", boom)
    agent = ContextAgent(inference, store, settings)

    context = _context(Stage.GATHER, evidence=[])
    context.evidence.append(_evidence())
    curated = await agent.curate(context=context, available_tools=TOOLS)

    assert [tool.name for tool in curated] == ["web_search", "fetch_url"]


def _evidence():
    from backend.models.schemas.context import Evidence

    return Evidence(title="a source", url="https://example.org/a", snippet="text")


async def test_commit_turns_search_results_into_evidence(context_agent: ContextAgent):
    context = _context(Stage.GATHER, claims=[Claim(text="a claim")])
    decision = Decision(stage=Stage.GATHER, summary="searching")
    result = ToolResult(
        tool="web_search",
        arguments={"query": "a claim"},
        raw={
            "results": [
                {"title": "One", "url": "https://a.example/1", "snippet": "s1", "source": "a"},
                {"title": "Dup", "url": "https://a.example/1", "snippet": "s2", "source": "a"},
                {"title": "Two", "url": "https://b.example/2", "snippet": "s3", "source": "b"},
            ]
        },
    )

    committed = await context_agent.commit(
        context=context, decision=decision, tool_results=[result]
    )

    assert [e.url for e in committed.evidence] == ["https://a.example/1", "https://b.example/2"]
    assert committed.evidence[0].claim_id == context.claims[0].id
    assert committed.tools_used == 1
    assert result.evidence_ids == [e.id for e in committed.evidence]


async def test_commit_records_a_failed_tool_call_as_an_open_question(context_agent: ContextAgent):
    context = _context(Stage.GATHER)
    result = ToolResult(tool="fetch_url", ok=False, error="404")

    committed = await context_agent.commit(
        context=context,
        decision=Decision(stage=Stage.GATHER),
        tool_results=[result],
    )

    assert committed.open_questions == ["fetch_url failed: 404"]
    assert committed.evidence == []


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

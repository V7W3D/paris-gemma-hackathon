from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator
from uuid import uuid4

from backend.config import Settings
from backend.db.store import Store
from backend.models.schemas.chat import Conversation, Message, Role, TraceStep
from backend.models.schemas.context import (
    ContextObject,
    Decision,
    Stage,
    ToolResult,
    ToolSpec,
    Verdict,
    VerdictLabel,
)
from backend.services.agent.context_agent import ContextAgent
from backend.services.agent.verifier_agent import VerifierAgent
from backend.services.mcp.client import MCPToolClient, MCPUnavailableError

logger = logging.getLogger(__name__)

Event = dict[str, Any]

DEFAULT_TITLE = "New verification"
REVEAL_CHUNKS = 40
REVEAL_DELAY_SECONDS = 0.02


class ConversationNotFound(LookupError):
    pass


class ClaimVerificationWorkflow:
    """Drives one user turn through the five decision points.

    At every point the context agent assembles context and curates tools, the
    verifier agent decides, the orchestrator executes any tool calls over MCP,
    and the context agent folds the outcome back into a new context revision.
    """

    def __init__(
        self,
        *,
        verifier: VerifierAgent,
        context_agent: ContextAgent,
        tools: MCPToolClient,
        store: Store,
        settings: Settings,
    ) -> None:
        self._verifier = verifier
        self._context = context_agent
        self._tools = tools
        self._store = store
        self._settings = settings

    async def run(self, *, conversation_id: str, content: str) -> AsyncIterator[Event]:
        conversation = await self._store.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFound(conversation_id)

        turn_id = uuid4().hex[:12]
        user_message = Message(role=Role.USER, content=content, turn_id=turn_id)
        conversation = await self._store.append_message(conversation_id, user_message) or conversation
        if conversation.title == DEFAULT_TITLE:
            conversation = (
                await self._store.rename_conversation(conversation_id, _title_from(content))
                or conversation
            )

        yield {
            "type": "turn_started",
            "turn_id": turn_id,
            "conversation_title": conversation.title,
            "message": user_message.model_dump(mode="json"),
        }

        try:
            async for event in self._run_turn(conversation, content, turn_id):
                yield event
        except Exception as exc:  # noqa: BLE001 - the turn must always terminate cleanly
            logger.exception("Verification turn failed")
            message = await self._finish(
                conversation_id,
                turn_id,
                content=f"The verification could not be completed: {exc}",
                verdict=None,
                trace=[],
            )
            yield {"type": "error", "error": str(exc)}
            yield {"type": "message", "message": message.model_dump(mode="json")}

        yield {"type": "done"}

    # ------------------------------------------------------------------ turn

    async def _run_turn(
        self, conversation: Conversation, content: str, turn_id: str
    ) -> AsyncIterator[Event]:
        context = await self._context.assemble(
            conversation=conversation, question=content, turn_id=turn_id
        )
        trace: list[TraceStep] = []

        try:
            available = await self._tools.list_tools()
        except MCPUnavailableError as exc:
            logger.warning("%s; continuing without tools", exc)
            available = []
            yield {"type": "warning", "message": str(exc)}

        # 1. DECOMPOSE
        context, decision, events = await self._step(context, Stage.DECOMPOSE, available, trace)
        for event in events:
            yield event
        if not context.claims:
            reply = str(decision.output.get("reply") or "").strip() or (
                "I could not find a factual claim to verify in that message. "
                "Send me a statement and I will check it against sources."
            )
            async for event in self._reveal(reply):
                yield event
            message = await self._finish(
                conversation.id, turn_id, content=reply, verdict=None, trace=trace
            )
            await self._context.compact(context)
            yield {"type": "message", "message": message.model_dump(mode="json")}
            return

        yield {"type": "claims", "claims": [c.model_dump(mode="json") for c in context.claims]}

        # 2. PLAN
        context, _, events = await self._step(context, Stage.PLAN, available, trace)
        for event in events:
            yield event

        # 3. GATHER (loop until the verifier is satisfied or the budget runs out)
        for _ in range(self._settings.max_gather_steps):
            context, decision, events = await self._step(context, Stage.GATHER, available, trace)
            for event in events:
                yield event
            if decision.done or not decision.tool_calls:
                break

        # 4. ASSESS
        context, _, events = await self._step(context, Stage.ASSESS, available, trace)
        for event in events:
            yield event

        # 5. VERDICT
        context, _, events = await self._step(context, Stage.VERDICT, available, trace)
        for event in events:
            yield event

        verdict = context.verdict or Verdict(label=VerdictLabel.UNVERIFIED, claims=context.claims)
        answer = _render_answer(verdict)
        async for event in self._reveal(answer):
            yield event

        message = await self._finish(
            conversation.id, turn_id, content=answer, verdict=verdict, trace=trace
        )
        await self._context.compact(context)
        yield {"type": "message", "message": message.model_dump(mode="json")}

    async def _step(
        self,
        context: ContextObject,
        stage: Stage,
        available: list[ToolSpec],
        trace: list[TraceStep],
    ) -> tuple[ContextObject, Decision, list[Event]]:
        """One decision point: curate, decide, execute tools, commit."""
        context = self._context.advance(context, stage)
        curated = await self._context.curate(context=context, available_tools=available)
        context.curated_tools = [tool.name for tool in curated]

        events: list[Event] = [
            {
                "type": "stage",
                "stage": stage.value,
                "status": "started",
                "curated_tools": context.curated_tools,
            }
        ]

        decision = await self._verifier.run(context=context, tools=curated)

        results: list[ToolResult] = []
        for call in decision.tool_calls:
            outcome = await self._tools.call_tool(call.tool, call.arguments)
            results.append(
                ToolResult(
                    tool=call.tool,
                    arguments=call.arguments,
                    ok=bool(outcome.get("ok")),
                    error=str(outcome.get("error", "")),
                    raw=outcome.get("data"),
                )
            )

        context = await self._context.commit(
            context=context, decision=decision, tool_results=results
        )

        for result in results:
            events.append(
                {
                    "type": "tool_call",
                    "stage": stage.value,
                    "tool": result.tool,
                    "arguments": result.arguments,
                    "ok": result.ok,
                    "error": result.error,
                    "evidence": [
                        e.model_dump(mode="json")
                        for e in context.evidence
                        if e.id in result.evidence_ids
                    ],
                }
            )

        trace.append(
            TraceStep(
                stage=stage,
                summary=decision.summary,
                curated_tools=context.curated_tools,
                tool_calls=[
                    {
                        "tool": r.tool,
                        "arguments": r.arguments,
                        "ok": r.ok,
                        "error": r.error,
                        "evidence_count": len(r.evidence_ids),
                    }
                    for r in results
                ],
            )
        )

        events.append(
            {
                "type": "stage",
                "stage": stage.value,
                "status": "completed",
                "summary": decision.summary,
                "detail": _stage_detail(stage, context),
            }
        )
        return context, decision, events

    async def _reveal(self, text: str) -> AsyncIterator[Event]:
        """Push the final answer out in chunks so the UI can render it as it lands."""
        if not text:
            return
        size = max(len(text) // REVEAL_CHUNKS, 24)
        for start in range(0, len(text), size):
            yield {"type": "token", "text": text[start : start + size]}
            await asyncio.sleep(REVEAL_DELAY_SECONDS)

    async def _finish(
        self,
        conversation_id: str,
        turn_id: str,
        *,
        content: str,
        verdict: Verdict | None,
        trace: list[TraceStep],
    ) -> Message:
        message = Message(
            role=Role.ASSISTANT,
            content=content,
            turn_id=turn_id,
            verdict=verdict,
            trace=trace,
        )
        await self._store.append_message(conversation_id, message)
        return message


def _title_from(content: str) -> str:
    flat = " ".join(content.split())
    return flat[:60] + ("..." if len(flat) > 60 else "") if flat else DEFAULT_TITLE


def _stage_detail(stage: Stage, context: ContextObject) -> dict[str, Any]:
    if stage is Stage.DECOMPOSE:
        return {"claims": [c.text for c in context.claims]}
    if stage is Stage.PLAN:
        return {"open_questions": context.open_questions[-4:]}
    if stage is Stage.GATHER:
        return {"evidence_count": len(context.evidence)}
    if stage is Stage.ASSESS:
        return {
            "statuses": [
                {"claim": c.text, "status": c.status.value, "rationale": c.rationale}
                for c in context.claims
            ]
        }
    if stage is Stage.VERDICT and context.verdict is not None:
        return {
            "label": context.verdict.label.value,
            "confidence": context.verdict.confidence,
        }
    return {}


def _render_answer(verdict: Verdict) -> str:
    body = verdict.summary.strip()
    if not body:
        lines = [f"**Verdict: {verdict.label.value}**", ""]
        lines += [f"- {c.text} — {c.status.value}. {c.rationale}".rstrip() for c in verdict.claims]
        body = "\n".join(lines)

    sourced = [s for s in verdict.sources if s.url]
    if sourced:
        body += "\n\n**Sources**\n"
        body += "\n".join(
            f"{index}. [{source.title or source.url}]({source.url})"
            for index, source in enumerate(sourced, start=1)
        )
    return body

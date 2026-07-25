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
    Retrieval,
    Stage,
    Verdict,
    VerdictLabel,
)
from backend.services.agent.context_agent import ContextAgent
from backend.services.agent.direct_agent import DirectAgent
from backend.services.agent.verifier_agent import VerifierAgent
from backend.services.retrieval.alien_client import AlienRetriever, RetrievalError

logger = logging.getLogger(__name__)

Event = dict[str, Any]

DEFAULT_TITLE = "New verification"
REVEAL_CHUNKS = 40
REVEAL_DELAY_SECONDS = 0.02
DIRECT_HISTORY_MESSAGES = 6


class ConversationNotFound(LookupError):
    pass


class ClaimVerificationWorkflow:
    """Drives one user turn through the five decision points.

    At every point the context agent assembles the context, the verifier agent
    decides, the orchestrator runs any searches the verifier asked for against
    the Alien corpus, and the context agent folds the outcome back into a new
    context revision.

    A turn can also be run with the verifier switched off, which skips all of
    that and hands the message to the direct agent instead.
    """

    def __init__(
        self,
        *,
        verifier: VerifierAgent,
        context_agent: ContextAgent,
        direct_agent: DirectAgent,
        retriever: AlienRetriever,
        store: Store,
        settings: Settings,
    ) -> None:
        self._verifier = verifier
        self._context = context_agent
        self._direct = direct_agent
        self._retriever = retriever
        self._store = store
        self._settings = settings

    async def run(
        self, *, conversation_id: str, content: str, use_verifier: bool = True
    ) -> AsyncIterator[Event]:
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
            "mode": "verified" if use_verifier else "direct",
            "message": user_message.model_dump(mode="json"),
        }

        turn = (
            self._run_turn(conversation, content, turn_id)
            if use_verifier
            else self._run_direct_turn(conversation, content, turn_id)
        )

        try:
            async for event in turn:
                yield event
        except Exception as exc:  # noqa: BLE001 - the turn must always terminate cleanly
            logger.exception("Turn failed")
            failed = (
                "The verification could not be completed"
                if use_verifier
                else "The answer could not be generated"
            )
            message = await self._finish(
                conversation_id,
                turn_id,
                content=f"{failed}: {exc}",
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

        # 1. DECOMPOSE
        context, decision, events = await self._step(context, Stage.DECOMPOSE, trace)
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
        context, _, events = await self._step(context, Stage.PLAN, trace)
        for event in events:
            yield event

        # 3. GATHER (loop until the verifier is satisfied or the budget runs out)
        for _ in range(self._settings.max_gather_steps):
            context, decision, events = await self._step(context, Stage.GATHER, trace)
            for event in events:
                yield event
            if decision.done or not decision.queries or context.searches_used >= context.search_budget:
                break

        # 4. ASSESS
        context, _, events = await self._step(context, Stage.ASSESS, trace)
        for event in events:
            yield event

        # 5. VERDICT
        context, _, events = await self._step(context, Stage.VERDICT, trace)
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

    async def _run_direct_turn(
        self, conversation: Conversation, content: str, turn_id: str
    ) -> AsyncIterator[Event]:
        """The verifier is off: answer in one call, with no context and no sources."""
        answer = await self._direct.run(
            question=content, history=_transcript(conversation, turn_id)
        )
        async for event in self._reveal(answer):
            yield event

        message = await self._finish(
            conversation.id, turn_id, content=answer, verdict=None, trace=[]
        )
        yield {"type": "message", "message": message.model_dump(mode="json")}

    async def _step(
        self,
        context: ContextObject,
        stage: Stage,
        trace: list[TraceStep],
    ) -> tuple[ContextObject, Decision, list[Event]]:
        """One decision point: decide, run any searches asked for, commit."""
        context = self._context.advance(context, stage)

        events: list[Event] = [{"type": "stage", "stage": stage.value, "status": "started"}]

        decision = await self._verifier.run(context=context)
        retrievals = (
            await self._search(decision.queries, context) if stage is Stage.GATHER else []
        )

        context = await self._context.commit(
            context=context, decision=decision, retrievals=retrievals
        )

        for retrieval in retrievals:
            events.append(
                {
                    "type": "retrieval",
                    "stage": stage.value,
                    "query": retrieval.query,
                    "ok": retrieval.ok,
                    "error": retrieval.error,
                    "evidence": [
                        e.model_dump(mode="json")
                        for e in context.evidence
                        if e.id in retrieval.evidence_ids
                    ],
                }
            )

        trace.append(
            TraceStep(
                stage=stage,
                summary=decision.summary,
                retrievals=[
                    {
                        "query": r.query,
                        "ok": r.ok,
                        "error": r.error,
                        "evidence_count": len(r.evidence_ids),
                    }
                    for r in retrievals
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

    async def _search(self, queries: list[str], context: ContextObject) -> list[Retrieval]:
        """Run the verifier's queries, capped by what is left of the search budget."""
        budget = max(context.search_budget - context.searches_used, 0)
        retrievals: list[Retrieval] = []

        for query in queries[:budget]:
            try:
                chunks = await self._retriever.search(query)
            except RetrievalError as exc:
                retrievals.append(Retrieval(query=query, ok=False, error=str(exc)))
                continue
            retrievals.append(Retrieval(query=query, chunks=chunks))

        if len(queries) > budget:
            logger.info("Dropped %d queries over the search budget", len(queries) - budget)
        return retrievals

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


def _transcript(conversation: Conversation, turn_id: str) -> str:
    """The last few exchanges, flattened for the direct agent's prompt.

    The message this turn just appended is left out: it goes in as the question.
    """
    earlier = [m for m in conversation.messages if m.turn_id != turn_id]
    return "\n".join(
        f"{message.role.value}: {message.content[:400]}"
        for message in earlier[-DIRECT_HISTORY_MESSAGES:]
    )


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

    # A corpus of documents does not always have somewhere to link to, so a
    # source counts as citable once it has a name.
    cited = [s for s in verdict.sources if s.url or s.title]
    if cited:
        body += "\n\n**Sources**\n"
        body += "\n".join(
            f"{index}. [{source.title or source.url}]({source.url})"
            if source.url
            else f"{index}. {source.title}" + (f" — {source.source}" if source.source else "")
            for index, source in enumerate(cited, start=1)
        )
    return body

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from backend.config import Settings
from backend.db.store import Store
from backend.models.schemas.actions import CompactionOutput
from backend.models.schemas.chat import Conversation, Role
from backend.models.schemas.context import (
    Claim,
    ClaimStatus,
    ContextObject,
    Decision,
    Evidence,
    Retrieval,
    Stage,
    Stance,
    Verdict,
    VerdictLabel,
)
from backend.services.agent.llm_client import Inference, LLMError
from backend.services.agent.prompts import build_compaction_prompt

logger = logging.getLogger(__name__)

COMPACTION_TRIGGER_CHARS = 6000


class ContextAgent:
    """Agent 2: owns the context object, and only the context object.

    Everything agent 1 knows at a decision point was put there by this class:
    it reads the conversation from Mongo, assembles the context, folds each
    decision and its retrievals back in, and persists the next revision.
    """

    def __init__(self, inference: Inference, store: Store, settings: Settings) -> None:
        self._inference = inference
        self._store = store
        self._settings = settings

    # ------------------------------------------------------------------ reads

    async def fetch_conversation(self, conversation_id: str) -> Conversation | None:
        return await self._store.get_conversation(conversation_id)

    async def latest_context(self, conversation_id: str) -> ContextObject | None:
        return await self._store.latest_context(conversation_id)

    async def list_contexts(self, conversation_id: str) -> list[ContextObject]:
        return await self._store.list_contexts(conversation_id)

    # -------------------------------------------------------------- assemble

    async def assemble(
        self,
        *,
        conversation: Conversation,
        question: str,
        turn_id: str | None = None,
    ) -> ContextObject:
        """Build the context object for a new turn.

        On the very first turn of a conversation this is an empty context, as
        specified: no claims, no evidence, no decisions. Later turns inherit the
        compacted summary and the sources already collected.
        """
        previous = await self.latest_context(conversation.id)
        carried: list[Evidence] = []
        summary = ""
        revision = 0

        if previous is not None:
            revision = previous.revision + 1
            summary = previous.running_summary
            carried = previous.evidence[-self._settings.max_evidence_in_context :]
            if not summary:
                summary = self._fallback_summary(conversation)

        context = ContextObject(
            conversation_id=conversation.id,
            turn_id=turn_id or uuid4().hex[:12],
            revision=revision,
            stage=Stage.DECOMPOSE,
            question=question,
            running_summary=summary,
            evidence=carried,
            search_budget=self._settings.max_gather_steps,
        )
        await self._store.save_context(context)
        return context

    @staticmethod
    def _fallback_summary(conversation: Conversation) -> str:
        earlier = [m for m in conversation.messages if m.role == Role.USER][-3:]
        if not earlier:
            return ""
        asked = "; ".join(m.content[:120] for m in earlier)
        return f"Earlier in this conversation the user asked to verify: {asked}"

    def advance(self, context: ContextObject, stage: Stage) -> ContextObject:
        """Move the context to the next decision point without touching content."""
        return context.model_copy(
            update={
                "id": uuid4().hex[:12],
                "stage": stage,
                "revision": context.revision + 1,
            },
            deep=True,
        )

    # ---------------------------------------------------------------- commit

    async def commit(
        self,
        *,
        context: ContextObject,
        decision: Decision,
        retrievals: list[Retrieval] | None = None,
    ) -> ContextObject:
        """Fold a decision (and anything its searches returned) into a new revision."""
        updated = context.model_copy(deep=True)
        updated.id = uuid4().hex[:12]
        updated.revision = context.revision + 1
        updated.decisions = [*context.decisions, decision]

        for retrieval in retrievals or []:
            updated.searches_used += 1
            updated.retrievals.append(retrieval)
            if retrieval.ok:
                new_evidence = self._evidence_from_chunks(retrieval, updated)
                retrieval.evidence_ids = [e.id for e in new_evidence]
                updated.evidence.extend(new_evidence)
            else:
                updated.open_questions.append(
                    f"search for {retrieval.query!r} failed: {retrieval.error}"
                )

        handler = {
            Stage.DECOMPOSE: self._apply_decompose,
            Stage.PLAN: self._apply_plan,
            Stage.ASSESS: self._apply_assess,
            Stage.VERDICT: self._apply_verdict,
        }.get(decision.stage)
        if handler is not None:
            handler(updated, decision.output)

        await self._store.save_context(updated)
        return updated

    @staticmethod
    def _evidence_from_chunks(retrieval: Retrieval, context: ContextObject) -> list[Evidence]:
        """Admit the hits the context does not already hold, attributed to the query.

        Several chunks routinely come out of the same document, so identity is
        the passage rather than the source URL.
        """
        seen = {(e.url, e.snippet[:120]) for e in context.evidence}
        claim_id = context.claims[0].id if len(context.claims) == 1 else None
        found: list[Evidence] = []

        for chunk in retrieval.chunks:
            key = (chunk.url, chunk.text[:120])
            if key in seen:
                continue
            seen.add(key)
            found.append(
                Evidence(
                    claim_id=claim_id,
                    title=chunk.title[:300],
                    url=chunk.url,
                    snippet=chunk.text[:1200],
                    source=chunk.source,
                    credibility=chunk.score or 0.5,
                    query=retrieval.query,
                )
            )
        return found

    def _apply_decompose(self, context: ContextObject, output: dict[str, Any]) -> None:
        raw_claims = output.get("claims")
        claims: list[Claim] = []
        if isinstance(raw_claims, list):
            for item in raw_claims[: self._settings.max_claims]:
                text = item.get("text") if isinstance(item, dict) else item
                text = str(text or "").strip()
                if text:
                    claims.append(Claim(text=text))
        context.claims = claims
        reply = output.get("reply")
        if not claims and isinstance(reply, str) and reply.strip():
            context.open_questions.append("no checkable claim in the user message")

    @staticmethod
    def _apply_plan(context: ContextObject, output: dict[str, Any]) -> None:
        needs = output.get("evidence_needs")
        if isinstance(needs, list):
            for item in needs:
                if isinstance(item, dict) and item.get("needs"):
                    context.open_questions.append(str(item["needs"])[:300])
                elif isinstance(item, str):
                    context.open_questions.append(item[:300])
        queries = output.get("queries")
        if isinstance(queries, list) and queries:
            context.running_summary = (
                context.running_summary
                or f"Planned queries: {'; '.join(str(q) for q in queries[:4])}"
            )

    @staticmethod
    def _apply_assess(context: ContextObject, output: dict[str, Any]) -> None:
        assessments = output.get("assessments")
        if not isinstance(assessments, list):
            return
        for item in assessments:
            if not isinstance(item, dict):
                continue
            claim = context.claim_by_id(str(item.get("claim_id", "")))
            if claim is None:
                continue
            try:
                claim.status = ClaimStatus(str(item.get("status", "insufficient")).lower())
            except ValueError:
                claim.status = ClaimStatus.INSUFFICIENT
            claim.rationale = str(item.get("rationale", ""))[:600]
            evidence_ids = item.get("evidence_ids")
            if isinstance(evidence_ids, list):
                claim.evidence_ids = [str(e) for e in evidence_ids if context.evidence_by_id(str(e))]
            stance = {
                ClaimStatus.SUPPORTED: Stance.SUPPORTS,
                ClaimStatus.REFUTED: Stance.REFUTES,
            }.get(claim.status, Stance.UNCLEAR)
            for evidence_id in claim.evidence_ids:
                evidence = context.evidence_by_id(evidence_id)
                if evidence is not None:
                    evidence.stance = stance
                    evidence.claim_id = evidence.claim_id or claim.id

    def _apply_verdict(self, context: ContextObject, output: dict[str, Any]) -> None:
        try:
            label = VerdictLabel(str(output.get("label", "unverified")).lower())
        except ValueError:
            label = VerdictLabel.UNVERIFIED
        try:
            confidence = min(max(float(output.get("confidence", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            confidence = 0.0
        cited = [e for e in context.evidence if e.stance is not Stance.UNCLEAR] or context.evidence
        context.verdict = Verdict(
            label=label,
            confidence=confidence,
            summary=str(output.get("summary", "")).strip(),
            claims=context.claims,
            sources=cited[: self._settings.max_evidence_in_context],
        )

    # --------------------------------------------------------------- compact

    async def compact(self, context: ContextObject, *, force: bool = False) -> ContextObject:
        """Summarise the turn so the next one starts from a small context.

        The model is only asked to do it when the context has actually grown;
        below that threshold a deterministic summary is cheaper and sufficient.
        """
        size = len(json.dumps(context.prompt_view(self._settings.max_evidence_in_context)))
        summary = ""

        if force or size >= COMPACTION_TRIGGER_CHARS:
            prompt = build_compaction_prompt(
                stage=context.stage,
                context_view=context.prompt_view(self._settings.max_evidence_in_context),
            )
            try:
                output = await self._inference.run(
                    prompt, CompactionOutput, temperature=0.0, max_tokens=400
                )
                summary = output.summary.strip()
            except LLMError as exc:
                logger.warning("Compaction failed, keeping the deterministic summary: %s", exc)

        if not summary:
            statuses = "; ".join(f"{c.text[:80]} -> {c.status.value}" for c in context.claims)
            summary = (
                f"The user asked about: {context.question[:200]}. Outcome: {statuses}"
                if statuses
                else context.running_summary
            )

        updated = context.model_copy(deep=True)
        updated.id = uuid4().hex[:12]
        updated.revision = context.revision + 1
        updated.running_summary = summary[:1200]
        await self._store.save_context(updated)
        return updated

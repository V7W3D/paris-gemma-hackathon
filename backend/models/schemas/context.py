from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return uuid4().hex[:12]


class Stage(str, Enum):
    """The decision points of the claim-verification workflow."""

    DECOMPOSE = "decompose"
    PLAN = "plan"
    ASSESS = "assess"
    VERDICT = "verdict"


STAGE_ORDER: list[Stage] = [
    Stage.DECOMPOSE,
    Stage.PLAN,
    Stage.ASSESS,
    Stage.VERDICT,
]


class ClaimStatus(str, Enum):
    PENDING = "pending"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INSUFFICIENT = "insufficient"


class Stance(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    UNCLEAR = "unclear"


class VerdictLabel(str, Enum):
    TRUE = "true"
    FALSE = "false"
    MIXED = "mixed"
    UNVERIFIED = "unverified"


class Claim(BaseModel):
    id: str = Field(default_factory=_uid)
    text: str
    status: ClaimStatus = ClaimStatus.PENDING
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    id: str = Field(default_factory=_uid)
    claim_id: str | None = None
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""
    stance: Stance = Stance.UNCLEAR
    credibility: float = 0.5
    query: str = ""
    retrieved_at: datetime = Field(default_factory=_now)


class Chunk(BaseModel):
    """A raw source passage, before agent 2 admits it as evidence."""

    title: str = ""
    text: str = ""
    url: str = ""
    source: str = ""
    score: float = 0.0


class Retrieval(BaseModel):
    """One semantic search run against the corpus, and what came back."""

    id: str = Field(default_factory=_uid)
    query: str
    ok: bool = True
    error: str = ""
    chunks: list[Chunk] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    """What agent 1 decided at one decision point."""

    id: str = Field(default_factory=_uid)
    stage: Stage
    summary: str = ""
    queries: list[str] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    done: bool = True
    created_at: datetime = Field(default_factory=_now)


class Verdict(BaseModel):
    label: VerdictLabel = VerdictLabel.UNVERIFIED
    confidence: float = 0.0
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    sources: list[Evidence] = Field(default_factory=list)


class ContextObject(BaseModel):
    """The engineered context handed to agent 1 at every decision point.

    Agent 2 owns this object and nothing else: it assembles it, folds each
    decision and its retrievals into it, and persists a new revision.
    """

    id: str = Field(default_factory=_uid)
    conversation_id: str
    turn_id: str
    revision: int = 0
    stage: Stage = Stage.DECOMPOSE
    question: str = ""
    running_summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    retrievals: list[Retrieval] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    search_budget: int = 0
    searches_used: int = 0
    verdict: Verdict | None = None
    created_at: datetime = Field(default_factory=_now)

    def claim_by_id(self, claim_id: str) -> Claim | None:
        return next((c for c in self.claims if c.id == claim_id), None)

    def evidence_by_id(self, evidence_id: str) -> Evidence | None:
        return next((e for e in self.evidence if e.id == evidence_id), None)

    def prompt_view(self, max_evidence: int = 12) -> dict[str, Any]:
        """The trimmed projection of the context that actually enters the prompt.

        Raw retrieval payloads and the full decision history stay in the
        database; the model only sees claims, recent evidence and a compacted
        trail.
        """
        return {
            "question": self.question,
            "stage": self.stage.value,
            "running_summary": self.running_summary,
            "claims": [
                {
                    "id": c.id,
                    "text": c.text,
                    "status": c.status.value,
                    "rationale": c.rationale,
                    "evidence_ids": c.evidence_ids,
                }
                for c in self.claims
            ],
            "evidence": [
                {
                    "id": e.id,
                    "claim_id": e.claim_id,
                    "title": e.title,
                    "url": e.url,
                    "snippet": e.snippet[:600],
                    "stance": e.stance.value,
                }
                for e in self.evidence[-max_evidence:]
            ],
            "open_questions": self.open_questions,
            "decision_trail": [
                {"stage": d.stage.value, "summary": d.summary} for d in self.decisions
            ],
            "queries_already_run": [r.query for r in self.retrievals],
            "searches_left": max(self.search_budget - self.searches_used, 0),
        }

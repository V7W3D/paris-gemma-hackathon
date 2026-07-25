from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, Field, field_validator

from backend.models.schemas.context import ClaimStatus, Stage, VerdictLabel


class StageOutput(BaseModel):
    """Base of the JSON envelope agent 1 returns at a decision point.

    Gemma has no dependable native tool-calling, so these models are handed to
    Pydantic AI in prompted-output mode: the JSON schema is injected into the
    prompt and the reply is validated (and re-asked) against it.
    """

    thought: str = Field(default="", description="One short sentence explaining the decision.")


class DecomposeOutput(StageOutput):
    """Atomic factual claims found in the user's message."""

    claims: list[str] = Field(
        default_factory=list,
        description="Self-contained checkable claims. Empty when the message contains none.",
    )
    reply: str = Field(
        default="",
        description="Direct answer to the user, only when claims is empty.",
    )

    @field_validator("claims", mode="before")
    @classmethod
    def _flatten(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [item.get("text", "") if isinstance(item, dict) else item for item in value]
        return value


class EvidenceNeed(BaseModel):
    claim_id: str = Field(description="Id of the claim from the context object.")
    needs: str = Field(description="What evidence would settle this claim.")


class PlanOutput(StageOutput):
    """What it would take to settle each claim."""

    evidence_needs: list[EvidenceNeed] = Field(default_factory=list)
    queries: list[str] = Field(
        default_factory=list, description="Search queries worth running, most specific first."
    )


class GatherOutput(StageOutput):
    """The searches to run next, or the decision to stop gathering."""

    queries: list[str] = Field(
        default_factory=list,
        description="Searches to run against the corpus. Empty when no more are needed.",
    )
    done: bool = Field(
        default=False, description="True when the evidence collected is enough to judge."
    )

    @field_validator("queries", mode="before")
    @classmethod
    def _flatten(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [item.get("query", "") if isinstance(item, dict) else item for item in value]
        return value


class ClaimAssessment(BaseModel):
    claim_id: str = Field(description="Id of the claim from the context object.")
    status: ClaimStatus = Field(description="supported, refuted or insufficient.")
    rationale: str = Field(default="", description="One or two sentences tied to the evidence.")
    evidence_ids: list[str] = Field(
        default_factory=list, description="Ids of the evidence items relied on."
    )

    @field_validator("status", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        if isinstance(value, str):
            cleaned = value.strip().lower()
            return cleaned if cleaned in {s.value for s in ClaimStatus} else ClaimStatus.INSUFFICIENT
        return value


class AssessOutput(StageOutput):
    """Judgement of every claim against the collected evidence."""

    assessments: list[ClaimAssessment] = Field(default_factory=list)


class VerdictOutput(StageOutput):
    """The final answer shown to the user."""

    label: VerdictLabel = Field(
        default=VerdictLabel.UNVERIFIED, description="true, false, mixed or unverified."
    )
    confidence: float = Field(default=0.0, description="Between 0 and 1.")
    summary: str = Field(
        default="", description="Markdown answer citing sources as [1], [2] in evidence order."
    )

    @field_validator("label", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        if isinstance(value, str):
            cleaned = value.strip().lower()
            return cleaned if cleaned in {v.value for v in VerdictLabel} else VerdictLabel.UNVERIFIED
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> Any:
        try:
            return min(max(float(value), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0


class CompactionOutput(StageOutput):
    """A rewritten running summary of the verification so far."""

    summary: str = Field(default="", description="Under 120 words, facts only.")


class DirectAnswerOutput(StageOutput):
    """The answer given when the verifier is switched off."""

    answer: str = Field(default="", description="Markdown answer to the user, no citations.")


STAGE_OUTPUTS: dict[Stage, type[StageOutput]] = {
    Stage.DECOMPOSE: DecomposeOutput,
    Stage.PLAN: PlanOutput,
    Stage.GATHER: GatherOutput,
    Stage.ASSESS: AssessOutput,
    Stage.VERDICT: VerdictOutput,
}

OutputT = TypeVar("OutputT", bound=BaseModel)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first balanced JSON object out of a model response.

    Pydantic AI already strips code fences and retries on invalid output; this
    is the last-chance salvage for replies that wrap the object in prose.
    """
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(candidate)):
            char = candidate[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    blob = candidate[start : index + 1]
                    for attempt in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
                        try:
                            parsed = json.loads(attempt)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(parsed, dict):
                            return parsed
                    break
        start = candidate.find("{", start + 1)

    raise ValueError(f"no JSON object found in model output: {text[:200]!r}")


def salvage_output(text: str, output_type: type[OutputT]) -> OutputT:
    """Recover a stage output from a reply Pydantic AI could not validate."""
    return output_type.model_validate(extract_json_object(text))

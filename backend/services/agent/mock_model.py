from __future__ import annotations

import json
import re
from typing import Any, Callable

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from backend.services.agent.prompts import CONTEXT_CLOSE, CONTEXT_OPEN

_CONTEXT_RE = re.compile(
    re.escape(CONTEXT_OPEN) + r"\s*(.*?)\s*" + re.escape(CONTEXT_CLOSE), re.DOTALL
)
_CANDIDATE_RE = re.compile(r"^- ([a-z_][a-z0-9_]*):", re.MULTILINE)

MOCK_NOTE = "mock inference — no Brev endpoint configured"


def build_mock_model() -> FunctionModel:
    """A deterministic stand-in for the Brev deployment.

    It reads the context block out of the prompt and answers with the shape the
    stage expects, so the whole workflow — including MCP tool use — runs offline
    and in tests through exactly the same code paths.
    """
    return FunctionModel(_respond, model_name="mock")


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    prompt = _user_prompt(messages)
    context = _context(prompt)
    output_name = _output_name(info)
    handler = _HANDLERS.get(output_name, _decompose)
    return ModelResponse(parts=[TextPart(json.dumps(handler(context, prompt)))])


def _user_prompt(messages: list[ModelMessage]) -> str:
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                chunks.append(part.content)
    return "\n".join(chunks)


def _output_name(info: AgentInfo) -> str:
    output_object = getattr(info.model_request_parameters, "output_object", None)
    return getattr(output_object, "name", "") or ""


def _context(prompt: str) -> dict[str, Any]:
    match = _CONTEXT_RE.search(prompt)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _decompose(context: dict[str, Any], _prompt: str) -> dict[str, Any]:
    question = str(context.get("question") or "").strip()
    if len(question.split()) < 3:
        return {
            "thought": "no checkable claim in the message",
            "claims": [],
            "reply": "Send me a factual statement and I will check it against sources.",
        }
    return {"thought": "extracted one atomic claim", "claims": [question], "reply": ""}


def _plan(context: dict[str, Any], _prompt: str) -> dict[str, Any]:
    claims = context.get("claims", [])
    return {
        "thought": "planning the evidence needed",
        "evidence_needs": [
            {"claim_id": claim["id"], "needs": "an independent primary source"} for claim in claims
        ],
        "queries": [claim["text"] for claim in claims],
    }


def _gather(context: dict[str, Any], _prompt: str) -> dict[str, Any]:
    claims = context.get("claims", [])
    budget = context.get("tool_budget_left", 0)
    if budget <= 0 or context.get("evidence") or not claims:
        return {"thought": "the evidence collected is enough", "tool_calls": [], "done": True}
    return {
        "thought": "searching for sources on the first claim",
        "tool_calls": [
            {"tool": "web_search", "arguments": {"query": claims[0]["text"], "num_results": 4}}
        ],
        "done": False,
    }


def _assess(context: dict[str, Any], _prompt: str) -> dict[str, Any]:
    evidence_ids = [item["id"] for item in context.get("evidence", [])]
    return {
        "thought": "judging each claim against the evidence",
        "assessments": [
            {
                "claim_id": claim["id"],
                "status": "supported" if evidence_ids else "insufficient",
                "rationale": (
                    "The retrieved sources are consistent with the claim."
                    if evidence_ids
                    else "No source was retrieved for this claim."
                ),
                "evidence_ids": evidence_ids[:2],
            }
            for claim in context.get("claims", [])
        ],
    }


def _verdict(context: dict[str, Any], _prompt: str) -> dict[str, Any]:
    claims = context.get("claims", [])
    supported = [c for c in claims if c.get("status") == "supported"]
    if not claims:
        label = "unverified"
    elif len(supported) == len(claims):
        label = "true"
    elif supported:
        label = "mixed"
    else:
        label = "unverified"

    lines = [f"**Verdict: {label}** ({MOCK_NOTE})", ""]
    for index, claim in enumerate(claims, start=1):
        lines.append(
            f"{index}. {claim['text']} — {claim.get('status', 'pending')}. "
            f"{claim.get('rationale', '')}".rstrip()
        )
    return {
        "thought": "writing the final answer",
        "label": label,
        "confidence": 0.4 if claims else 0.0,
        "summary": "\n".join(lines).strip(),
    }


def _curation(_context: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {"thought": "exposing the search tool only", "tools": _CANDIDATE_RE.findall(prompt)[:1]}


def _compaction(context: dict[str, Any], _prompt: str) -> dict[str, Any]:
    statuses = "; ".join(
        f"{claim['text'][:60]} -> {claim.get('status', 'pending')}"
        for claim in context.get("claims", [])
    )
    question = context.get("question", "a claim")
    return {
        "thought": "compacting the turn",
        "summary": f"The user asked to verify: {question}. Outcome: {statuses}".strip(),
    }


_HANDLERS: dict[str, Callable[[dict[str, Any], str], dict[str, Any]]] = {
    "DecomposeOutput": _decompose,
    "PlanOutput": _plan,
    "GatherOutput": _gather,
    "AssessOutput": _assess,
    "VerdictOutput": _verdict,
    "CurationOutput": _curation,
    "CompactionOutput": _compaction,
}

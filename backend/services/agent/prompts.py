from __future__ import annotations

import json
from typing import Any

from backend.models.schemas.context import Stage

CONTEXT_OPEN = "[CONTEXT_JSON]"
CONTEXT_CLOSE = "[/CONTEXT_JSON]"

# Gemma's chat template only tolerates a single leading system message, and
# Pydantic AI already spends it on the output schema. So the role and the
# memory file travel at the top of the user prompt instead of as instructions.
VERIFIER_ROLE = """You are the Verifier, the reasoning agent of a claim-verification system.
You work one decision point at a time. A separate Context agent gives you the context
object below, which holds everything you are allowed to reason from.

Hard rules:
- Do not invent sources, titles, quotes or dates.
- If the available context is thin, say so instead of guessing."""

CONTEXT_ROLE = """You are the Context agent of a claim-verification system.
You do not answer the user and you do not decide anything about the claims. You keep the
running context of the verification compact, factual and useful to the Verifier."""

DIRECT_ROLE = """You are a helpful assistant answering the user directly.
The verification pipeline is switched off for this turn: there is no context object, no retrieved
evidence and no verdict, so answer from what you already know.

Hard rules:
- Never invent sources, titles, quotes or dates, and never imply anything was checked.
- Say plainly when you are unsure or when a claim would need a source to settle."""

DIRECT_HISTORY_HEADER = "Earlier in this conversation:"
DIRECT_QUESTION_HEADER = "User message:"

STAGE_INSTRUCTIONS: dict[Stage, str] = {
    Stage.DECOMPOSE: """Split the user's message into atomic, independently checkable factual claims.
Keep at most {max_claims} claims, drop rhetoric and opinion, and resolve pronouns using the running summary.
If the message contains no checkable factual claim (a greeting, small talk, a meta question),
return an empty claim list and put a short direct answer in "reply" instead.""",
    Stage.PLAN: """For each claim, state what information would settle it.
Reference claims by the ids given in the context object.""",
    Stage.ASSESS: """Judge every claim in the context against the evidence in the context, and nothing else.
Use "supported" when credible evidence confirms it, "refuted" when credible evidence contradicts it,
and "insufficient" when the evidence is missing, off-topic or conflicting.
Reference claims and evidence by their ids. Produce exactly one assessment per claim.""",
    Stage.VERDICT: """Write the final answer for the user.
The summary is markdown: one bold verdict line, then a short paragraph per claim, citing sources as
[1], [2] in the order they appear in the evidence list. State plainly what could not be verified.
Set the label from the claim statuses and keep the confidence honest.""",
}

COMPACTION_INSTRUCTION = """Rewrite the running summary of this verification so far.
Keep it under 120 words: what was asked, which claims are settled, what the evidence establishes.
Facts only, no speculation, no citation markers."""

DIRECT_INSTRUCTION = """Answer the user's message in markdown, in a few short paragraphs at most.
Do not decompose it into claims, do not cite anything, and do not state a verdict."""


def render_context_block(view: dict[str, Any]) -> str:
    return f"{CONTEXT_OPEN}\n{json.dumps(view, ensure_ascii=False, indent=2)}\n{CONTEXT_CLOSE}"


def build_stage_prompt(
    *,
    stage: Stage,
    context_view: dict[str, Any],
    max_claims: int,
    memory: str = "",
) -> str:
    role = f"{VERIFIER_ROLE}\n\n---\n{memory}" if memory else VERIFIER_ROLE
    return "\n\n".join(
        [
            role,
            f"Stage: {stage.value}",
            render_context_block(context_view),
            f"TASK\n{STAGE_INSTRUCTIONS[stage].format(max_claims=max_claims)}",
        ]
    )


def build_direct_prompt(*, question: str, history: str = "") -> str:
    blocks = [DIRECT_ROLE]
    if history:
        blocks.append(f"{DIRECT_HISTORY_HEADER}\n{history}")
    blocks.append(f"{DIRECT_QUESTION_HEADER}\n{question}")
    blocks.append(f"TASK\n{DIRECT_INSTRUCTION}")
    return "\n\n".join(blocks)


def build_compaction_prompt(*, stage: Stage, context_view: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            CONTEXT_ROLE,
            f"Stage: {stage.value}",
            render_context_block(context_view),
            f"TASK\n{COMPACTION_INSTRUCTION}",
        ]
    )

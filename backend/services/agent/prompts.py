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
- Never invent sources, titles, quotes or dates. Only cite evidence present in the context.
- Evidence comes from semantic search over a private document corpus, not the open web.
- If the evidence is thin, say so instead of guessing."""

CONTEXT_ROLE = """You are the Context agent of a claim-verification system.
You do not answer the user and you do not decide anything about the claims. You keep the
running context of the verification compact, factual and useful to the Verifier."""

STAGE_INSTRUCTIONS: dict[Stage, str] = {
    Stage.DECOMPOSE: """Split the user's message into atomic, independently checkable factual claims.
Keep at most {max_claims} claims, drop rhetoric and opinion, and resolve pronouns using the running summary.
If the message contains no checkable factual claim (a greeting, small talk, a meta question),
return an empty claim list and put a short direct answer in "reply" instead.""",
    Stage.PLAN: """For each claim, state what evidence would settle it, and write the search queries you would run.
Reference claims by the ids given in the context object.
Queries must be specific, phrased the way a source would phrase it, and free of hedging words.""",
    Stage.GATHER: """Write the searches to run next against the document corpus.
Each query is matched semantically against passages, so phrase it as the passage that would
answer it would be phrased, not as a question or a keyword soup.
Ask for a query only if it adds something the current evidence does not already cover, and
never repeat anything in queries_already_run.
Return an empty queries list with done=true as soon as the evidence is sufficient,
or when searches_left reaches zero.""",
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


def build_compaction_prompt(*, stage: Stage, context_view: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            CONTEXT_ROLE,
            f"Stage: {stage.value}",
            render_context_block(context_view),
            f"TASK\n{COMPACTION_INSTRUCTION}",
        ]
    )

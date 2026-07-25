from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from backend.config import Settings
from backend.models.schemas.actions import STAGE_OUTPUTS
from backend.models.schemas.context import ContextObject, Decision, Stage, ToolCall, ToolSpec
from backend.services.agent.llm_client import Inference, LLMError
from backend.services.agent.prompts import build_stage_prompt

logger = logging.getLogger(__name__)

STAGE_TEMPERATURE: dict[Stage, float] = {
    Stage.DECOMPOSE: 0.1,
    Stage.PLAN: 0.2,
    Stage.GATHER: 0.2,
    Stage.ASSESS: 0.1,
    Stage.VERDICT: 0.3,
}


@lru_cache(maxsize=4)
def load_memory(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("Memory file %s unreadable (%s); running without it", path, exc)
        return ""


class VerifierAgent:
    """Agent 1: decides what to do at a single decision point.

    It sees three things and nothing else: the durable memory file, the context
    object agent 2 assembled, and the tools agent 2 curated for this step. The
    reply is validated against the stage's output schema before it comes back.
    """

    def __init__(self, inference: Inference, settings: Settings) -> None:
        self._inference = inference
        self._settings = settings

    @property
    def memory(self) -> str:
        return load_memory(self._settings.memory_path)

    async def run(self, *, context: ContextObject, tools: list[ToolSpec]) -> Decision:
        output_type = STAGE_OUTPUTS[context.stage]
        prompt = build_stage_prompt(
            stage=context.stage,
            context_view=context.prompt_view(self._settings.max_evidence_in_context),
            tools=tools,
            max_claims=self._settings.max_claims,
            memory=self.memory,
        )
        allowed = {tool.name for tool in tools}

        try:
            output = await self._inference.run(
                prompt,
                output_type,
                temperature=STAGE_TEMPERATURE.get(context.stage, 0.2),
                max_tokens=1600,
            )
        except LLMError as exc:
            logger.error("Verifier failed at stage %s: %s", context.stage.value, exc)
            return Decision(
                stage=context.stage,
                summary=f"inference failed: {exc}",
                curated_tools=sorted(allowed),
                output={"error": str(exc)},
                done=True,
            )

        payload = output.model_dump(mode="json")
        thought = str(payload.pop("thought", ""))
        requested = payload.pop("tool_calls", []) or []

        tool_calls: list[ToolCall] = []
        rejected: list[str] = []
        for call in requested:
            if call["tool"] in allowed:
                tool_calls.append(ToolCall(tool=call["tool"], arguments=call.get("arguments") or {}))
            else:
                rejected.append(call["tool"])
        if rejected:
            logger.info("Dropped tool calls outside the curated set: %s", rejected)
            payload["rejected_tools"] = rejected

        return Decision(
            stage=context.stage,
            summary=thought,
            tool_calls=tool_calls,
            output=payload,
            curated_tools=sorted(allowed),
            done=bool(payload.get("done", True)) and not tool_calls,
        )

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from backend.config import Settings
from backend.models.schemas.actions import STAGE_OUTPUTS
from backend.models.schemas.context import ContextObject, Decision, Stage
from backend.services.agent.llm_client import Inference, LLMError
from backend.services.agent.prompts import build_stage_prompt

logger = logging.getLogger(__name__)

STAGE_TEMPERATURE: dict[Stage, float] = {
    Stage.DECOMPOSE: 0.1,
    Stage.PLAN: 0.2,
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

    It sees two things and nothing else: the durable memory file and the
    context object agent 2 assembled. The reply is validated against the
    stage's output schema before it comes back.
    """

    def __init__(self, inference: Inference, settings: Settings) -> None:
        self._inference = inference
        self._settings = settings

    @property
    def memory(self) -> str:
        return load_memory(self._settings.memory_path)

    async def run(self, *, context: ContextObject) -> Decision:
        output_type = STAGE_OUTPUTS[context.stage]
        prompt = build_stage_prompt(
            stage=context.stage,
            context_view=context.prompt_view(self._settings.max_evidence_in_context),
            max_claims=self._settings.max_claims,
            memory=self.memory,
        )

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
                output={"error": str(exc)},
                done=True,
            )

        payload = output.model_dump(mode="json")
        thought = str(payload.pop("thought", ""))

        return Decision(
            stage=context.stage,
            summary=thought,
            queries=[],
            output=payload,
            done=True,
        )

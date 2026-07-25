from __future__ import annotations

import logging

from backend.config import Settings
from backend.models.schemas.actions import DirectAnswerOutput
from backend.services.agent.llm_client import Inference, LLMError
from backend.services.agent.prompts import build_direct_prompt

logger = logging.getLogger(__name__)

EMPTY_ANSWER = "The model returned nothing for that message."


class DirectAgent:
    """What the app is when the verifier is switched off: one plain model call.

    No context object, no claims, no retrieval and no verdict — the same model
    answering from what it already knows, which is the baseline the verified
    path is meant to be compared against.
    """

    def __init__(self, inference: Inference, settings: Settings) -> None:
        self._inference = inference
        self._settings = settings

    async def run(self, *, question: str, history: str = "") -> str:
        prompt = build_direct_prompt(question=question, history=history)
        try:
            output = await self._inference.run(
                prompt, DirectAnswerOutput, temperature=0.4, max_tokens=900
            )
        except LLMError as exc:
            logger.error("Direct answer failed: %s", exc)
            return f"The answer could not be generated: {exc}"
        return output.answer.strip() or EMPTY_ANSWER

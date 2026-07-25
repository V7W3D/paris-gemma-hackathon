from __future__ import annotations

import asyncio
import logging
from typing import TypeVar

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, ModelHTTPError, PromptedOutput, capture_run_messages
from pydantic_ai.exceptions import AgentRunError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from backend.config import Settings
from backend.models.schemas.actions import salvage_output

logger = logging.getLogger(__name__)

OutputT = TypeVar("OutputT", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when inference cannot produce a usable structured answer."""


def normalise_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def build_model(settings: Settings) -> Model:
    """Point Pydantic AI at the OpenAI-compatible endpoint, or the offline mock."""
    if settings.llm_is_mocked:
        from backend.services.agent.mock_model import build_mock_model

        logger.warning(
            "VLLM_BASE_URL / BREV_BASE_URL not set (or MOCK_LLM=true): using the offline mock model"
        )
        return build_mock_model()

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(
        base_url=normalise_base_url(settings.brev_base_url),
        api_key=settings.brev_api_key or "not-required",
    )
    return OpenAIChatModel(settings.brev_model, provider=provider)


class Inference:
    """One prompted-output call against the model, with salvage and retries.

    Both agents share this. Prompted output is deliberate: the schema is
    injected into the prompt and validated on the way back, which is the only
    reliable way to get structure out of a model without native tool-calling.
    """

    def __init__(self, settings: Settings, model: Model | None = None) -> None:
        self._settings = settings
        self.model = model or build_model(settings)
        self._agent: Agent[None, str] = Agent(
            self.model,
            retries=settings.llm_max_retries,
            # No instructions: Pydantic AI spends the single system message the
            # Gemma chat template allows on the output schema.
        )

    @property
    def mocked(self) -> bool:
        return self.model.system == "function"

    async def run(
        self,
        prompt: str,
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1600,
    ) -> OutputT:
        settings = ModelSettings(
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self._settings.llm_timeout_seconds,
        )
        attempts = self._settings.llm_max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            with capture_run_messages() as messages:
                try:
                    result = await self._agent.run(
                        prompt,
                        output_type=PromptedOutput(output_type),
                        model_settings=settings,
                    )
                    return result.output
                except UnexpectedModelBehavior as exc:
                    salvaged = _salvage(messages, output_type)
                    if salvaged is not None:
                        logger.info("Salvaged %s from a malformed reply", output_type.__name__)
                        return salvaged
                    last_error = exc
                except ModelHTTPError as exc:
                    last_error = exc
                    if exc.status_code < 500:
                        break
                except (AgentRunError, httpx.HTTPError) as exc:
                    last_error = exc

            if attempt < attempts - 1:
                await asyncio.sleep(0.75 * (2**attempt))

        raise LLMError(f"inference failed for {output_type.__name__}: {last_error}")

    async def aclose(self) -> None:
        client = getattr(getattr(self.model, "client", None), "_client", None)
        if isinstance(client, httpx.AsyncClient):
            await client.aclose()


def _salvage(messages: list[ModelMessage], output_type: type[OutputT]) -> OutputT | None:
    for message in reversed(messages):
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, TextPart) and part.content:
                try:
                    return salvage_output(part.content, output_type)
                except (ValueError, TypeError):
                    continue
    return None

from __future__ import annotations

import pytest

from backend.config import Settings
from backend.db.store import MemoryStore
from backend.services.agent.context_agent import ContextAgent
from backend.services.agent.llm_client import Inference
from backend.services.agent.verifier_agent import VerifierAgent
from backend.services.mcp.client import MCPToolClient
from backend.services.workflow.claim_verification import ClaimVerificationWorkflow


@pytest.fixture
def settings() -> Settings:
    return Settings(
        mock_llm=True,
        mock_search=True,
        brev_base_url="",
        serpapi_api_key="",
        mcp_url="",
        max_gather_steps=2,
    )


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def inference(settings: Settings) -> Inference:
    return Inference(settings)


@pytest.fixture
def context_agent(inference: Inference, store: MemoryStore, settings: Settings) -> ContextAgent:
    return ContextAgent(inference, store, settings)


@pytest.fixture
def verifier(inference: Inference, settings: Settings) -> VerifierAgent:
    return VerifierAgent(inference, settings)


@pytest.fixture
def tools(settings: Settings) -> MCPToolClient:
    return MCPToolClient(settings)


@pytest.fixture
def workflow(
    verifier: VerifierAgent,
    context_agent: ContextAgent,
    tools: MCPToolClient,
    store: MemoryStore,
    settings: Settings,
) -> ClaimVerificationWorkflow:
    return ClaimVerificationWorkflow(
        verifier=verifier,
        context_agent=context_agent,
        tools=tools,
        store=store,
        settings=settings,
    )

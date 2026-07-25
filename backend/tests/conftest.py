from __future__ import annotations

import pytest

from backend.config import Settings
from backend.db.store import MemoryStore
from backend.services.agent.context_agent import ContextAgent
from backend.services.agent.llm_client import Inference
from backend.services.agent.verifier_agent import VerifierAgent
from backend.services.retrieval.alien_client import AlienRetriever
from backend.services.workflow.claim_verification import ClaimVerificationWorkflow


@pytest.fixture
def settings() -> Settings:
    return Settings(
        mock_llm=True,
        mock_search=True,
        brev_base_url="",
        alien_mcp_url="",
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
def retriever(settings: Settings) -> AlienRetriever:
    return AlienRetriever(settings)


@pytest.fixture
def workflow(
    verifier: VerifierAgent,
    context_agent: ContextAgent,
    retriever: AlienRetriever,
    store: MemoryStore,
    settings: Settings,
) -> ClaimVerificationWorkflow:
    return ClaimVerificationWorkflow(
        verifier=verifier,
        context_agent=context_agent,
        retriever=retriever,
        store=store,
        settings=settings,
    )

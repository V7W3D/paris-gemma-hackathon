from __future__ import annotations

import pytest

from backend.config import Settings
from backend.db.store import MemoryStore
from backend.services.agent.context_agent import ContextAgent
from backend.services.agent.direct_agent import DirectAgent
from backend.services.agent.llm_client import Inference
from backend.services.agent.verifier_agent import VerifierAgent
from backend.services.workflow.claim_verification import ClaimVerificationWorkflow


@pytest.fixture
def settings() -> Settings:
    # _env_file=None keeps the suite hermetic: a filled-in backend/.env, with a
    # live corpus and a live model behind it, must not change what is tested.
    return Settings(
        _env_file=None,
        mock_llm=True,
        brev_base_url="",
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
def direct_agent(inference: Inference, settings: Settings) -> DirectAgent:
    return DirectAgent(inference, settings)


@pytest.fixture
def workflow(
    verifier: VerifierAgent,
    context_agent: ContextAgent,
    direct_agent: DirectAgent,
    store: MemoryStore,
    settings: Settings,
) -> ClaimVerificationWorkflow:
    return ClaimVerificationWorkflow(
        verifier=verifier,
        context_agent=context_agent,
        direct_agent=direct_agent,
        store=store,
        settings=settings,
    )

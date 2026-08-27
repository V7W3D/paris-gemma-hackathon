from __future__ import annotations

from fastapi import Request

from backend.config import Settings, get_settings
from backend.db.mongo import Database
from backend.db.store import Store
from backend.services.agent.context_agent import ContextAgent
from backend.services.agent.direct_agent import DirectAgent
from backend.services.agent.llm_client import Inference
from backend.services.agent.verifier_agent import VerifierAgent
from backend.services.workflow.claim_verification import ClaimVerificationWorkflow


class AppContainer:
    """Wires the agents and store together."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.database = Database(self.settings)
        self.inference = Inference(self.settings)
        self.store: Store = self.database.store
        self.context_agent = ContextAgent(self.inference, self.store, self.settings)
        self.verifier = VerifierAgent(self.inference, self.settings)
        self.direct_agent = DirectAgent(self.inference, self.settings)
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> ClaimVerificationWorkflow:
        return ClaimVerificationWorkflow(
            verifier=self.verifier,
            context_agent=self.context_agent,
            direct_agent=self.direct_agent,
            store=self.store,
            settings=self.settings,
        )

    async def startup(self) -> None:
        self.store = await self.database.connect()
        self.context_agent = ContextAgent(self.inference, self.store, self.settings)
        self.workflow = self._build_workflow()
    async def shutdown(self) -> None:
        await self.inference.aclose()
        await self.database.close()

    def status(self) -> dict[str, object]:
        return {
            "mongo_connected": self.database.connected,
            "llm_mocked": self.inference.mocked,
            "llm_model": self.settings.brev_model,
        }


def get_container(request: Request) -> AppContainer:
    return request.app.state.container

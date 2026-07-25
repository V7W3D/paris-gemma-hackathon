from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(ROOT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    brev_base_url: str = ""
    brev_api_key: str = ""
    brev_model: str = "google/gemma-3-27b-it"
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "claim_verifier"
    mongodb_timeout_ms: int = 2000

    alien_mcp_url: str = ""
    alien_mcp_token: str = ""
    alien_mcp_search_tool: str = ""
    alien_dataset_ids: list[int] = []
    alien_search_limit: int = 8
    alien_mcp_timeout_seconds: float = 30.0

    mock_llm: bool = False
    mock_search: bool = False

    max_gather_steps: int = 3
    max_claims: int = 4
    max_evidence_in_context: int = 12

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def memory_path(self) -> Path:
        return BACKEND_DIR / "memory" / "verifier_memory.md"

    @property
    def llm_is_mocked(self) -> bool:
        """Mock whenever explicitly asked, or when no Brev endpoint is configured."""
        return self.mock_llm or not self.brev_base_url

    @property
    def search_is_mocked(self) -> bool:
        return self.mock_search or not self.alien_mcp_url


@lru_cache
def get_settings() -> Settings:
    return Settings()

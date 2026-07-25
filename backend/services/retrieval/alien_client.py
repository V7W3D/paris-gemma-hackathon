from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from backend.config import Settings
from backend.models.schemas.context import Chunk

logger = logging.getLogger(__name__)

# The Alien MCP server advertises its own tool names and schemas, so nothing
# here is hardcoded to one deployment: the search tool and its arguments are
# resolved against whatever the server reports at startup. Set
# ALIEN_MCP_SEARCH_TOOL to pin a specific tool.
QUERY_ARGS = ("query", "q", "question", "text", "search", "prompt")
LIMIT_ARGS = ("limit", "top_k", "k", "num_results", "max_results")
DATASET_ARGS = ("dataset_ids", "datasets", "dataset_id")

TITLE_FIELDS = ("title", "name", "entry_name", "document", "filename")
TEXT_FIELDS = ("text", "chunk_text", "content", "snippet", "passage", "excerpt", "body")
URL_FIELDS = ("url", "link", "uri", "download_url", "source_url")
SOURCE_FIELDS = ("source", "dataset_name", "dataset", "collection")
RESULT_KEYS = ("results", "hits", "chunks", "matches", "items", "data")


class RetrievalError(RuntimeError):
    """A search against the Alien MCP server failed."""


class RetrievalUnavailableError(RetrievalError):
    """The Alien MCP server could not be reached."""


class AlienRetriever:
    """Semantic search over the Alien corpus, over MCP.

    The gather stage is the only caller: it passes the queries agent 1 asked
    for and gets back passages. Agent 1 never sees a tool catalogue — MCP is
    the transport to one search tool, not a surface the model chooses from.
    With no server configured the retriever answers with deterministic
    placeholders, so the workflow still runs end to end offline.
    """

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client
        self._tools: list[Any] = []
        self._search_tool: Any = None

    @property
    def mocked(self) -> bool:
        return self._settings.search_is_mocked

    @property
    def endpoint(self) -> str:
        return self._settings.alien_mcp_url or "mock"

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self._tools]

    @property
    def search_tool(self) -> str:
        return getattr(self._search_tool, "name", "") or "mock_search"

    @property
    def dataset_ids(self) -> list[int]:
        return list(self._settings.alien_dataset_ids)

    # ----------------------------------------------------------------- client

    def _ensure_client(self) -> Any:
        if self._client is None:
            from fastmcp import Client

            self._client = Client(
                self._settings.alien_mcp_url,
                auth=self._settings.alien_mcp_token or None,
                timeout=self._settings.alien_mcp_timeout_seconds,
            )
        return self._client

    async def connect(self) -> None:
        """List the server's tools once at startup and resolve the search tool."""
        if self.mocked:
            logger.info("Alien retrieval is mocked: no MCP server configured")
            return

        client = self._ensure_client()
        try:
            async with client:
                self._tools = list(await client.list_tools())
        except Exception as exc:  # noqa: BLE001 - transport and auth errors alike
            raise RetrievalUnavailableError(
                f"cannot reach the Alien MCP server at {self.endpoint}: {exc}"
            ) from exc

        self._search_tool = self._select_search_tool()
        if self._search_tool is None:
            raise RetrievalUnavailableError(
                f"no search tool found on {self.endpoint}; advertised tools: "
                f"{', '.join(self.tool_names) or 'none'}. "
                "Set ALIEN_MCP_SEARCH_TOOL to pick one explicitly."
            )
        logger.info(
            "Alien MCP ready on %s: searching with %r out of %s",
            self.endpoint,
            self.search_tool,
            self.tool_names,
        )

    def _select_search_tool(self) -> Any | None:
        pinned = self._settings.alien_mcp_search_tool
        if pinned:
            return next((tool for tool in self._tools if tool.name == pinned), None)
        ranked = sorted(self._tools, key=lambda tool: _rank_tool(tool.name), reverse=True)
        best = ranked[0] if ranked else None
        return best if best is not None and _rank_tool(best.name) > 0 else None

    async def aclose(self) -> None:
        self._client = None
        self._tools = []
        self._search_tool = None

    # ----------------------------------------------------------------- search

    async def search(self, query: str, *, limit: int | None = None) -> list[Chunk]:
        """Run one search against the corpus and return the passages it found."""
        query = query.strip()
        if not query:
            return []
        size = limit or self._settings.alien_search_limit
        if self.mocked:
            return _mock_chunks(query, size)
        if self._search_tool is None:
            raise RetrievalUnavailableError(f"no search tool resolved on {self.endpoint}")

        client = self._ensure_client()
        arguments = self._arguments(query, size)
        try:
            async with client:
                result = await client.call_tool(
                    self._search_tool.name, arguments, raise_on_error=False
                )
        except Exception as exc:  # noqa: BLE001 - reported to the agent as a failed search
            logger.warning("Alien search for %r failed: %s", query, exc)
            raise RetrievalError(str(exc)) from exc

        if getattr(result, "is_error", False):
            detail = _text_of(result.content[0]) if result.content else "the tool reported an error"
            raise RetrievalError(detail)

        payload = result.data
        if payload is None and result.content:
            payload = _text_of(result.content[0])
        return _to_chunks(payload)[:size]

    def _arguments(self, query: str, size: int) -> dict[str, Any]:
        """Fill the search tool's own argument names from its advertised schema."""
        schema = getattr(self._search_tool, "inputSchema", None) or {}
        properties: dict[str, Any] = schema.get("properties") or {}
        required: list[str] = schema.get("required") or []

        query_arg = _first_present(properties, QUERY_ARGS)
        if query_arg is None:
            query_arg = next(
                (
                    name
                    for name in required
                    if (properties.get(name) or {}).get("type") in (None, "string")
                ),
                "query",
            )
        arguments: dict[str, Any] = {query_arg: query}

        limit_arg = _first_present(properties, LIMIT_ARGS)
        if limit_arg is not None:
            arguments[limit_arg] = size

        dataset_arg = _first_present(properties, DATASET_ARGS)
        if dataset_arg is not None and self.dataset_ids:
            arguments[dataset_arg] = (
                self.dataset_ids
                if _accepts_array(properties.get(dataset_arg) or {})
                else self.dataset_ids[0]
            )
        return arguments


def _rank_tool(name: str) -> int:
    """Guess how likely a tool name is to be the corpus search."""
    lowered = name.lower()
    if "search" not in lowered and "quer" not in lowered:
        return 0
    score = 1
    if any(hint in lowered for hint in ("vector", "semantic", "chunk", "passage")):
        score += 2
    if any(hint in lowered for hint in ("dataset", "list", "catalog")):
        score -= 2
    return score


def _first_present(properties: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in properties), None)


def _accepts_array(spec: dict[str, Any]) -> bool:
    """An optional list argument hides inside anyOf, sometimes nested twice over."""
    if spec.get("type") == "array" or "items" in spec:
        return True
    branches = [*(spec.get("anyOf") or []), *(spec.get("oneOf") or [])]
    return any(isinstance(branch, dict) and _accepts_array(branch) for branch in branches)


def _text_of(block: Any) -> str:
    return str(getattr(block, "text", "") or "")


def _to_chunks(payload: Any) -> list[Chunk]:
    """Normalise whatever the tool returned into passages.

    MCP tools are free to answer with a list, an envelope around a list, or one
    blob of text, so the shape is discovered rather than assumed.
    """
    if payload is None:
        return []
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return [Chunk(text=payload[:1500])] if payload.strip() else []

    if isinstance(payload, dict):
        for key in RESULT_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return [_to_chunk(item) for item in value]
        return [_to_chunk(payload)]
    if isinstance(payload, list):
        return [_to_chunk(item) for item in payload]
    return [Chunk(text=str(payload)[:1500])]


def _to_chunk(item: Any) -> Chunk:
    if not isinstance(item, dict):
        return Chunk(text=str(item)[:1500])
    return Chunk(
        title=_field(item, TITLE_FIELDS)[:300],
        text=_field(item, TEXT_FIELDS)[:1500] or json.dumps(item, ensure_ascii=False)[:1500],
        url=_field(item, URL_FIELDS),
        source=_field(item, SOURCE_FIELDS),
        score=_score(item),
    )


def _field(item: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for name in candidates:
        value = item.get(name)
        if value not in (None, "", [], {}):
            return str(value)
    # Metadata-carrying tools often nest the descriptive fields one level down.
    metadata = item.get("metadata") or item.get("payload")
    if isinstance(metadata, dict):
        for name in candidates:
            value = metadata.get(name)
            if value not in (None, "", [], {}):
                return str(value)
    return ""


def _score(item: dict[str, Any]) -> float:
    for name in ("score", "similarity", "relevance", "_score"):
        if name in item:
            try:
                return min(max(float(item[name]), 0.0), 1.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _mock_chunks(query: str, limit: int) -> list[Chunk]:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
    datasets = ["regulatory-filings", "press-archive", "research-corpus"]
    return [
        Chunk(
            title=f"{query[:70]} — passage {index + 1}",
            text=(
                f"Offline placeholder passage {index + 1} for '{query[:80]}'. "
                "Set ALIEN_MCP_URL to search the real corpus."
            ),
            url=f"https://mock.alien.local/{digest}/{index + 1}",
            source=datasets[index % len(datasets)],
            score=round(0.9 - index * 0.1, 2),
        )
        for index in range(max(1, min(limit, 3)))
    ]

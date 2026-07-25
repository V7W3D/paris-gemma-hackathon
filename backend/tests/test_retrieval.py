from __future__ import annotations

from typing import Annotated, Any

import pytest
from fastmcp import Client, FastMCP
from pydantic import Field

from backend.config import Settings
from backend.services.retrieval.alien_client import (
    AlienRetriever,
    RetrievalError,
    RetrievalUnavailableError,
)

HITS = [
    {
        "entry_name": "Tower maintenance report",
        "chunk_text": "The tower measures 330 metres including its antennas.",
        "url": "https://corpus.example/entries/1",
        "dataset_name": "press-archive",
        "score": 0.82,
    },
    {
        "entry_name": "Structural survey",
        "chunk_text": "Height to tip: 330 m.",
        "url": "https://corpus.example/entries/2",
        "dataset_name": "research-corpus",
        "score": 0.61,
    },
]


def _live(**overrides) -> Settings:
    return Settings(**{"alien_mcp_url": "http://alien.test/mcp", "mock_search": False, **overrides})


def _server(hits: Any = None, *, fail: bool = False) -> FastMCP:
    """An MCP server shaped like Alien's: a search tool beside other data tools."""
    server = FastMCP("alien-test")

    @server.tool
    async def list_datasets() -> dict[str, Any]:
        """List the datasets on the cluster."""
        return {"datasets": [{"id": 1, "name": "press-archive"}]}

    @server.tool
    async def vector_search_chunks(
        query: Annotated[str, Field(description="Semantic query")],
        limit: Annotated[int, Field(description="How many passages")] = 10,
        dataset_ids: Annotated[list[int] | None, Field(description="Datasets")] = None,
    ) -> dict[str, Any]:
        """Search document passages semantically."""
        if fail:
            raise ValueError("cluster offline")
        return {"results": HITS if hits is None else hits, "asked_for": limit, "in": dataset_ids}

    return server


def _retriever(server: FastMCP, **overrides) -> AlienRetriever:
    return AlienRetriever(_live(**overrides), client=Client(server))


async def test_the_search_tool_is_picked_out_of_the_advertised_tools():
    retriever = _retriever(_server())
    await retriever.connect()

    assert set(retriever.tool_names) == {"list_datasets", "vector_search_chunks"}
    assert retriever.search_tool == "vector_search_chunks"


async def test_a_pinned_tool_wins_over_the_guess():
    retriever = _retriever(_server(), alien_mcp_search_tool="list_datasets")
    await retriever.connect()

    assert retriever.search_tool == "list_datasets"


async def test_a_pinned_tool_that_is_absent_is_reported():
    retriever = _retriever(_server(), alien_mcp_search_tool="nope")

    with pytest.raises(RetrievalUnavailableError, match="no search tool"):
        await retriever.connect()


async def test_hits_become_passages():
    retriever = _retriever(_server())
    await retriever.connect()

    chunks = await retriever.search("height of the tower")

    assert [chunk.title for chunk in chunks] == [
        "Tower maintenance report",
        "Structural survey",
    ]
    assert chunks[0].text.startswith("The tower measures 330 metres")
    assert chunks[0].url == "https://corpus.example/entries/1"
    assert chunks[0].source == "press-archive"
    assert chunks[0].score == 0.82


async def test_arguments_are_filled_from_the_tool_schema():
    retriever = _retriever(_server(hits=[]), alien_dataset_ids=[7, 9], alien_search_limit=4)
    await retriever.connect()

    arguments = retriever._arguments("a query", 4)

    assert arguments == {"query": "a query", "limit": 4, "dataset_ids": [7, 9]}


async def test_the_limit_caps_the_passages_returned():
    retriever = _retriever(_server())
    await retriever.connect()

    assert len(await retriever.search("height of the tower", limit=1)) == 1


async def test_a_tool_answering_with_plain_text_still_yields_a_passage():
    server = FastMCP("alien-text")

    @server.tool
    async def search_corpus(query: str) -> str:
        """Search the corpus."""
        return f"one long passage about {query}"

    retriever = AlienRetriever(_live(), client=Client(server))
    await retriever.connect()
    chunks = await retriever.search("towers")

    assert len(chunks) == 1
    assert "one long passage about towers" in chunks[0].text


async def test_a_failing_tool_raises_a_retrieval_error():
    retriever = _retriever(_server(fail=True))
    await retriever.connect()

    with pytest.raises(RetrievalError):
        await retriever.search("height of the tower")


async def test_an_unreachable_server_is_reported_at_startup():
    retriever = AlienRetriever(_live())

    with pytest.raises(RetrievalUnavailableError, match="cannot reach"):
        await retriever.connect()


async def test_mocked_search_returns_deterministic_passages(retriever: AlienRetriever):
    assert retriever.mocked is True
    assert retriever.endpoint == "mock"

    chunks = await retriever.search("the Eiffel Tower is 330 metres tall")

    assert chunks == await retriever.search("the Eiffel Tower is 330 metres tall")
    assert all(chunk.text for chunk in chunks)


async def test_an_empty_query_is_not_searched(retriever: AlienRetriever):
    assert await retriever.search("   ") == []


def test_retrieval_is_mocked_until_an_mcp_server_is_configured():
    assert Settings().search_is_mocked is True
    assert _live().search_is_mocked is False
    assert _live(mock_search=True).search_is_mocked is True

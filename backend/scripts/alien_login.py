"""Log in to the Alien MCP server once, and prove the search tool works.

    python -m backend.scripts.alien_login ["a probe query"]

With ALIEN_MCP_OAUTH=true this opens a browser for the Alien login, then caches
the tokens under backend/.oauth. Every later connection — including the app's
own startup — reuses them, so this is a one-off per machine.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from backend.config import get_settings
from backend.services.retrieval.alien_client import AlienRetriever, RetrievalError

DEFAULT_QUERY = "CRISPR base editing corrects a pathogenic point mutation in human cells"

# Long enough to log in by hand; the app itself keeps its own short timeout.
LOGIN_TIMEOUT_SECONDS = 300.0


async def main(query: str) -> int:
    # force=True: the libraries install a wrapping rich handler, and a
    # half-printed authorization URL is not much use to anyone.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    settings = get_settings().model_copy(
        update={"alien_mcp_timeout_seconds": LOGIN_TIMEOUT_SECONDS}
    )

    if not settings.alien_mcp_url:
        print("ALIEN_MCP_URL is empty: set it in backend/.env before logging in.")
        return 1
    if settings.mock_search:
        print("MOCK_SEARCH=true: unset it to talk to the real server.")
        return 1

    retriever = AlienRetriever(settings)
    print(f"Connecting to {retriever.endpoint} (auth: {retriever.auth_mode})")
    if retriever.auth_mode == "oauth":
        print(
            "A browser will open for the Alien login. If it does not, copy the "
            "authorization URL printed below and open it yourself.\n"
        )

    try:
        await retriever.connect()
        print(f"Tools advertised: {', '.join(retriever.tool_names) or 'none'}")
        print(f"Searching with:   {retriever.search_tool}")
        print(f"Probe query:      {query}")
        chunks = await retriever.search(query)
    except RetrievalError as exc:
        print(f"\nFailed: {exc}")
        return 1
    finally:
        await retriever.aclose()

    if not chunks:
        print("\nThe search returned no passages. The connection works; the query found nothing.")
        return 0

    print(f"\n{len(chunks)} passages:")
    for index, chunk in enumerate(chunks, start=1):
        print(f"\n{index}. {chunk.title or '(untitled)'}  [{chunk.source or '-'}] {chunk.score}")
        print(f"   {chunk.url or '(no url)'}")
        print(f"   {chunk.text[:220].strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(" ".join(sys.argv[1:]) or DEFAULT_QUERY)))

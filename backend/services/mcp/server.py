from __future__ import annotations

import hashlib
import html
import logging
import re
from typing import Annotated, Any
from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP
from pydantic import Field

from backend.config import get_settings

logger = logging.getLogger(__name__)

mcp = FastMCP("claim-tools")

_SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RE = re.compile(r"\n{3,}")

USER_AGENT = "claim-verifier/0.1 (+https://github.com/paris-gemma-hackathon)"


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except ValueError:
        return ""


def html_to_text(markup: str) -> str:
    text = _SCRIPT_RE.sub(" ", markup)
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    return _BLANK_RE.sub("\n\n", text).strip()


def _mock_results(query: str, num_results: int) -> list[dict[str, Any]]:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
    hosts = ["reuters.com", "apnews.com", "en.wikipedia.org", "nature.com", "who.int"]
    return [
        {
            "title": f"{query[:70]} — reference {index + 1}",
            "url": f"https://{hosts[index % len(hosts)]}/mock/{digest}/{index + 1}",
            "snippet": (
                f"Offline placeholder result {index + 1} for '{query[:80]}'. "
                "Set SERPAPI_API_KEY to retrieve real sources."
            ),
            "source": hosts[index % len(hosts)],
            "date": "",
        }
        for index in range(max(1, min(num_results, 5)))
    ]


@mcp.tool
async def web_search(
    query: Annotated[str, Field(description="Search query, phrased as it would be typed into Google")],
    num_results: Annotated[int, Field(description="How many results to return (1-10)", ge=1, le=10)] = 5,
) -> dict[str, Any]:
    """Search the live web through SerpAPI and return ranked results with snippets."""
    settings = get_settings()
    if settings.search_is_mocked:
        return {"query": query, "mocked": True, "results": _mock_results(query, num_results)}

    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": settings.serpapi_api_key,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(settings.serpapi_base_url, params=params)
        response.raise_for_status()
        payload = response.json()

    results: list[dict[str, Any]] = []
    for item in (payload.get("organic_results") or [])[:num_results]:
        link = item.get("link", "")
        results.append(
            {
                "title": item.get("title", ""),
                "url": link,
                "snippet": item.get("snippet", "") or item.get("about_this_result", {}).get("source", {}).get("description", ""),
                "source": item.get("source") or _domain(link),
                "date": item.get("date", ""),
            }
        )

    answer_box = payload.get("answer_box") or {}
    if answer_box.get("answer") or answer_box.get("snippet"):
        results.insert(
            0,
            {
                "title": answer_box.get("title", "Answer box"),
                "url": answer_box.get("link", ""),
                "snippet": answer_box.get("answer") or answer_box.get("snippet", ""),
                "source": _domain(answer_box.get("link", "")) or "google",
                "date": "",
            },
        )
    return {"query": query, "mocked": False, "results": results}


@mcp.tool
async def fetch_url(
    url: Annotated[str, Field(description="Absolute http(s) URL of a page to read")],
    max_chars: Annotated[int, Field(description="Maximum characters of text to return", ge=200, le=20000)] = 4000,
) -> dict[str, Any]:
    """Fetch a web page and return its readable text, for checking a source in detail."""
    if not url.startswith(("http://", "https://")):
        return {"url": url, "ok": False, "error": "url must start with http:// or https://", "text": ""}

    try:
        async with httpx.AsyncClient(
            timeout=25.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            markup = response.text
    except httpx.HTTPError as exc:
        return {"url": url, "ok": False, "error": f"fetch failed: {exc}", "text": ""}

    title_match = _TITLE_RE.search(markup)
    text = html_to_text(markup)
    return {
        "url": url,
        "ok": True,
        "error": "",
        "source": _domain(url),
        "title": html.unescape(title_match.group(1)).strip() if title_match else "",
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="http", host=settings.mcp_host, port=settings.mcp_port)


if __name__ == "__main__":
    main()

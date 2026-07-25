from __future__ import annotations

import logging
from typing import Any

from backend.config import Settings
from backend.models.schemas.context import ToolSpec

logger = logging.getLogger(__name__)


class MCPUnavailableError(RuntimeError):
    pass


class MCPToolClient:
    """Thin wrapper over the FastMCP client used by the workflow.

    Connects over streamable HTTP when MCP_URL is set, and otherwise talks to
    the tool server in-process so the whole demo runs from a single command.
    """

    def __init__(self, settings: Settings, server: Any | None = None) -> None:
        self._settings = settings
        self._server = server
        self._client: Any = None
        self._tools: list[ToolSpec] | None = None

    @property
    def transport_label(self) -> str:
        return self._settings.mcp_url or "in-process"

    def _ensure_client(self) -> Any:
        if self._client is None:
            from fastmcp import Client

            if self._settings.mcp_url:
                self._client = Client(self._settings.mcp_url)
            else:
                if self._server is None:
                    from backend.services.mcp.server import mcp as local_server

                    self._server = local_server
                self._client = Client(self._server)
        return self._client

    async def connect(self) -> None:
        """Open a connection once at startup so failures surface early."""
        await self.list_tools(refresh=True)
        logger.info("MCP tools available over %s: %s", self.transport_label, self.tool_names)

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in (self._tools or [])]

    async def list_tools(self, *, refresh: bool = False) -> list[ToolSpec]:
        if self._tools is not None and not refresh:
            return self._tools
        client = self._ensure_client()
        try:
            async with client:
                tools = await client.list_tools()
        except Exception as exc:  # noqa: BLE001 - transport errors are not actionable here
            raise MCPUnavailableError(f"cannot reach MCP server at {self.transport_label}: {exc}") from exc

        self._tools = [
            ToolSpec(
                name=tool.name,
                description=(tool.description or "").strip().split("\n")[0],
                input_schema=tool.inputSchema or {},
            )
            for tool in tools
        ]
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a tool and normalise the outcome into {ok, data, error}."""
        client = self._ensure_client()
        try:
            async with client:
                result = await client.call_tool(name, arguments, raise_on_error=False)
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent as a failed tool call
            logger.warning("MCP call %s failed: %s", name, exc)
            return {"ok": False, "error": str(exc), "data": None}

        if getattr(result, "is_error", False):
            text = result.content[0].text if result.content else "tool reported an error"
            return {"ok": False, "error": text, "data": None}

        data = result.data
        if data is None and result.content:
            data = getattr(result.content[0], "text", None)
        return {"ok": True, "error": "", "data": data}

    async def aclose(self) -> None:
        self._client = None
        self._tools = None

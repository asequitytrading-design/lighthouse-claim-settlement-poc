"""Minimal MCP client for Databricks managed MCP servers (JSON-RPC 2.0 over HTTPS POST).

Managed servers (no hosting on our side):
  UC functions   /api/2.0/mcp/functions/<catalog>/<schema>      every function = one tool
  Vector Search  /api/2.0/mcp/vector-search/<catalog>/<schema>  every index = one tool
  Genie          /api/2.0/mcp/genie/<space_id>                  query_space / poll_response

The servers answer a plain POST with a JSON result (no SSE session needed for tools/list and
tools/call), so this client stays dependency-free (httpx, already used by openai) and uses the
same Databricks auth provider as the gateway clients. Tool arguments and results are never
logged (they may hold case facts); only tool name, latency and status.
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from ...exceptions import LighthouseError


class McpError(LighthouseError):
    """An MCP server returned an error or could not be reached."""

    def __init__(self, message: str, category: str = "mcp"):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: dict


def uc_functions_path(catalog: str, schema: str) -> str:
    return f"/api/2.0/mcp/functions/{catalog}/{schema}"


def vector_search_path(catalog: str, schema: str) -> str:
    return f"/api/2.0/mcp/vector-search/{catalog}/{schema}"


def genie_path(space_id: str) -> str:
    return f"/api/2.0/mcp/genie/{space_id}"


class McpHttpClient:
    def __init__(self, host: str, path: str, auth: Any, logger: logging.Logger,
                 timeout_seconds: float = 45.0, http: Any = None):
        self.url = host.rstrip("/") + path
        self._auth, self._logger, self._timeout = auth, logger, timeout_seconds
        self._http = http
        self._ids = itertools.count(1)

    def _client(self):
        if self._http is None:
            import httpx
            self._http = httpx.Client(timeout=self._timeout)
        return self._http

    def _rpc(self, method: str, params: dict) -> Any:
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   **self._auth.headers()}
        try:
            response = self._client().post(self.url, json=payload, headers=headers)
        except Exception as exc:
            raise McpError(f"MCP server unreachable: {type(exc).__name__}", "mcp_unreachable") from exc
        if response.status_code in (401, 403):
            raise McpError(f"MCP server refused access (HTTP {response.status_code})", "permission")
        if response.status_code >= 400:
            raise McpError(f"MCP server HTTP {response.status_code}", "mcp_http")
        try:
            body = response.json()
        except ValueError as exc:
            raise McpError("MCP server returned non-JSON", "mcp_protocol") from exc
        if "error" in body:
            error = body["error"] or {}
            raise McpError(f"MCP error {error.get('code')}: {str(error.get('message'))[:300]}",
                           "mcp_tool_error")
        return body.get("result") or {}

    def list_tools(self) -> list[McpTool]:
        result = self._rpc("tools/list", {})
        return [McpTool(t["name"], t.get("description", ""), t.get("inputSchema") or {})
                for t in result.get("tools", [])]

    def call_tool(self, name: str, arguments: dict) -> str:
        """The tool's text content (servers return JSON text); raises McpError on failure."""
        started = time.perf_counter()
        status = "ok"
        try:
            result = self._rpc("tools/call", {"name": name, "arguments": arguments})
            if result.get("isError"):
                status = "error"
                raise McpError(f"tool {name} reported an error: {_text(result)[:300]}", "mcp_tool_error")
            return _text(result)
        except McpError:
            status = "error"
            raise
        finally:
            self._logger.info("mcp_tool_called", extra={
                "tool": name, "status": status,
                "latency_ms": round((time.perf_counter() - started) * 1000)})


def _text(result: dict) -> str:
    return "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")


def rows_from_text(text: str) -> tuple[list[str], list[list]]:
    """Columns and rows from the {"columns": [...], "rows": [...]} text the servers return."""
    data = json.loads(text)
    return data.get("columns", []), data.get("rows", [])

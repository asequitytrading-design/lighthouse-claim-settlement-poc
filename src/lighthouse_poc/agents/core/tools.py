"""Tool contract shared by every agent (agents/core).

A tool describes itself to the model (`spec`, OpenAI function-calling format) and runs with the
arguments the model chose. Results are JSON-serialisable dicts: `content` is what goes back to
the model, `summary` is a short log- and UI-safe line, `data` keeps structured extras for the UI
(citations, SQL, rows). Tools never raise for expected failures (not found, no results): they
return status "error" or "empty" so the model can recover; unexpected errors become status
"error" with the error category.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict

    def openai(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


@dataclass
class ToolResult:
    name: str
    status: str                       # ok | empty | error
    content: Any                      # sent back to the model (JSON-serialisable)
    summary: str                      # short, safe to log and show
    latency_ms: int = 0
    data: dict = field(default_factory=dict)
    error_category: str | None = None


class Tool(Protocol):
    spec: ToolSpec

    def run(self, arguments: dict) -> ToolResult: ...


def timed(name: str, fn) -> ToolResult:
    """Run fn() -> ToolResult, filling latency; unexpected exceptions become an error result."""
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:                       # surfaced to the model and the UI, not hidden
        category = getattr(exc, "category", type(exc).__name__)
        result = ToolResult(name, "error", {"error": f"{name} failed ({category})"},
                            f"failed ({category})", error_category=category)
    result.latency_ms = round((time.perf_counter() - started) * 1000)
    return result

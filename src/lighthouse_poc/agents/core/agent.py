"""Tool-calling agent loop shared by every marketplace agent (agents/core).

    for each round (max_rounds):
        model (selected logical model, Unity Gateway, tagged) -> text and/or tool calls
        tool calls -> tools (MCP) -> results back to the model
    final answer -> guardrails -> answer event

`run` yields AgentEvents as they happen, so an API can stream them (SSE) to the UI:
  model_call  a model round started / finished (tokens, latency)
  tool_call   the model asked for a tool (name, arguments)
  tool_result the tool's status, summary, latency and UI data (citations, SQL, rows)
  guardrail   a guardrail changed or flagged the answer
  answer      the final text (after guardrails) with citations
  usage       totals for the turn: model calls, tool calls, tokens, cost (rate card), latency
  error       the turn failed (category); always followed by done
  done        end of the turn

Model calls go through GatewayChatClient.complete_with_tools, so each one is tagged and recorded
by the active usage tracker. Logs carry tool names, statuses and latencies, never tool arguments
or results (they can hold case facts).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

from ...exceptions import LighthouseError
from .tools import Tool, ToolResult


@dataclass
class AgentEvent:
    type: str
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"type": self.type, **self.data}


@dataclass(frozen=True)
class AgentConfig:
    name: str
    system_prompt: str
    max_rounds: int = 4
    default_model: str = "claims-small"


class Guardrail(Protocol):
    name: str

    def check(self, answer: str, context: "TurnContext") -> tuple[str, str | None]:
        """Return (answer, note). A note means the guardrail changed or flagged the answer."""


@dataclass
class TurnContext:
    """What a turn produced, for guardrails and the usage summary."""
    tool_results: list[ToolResult] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


# error category -> seconds to wait before the single retry of a model round
RETRYABLE = {"empty_response": 0.0, "rate_limit": 2.0}


class Agent:
    def __init__(self, config: AgentConfig, tools: dict[str, Tool],
                 model_factory: Callable[[str], Any], logger: logging.Logger,
                 guardrails: list[Guardrail] | None = None,
                 price_of: Callable[[str, int, int], float | None] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.config, self.tools, self._models = config, tools, model_factory
        self._logger, self._guardrails = logger, guardrails or []
        self._price_of = price_of or (lambda model, i, o: None)
        self._sleep = sleep

    def run(self, user_message: str, history: list[dict] | None = None,
            model_key: str | None = None) -> Iterator[AgentEvent]:
        model_key = model_key or self.config.default_model
        started = time.perf_counter()
        ctx = TurnContext()
        messages = [{"role": "system", "content": self.config.system_prompt},
                    *(history or []), {"role": "user", "content": user_message}]
        specs = [tool.spec.openai() for tool in self.tools.values()]
        try:
            model = self._models(model_key)
            answer = None
            for round_number in range(1, self.config.max_rounds + 2):
                last = round_number > self.config.max_rounds   # out of rounds: answer without tools
                yield AgentEvent("model_call", {"round": round_number, "model": model_key, "status": "started"})
                turn = None
                for attempt in range(1, 3):
                    try:
                        turn = model.complete_with_tools(messages, [] if last else specs)
                        break
                    except LighthouseError as exc:
                        category = getattr(exc, "category", "")
                        if attempt == 2 or category not in RETRYABLE:
                            raise
                        # reasoning models occasionally return an empty message; free-tier rate limits
                        yield AgentEvent("model_call", {"round": round_number, "model": model_key,
                                                        "status": "retry", "reason": category})
                        self._logger.warning("agent_model_retry", extra={
                            "agent": self.config.name, "model_alias": model_key, "error_category": category})
                        self._sleep(RETRYABLE[category])
                ctx.model_calls += 1
                ctx.input_tokens += turn.prompt_tokens or 0
                ctx.output_tokens += turn.completion_tokens or 0
                yield AgentEvent("model_call", {
                    "round": round_number, "model": model_key, "status": "done",
                    "latency_ms": turn.duration_ms, "tool_calls": len(turn.tool_calls),
                    "input_tokens": turn.prompt_tokens, "output_tokens": turn.completion_tokens})
                if not turn.tool_calls:
                    answer = turn.text.strip()
                    break
                messages.append({"role": "assistant", "content": turn.text or None, "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c in turn.tool_calls]})
                for call in turn.tool_calls:
                    arguments = _arguments(call.arguments)
                    yield AgentEvent("tool_call", {"id": call.id, "name": call.name, "arguments": arguments})
                    result = self._run_tool(call.name, arguments)
                    ctx.tool_results.append(result)
                    ctx.citations += result.data.get("citations", [])
                    yield AgentEvent("tool_result", {
                        "id": call.id, "name": call.name, "status": result.status,
                        "summary": result.summary, "latency_ms": result.latency_ms,
                        "data": result.data})
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": json.dumps(result.content, default=str)})
            for guardrail in self._guardrails:
                answer, note = guardrail.check(answer or "", ctx)
                if note:
                    yield AgentEvent("guardrail", {"name": guardrail.name, "note": note})
            yield AgentEvent("answer", {"text": answer, "citations": _unique(ctx.citations)})
        except LighthouseError as exc:
            category = getattr(exc, "category", type(exc).__name__)
            self._logger.error("agent_turn_failed", extra={"agent": self.config.name,
                                                           "model_alias": model_key,
                                                           "error_category": category})
            yield AgentEvent("error", {"category": category, "message": str(exc)[:300]})
        finally:
            latency = round((time.perf_counter() - started) * 1000)
            cost = self._price_of(model_key, ctx.input_tokens, ctx.output_tokens)
            usage = {"model": model_key, "model_calls": ctx.model_calls,
                     "tool_calls": len(ctx.tool_results), "input_tokens": ctx.input_tokens,
                     "output_tokens": ctx.output_tokens, "cost_usd": cost, "latency_ms": latency}
            self._logger.info("agent_turn_completed", extra={
                "agent": self.config.name, "model_alias": model_key,
                "tools": [r.name for r in ctx.tool_results],
                "tool_statuses": [r.status for r in ctx.tool_results],
                "kpi_name": "agent_turn_latency_ms", "kpi_value": latency})
        yield AgentEvent("usage", usage)
        yield AgentEvent("done", {})

    def _run_tool(self, name: str, arguments: dict) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(name, "error", {"error": f"unknown tool {name}"}, "unknown tool",
                              error_category="unknown_tool")
        return tool.run(arguments)


def _arguments(text: str) -> dict:
    try:
        value = json.loads(text or "{}")
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        return {"_unparsed": (text or "")[:200]}


def _unique(citations: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in citations:
        key = c.get("chunk_id") or c.get("doc_id")
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

"""The claims examiner copilot (demo agent): prompt, handoff tool and factory.

Tools: get_claim_case, search_guidance, ask_portfolio (Databricks managed MCP, tools.py) and
draft_handoff, which runs the governed claim settlement assistant with the APPROVED prompt version
(<prompt>@champion from the Unity Catalog prompt registry). It fails closed when no version is
approved, so the copilot never drafts with an unreviewed prompt. Synthetic data only; the copilot
prepares files for a human examiner and never decides coverage, eligibility, fraud or amounts.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Callable

from ... import __version__
from ...config import Profile
from ...exceptions import LighthouseError
from ..core.agent import Agent, AgentConfig
from ..core.guardrails import (CaseFactVerifier, DecisionQuestionGuardrail, NonDecisionGuardrail,
                               QuoteGuardrail)
from ..core.tools import Tool, ToolResult, ToolSpec, timed

NO_INFORMATION = "I do not have this information in the knowledge base."

SYSTEM_PROMPT = f"""You are the Lighthouse claims examiner copilot. You help a human claims examiner prepare claim files. All data is synthetic.

Use the tools to get facts; never invent case facts or claims guidance:
- A question about one claim (a case ID such as CSET-LTD-002): call get_claim_case first.
- A question about procedures, required documents or the handoff format: call search_guidance.
- A question across many claims (counts, lists, statuses by product): call ask_portfolio.
- A request to prepare or draft the examiner handoff for a claim: call draft_handoff. If it returns case facts instead of a handoff (no approved template), write the handoff from those facts only, with these sections: Case summary, Evidence received, Missing or conflicting items, Suggested next administrative steps, Benefit determination: Not determined. Say that it was prepared without the approved template.
- Follow-up questions ("which of those...", "and by region?") about claims data: call the tool again with the complete question including the earlier filters. Earlier answers in this conversation are not data: never list case IDs, counts or dates from memory.

Grounding rules:
- State only facts that appear in tool results of this turn (product line, dates, documents, counts, case IDs). When you use guidance, quote the exact sentence in double quotes and name its source document in brackets, e.g. "..." [kb-ltd-evidence].
- If search_guidance returns no guidance, or the passages are about a different product or topic than the question, answer exactly: "{NO_INFORMATION}" Do not answer from general knowledge.

Boundaries:
- Never decide or suggest coverage, eligibility, exclusions, diagnosis, causation, fraud, claim validity or benefit amounts. If asked to, start your answer with a sentence like: "I don't make fraud or claim decisions; a human claims examiner does." Then offer what you can prepare (missing documents, conflicts, next administrative steps), using tool results only.

Style: concise markdown, short sections or bullet lists, case IDs in bold."""

# Default chat model: qwen3-next (instruct, no reasoning channel) answered every tool-calling probe
# and never returned an empty reply; gpt-oss-20b (claims-small) occasionally did on follow-ups.
# Still a catalog *candidate*: approve it in config/model_catalog.yaml before production use.
DEFAULT_MODEL = "claims-alt-qwen-next"

PRODUCT_SEGMENTS = {"std": "short_term_disability", "ltd": "long_term_disability",
                    "life": "life_insurance", "accident": "accident_insurance",
                    "critical_illness": "critical_illness", "dental": "dental_insurance"}


class HandoffTool:
    spec = ToolSpec(
        "draft_handoff",
        "Draft the structured handoff for the human examiner for one claim, using the approved "
        "(champion) claim settlement prompt: summary, key facts, evidence received, evidence to "
        "confirm, inconsistencies, recommended examiner actions; benefit determination is always "
        "'Not determined'.",
        {"type": "object", "properties": {"case_id": {"type": "string",
                                                      "description": "Claim case ID, e.g. CSET-LTD-002"}},
         "required": ["case_id"]})

    def __init__(self, case_tool: Tool, assistant_factory: Callable[[], Any]):
        self._case_tool, self._assistant_factory = case_tool, assistant_factory

    def run(self, arguments: dict) -> ToolResult:
        def call() -> ToolResult:
            case = self._case_tool.run({"case_id": arguments.get("case_id", "")})
            if case.status != "ok":
                return ToolResult(self.spec.name, case.status, case.content, case.summary)
            facts = case.content
            try:
                assistant = self._assistant_factory()
            except LighthouseError:     # no approved prompt: the governed draft fails closed,
                return ToolResult(      # the model gets the facts to write a manual handoff
                    self.spec.name, "ok",
                    {"approved_template": "unavailable: no approved prompt version (@champion)",
                     "case": facts,
                     "instruction": "Write the handoff from these case facts only: Case summary, "
                                    "Evidence received, Missing or conflicting items, Suggested next "
                                    "administrative steps, Benefit determination: Not determined."},
                    "no approved prompt (@champion): case facts returned for a manual handoff",
                    data={"fallback": True, "case_id": facts.get("case_id")})
            documents = [d["document_type"] for d in facts.get("documents", []) if d.get("status") == "received"]
            missing = [d["document_type"] for d in facts.get("documents", []) if d.get("status") != "received"]
            narrative = f"{facts.get('case_summary', '')} Documents not received or conflicting: {', '.join(missing) or 'none'}."
            output = assistant.predict(
                facts["case_id"], PRODUCT_SEGMENTS.get(facts.get("product_line"), facts.get("product_line")),
                narrative, documents,
                "Synthetic example only. Examiner should consult the applicable contract and procedures. No determination.")
            return ToolResult(self.spec.name, "ok", output,
                              f"handoff for {facts['case_id']} (prompt v{getattr(assistant, '_prompt_version', '?')})",
                              data={"handoff": output,
                                    "prompt_version": getattr(assistant, "_prompt_version", None)})
        return timed(self.spec.name, call)


class CachedFactory:
    """The approved-prompt assistant, cached (versions are immutable); a failure ("no approved
    version") is cached for `retry_seconds` so each handoff request does not repeat a slow
    registry lookup. A promote takes effect within that time or on restart."""

    def __init__(self, load: Callable[[], Any], retry_seconds: float = 120.0, ttl_seconds: float = 600.0,
                 clock: Callable[[], float] = time.monotonic):
        self._load, self._retry, self._ttl, self._clock = load, retry_seconds, ttl_seconds, clock
        self._value: Any = None
        self._error: Exception | None = None
        self._until = 0.0

    def __call__(self) -> Any:
        if self._clock() < self._until:
            if self._error is not None:
                raise self._error
            return self._value
        try:
            self._value, self._error = self._load(), None
            self._until = self._clock() + self._ttl          # re-check @champion periodically
        except LighthouseError as exc:
            self._value, self._error = None, exc
            self._until = self._clock() + self._retry
            raise
        return self._value


def price_function(profile: Profile) -> Callable[[str, int, int], float | None]:
    """Cost of a turn from the rate card (illustrative rates until contract rates exist)."""
    try:
        from ...platform.pricing import load_rate_card
        rates = {p.model_key: p for p in load_rate_card()}
    except LighthouseError:
        rates = {}

    def price(model: str, input_tokens: int, output_tokens: int) -> float | None:
        rate = rates.get(model)
        if rate is None:
            return None
        return round((input_tokens * rate.input_per_1m_tokens + output_tokens * rate.output_per_1m_tokens) / 1e6, 6)
    return price


def build_copilot(profile: Profile, logger: logging.Logger, tools: dict[str, Tool] | None = None,
                  model_factory: Callable[[str], Any] | None = None,
                  assistant_factory: Callable[[], Any] | None = None, cache: Any = None) -> Agent:
    from ...gateway import build_chat_model
    if tools is None:
        from .tools import build_tools
        tools = build_tools(profile, logger, cache=cache)

    @lru_cache(maxsize=8)
    def models(key: str):
        return build_chat_model(profile, key, logger)

    if assistant_factory is None:
        def load_assistant():
            from ...agent import published_assistant
            from ...prompt_registry import MLflowPromptRegistry
            import mlflow
            if profile.registry_uri:
                mlflow.set_registry_uri(profile.registry_uri)
            return published_assistant(MLflowPromptRegistry(logger), models("claims-medium"), logger,
                                       profile.prompt_name, __version__)
        assistant_factory = CachedFactory(load_assistant)
    tools = dict(tools)
    if "get_claim_case" in tools:
        tools["draft_handoff"] = HandoffTool(tools["get_claim_case"], assistant_factory)
    return Agent(AgentConfig("claims_copilot", SYSTEM_PROMPT, default_model=DEFAULT_MODEL), tools,
                 model_factory or models, logger,
                 guardrails=[NonDecisionGuardrail(), DecisionQuestionGuardrail(), QuoteGuardrail()],
                 price_of=price_function(profile),
                 verifiers=[CaseFactVerifier()])

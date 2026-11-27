"""Optional demo traffic for the observability dashboard (notebook 04).

A fresh platform has only the verification probes, so the dashboard looks empty. This sends a
small number of REAL calls through Unity Gateway with short synthetic prompts (no real
policyholder data, no health details) and records each one in <obs>.model_invocation with
request tags, exactly like an application would. Tokens, latency and errors are therefore real;
only the inputs are synthetic. Use case `claim_settlement_demo` keeps the traffic separate.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..exceptions import ModelInvocationError
from .model_catalog import ModelEntry
from .observability import EventWriter, ModelInvocation, new_event_id

DEMO_USE_CASE = "claim_settlement_demo"
DEMO_PROMPT_VERSION = "sample-traffic-v1"

# Synthetic, administrative case snippets: products and document names only.
SYNTHETIC_CASES = [
    ("SYN-STD-0101", "short-term disability",
     "claimant statement and employer statement received; attending provider statement missing"),
    ("SYN-LTD-0102", "long-term disability",
     "elimination period dates differ between the claim form and the employer statement"),
    ("SYN-LIFE-0103", "life",
     "death certificate received; beneficiary designation form is unsigned"),
    ("SYN-ACC-0104", "accident",
     "incident report received; date of incident differs from the claim form"),
    ("SYN-CI-0105", "critical illness",
     "claim form received; specialist confirmation letter not yet received"),
    ("SYN-DEN-0106", "dental",
     "itemised invoice received; treatment plan reference missing"),
]

APP_INSTRUCTION = ("You prepare claim files for a human examiner. Never decide coverage, "
                   "eligibility or benefit amounts. In two short sentences, summarise the file "
                   "status and name the missing or conflicting item.")
JUDGE_INSTRUCTION = ("Rate from 1 to 5 how clearly this handoff names the missing or conflicting "
                     "item. Reply with the number and five words of reason.")


@dataclass
class TrafficSummary:
    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, event: ModelInvocation) -> None:
        self.calls += 1
        self.errors += event.status == "error"
        self.input_tokens += event.input_tokens or 0
        self.output_tokens += event.output_tokens or 0


class SampleTrafficGenerator:
    """Sends synthetic app, judge and embedding calls for approved models and records them."""

    def __init__(self, entries: list[ModelEntry], names, engine: Any, logger: logging.Logger,
                 chat_factory: Callable[[ModelEntry], Any],
                 embed_factory: Callable[[ModelEntry], Any], environment: str = "dev"):
        self._entries, self._engine, self._logger = entries, engine, logger
        self._chat_factory, self._embed_factory = chat_factory, embed_factory
        self._environment = environment
        self._writer = EventWriter(names, logger)

    def _event(self, entry: ModelEntry, purpose: str, case_id: str) -> ModelInvocation:
        return ModelInvocation(event_id=new_event_id(), status="ok", purpose=purpose,
                               model_key=entry.key, model_service=entry.model_service,
                               task=entry.task, environment=self._environment,
                               use_case_id=DEMO_USE_CASE, prompt_name="sample_traffic", activity="demo",
                               prompt_version=DEMO_PROMPT_VERSION, case_id=case_id)

    def _call(self, entry: ModelEntry, purpose: str, case_id: str, text: str,
              instruction: str | None) -> tuple[ModelInvocation, str]:
        event = self._event(entry, purpose, case_id)
        # client construction is outside the try: a configuration error stops the run instead of
        # being recorded as a failed model call
        client = (self._chat_factory if entry.task == "chat" else self._embed_factory)(entry)
        started, answer = time.perf_counter(), ""
        try:
            if entry.task == "chat":
                result = client.complete_messages(
                    [{"role": "system", "content": instruction}, {"role": "user", "content": text}],
                    tags=event.request_tags())
                event.input_tokens, event.output_tokens = result.prompt_tokens, result.completion_tokens
                event.served_model, answer = result.served_model, result.text
            else:
                result = client.embed([text], tags=event.request_tags())
                event.input_tokens, event.input_count = result.prompt_tokens, 1
                event.embedding_dimensions, event.served_model = result.dimensions, result.served_model
        except ModelInvocationError as exc:  # recorded as an error event: the dashboard shows failures
            event.status = "error"
            event.error_category = exc.category
            self._logger.warning("sample_traffic_call_failed", extra={
                "model_alias": entry.key, "purpose": purpose, "case_id": case_id,
                "error_category": event.error_category})
        event.latency_ms = round((time.perf_counter() - started) * 1000)
        return event, answer

    def run(self, rounds: int = 2) -> TrafficSummary:
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        apps = [e for e in self._entries if e.allows("app")]
        judges = [e for e in self._entries if e.allows("judge")]
        embedders = [e for e in self._entries if e.allows("embedding")]
        summary, events = TrafficSummary(), []
        for round_no in range(rounds):
            for i, (case_id, product, facts) in enumerate(SYNTHETIC_CASES):
                text = f"Synthetic case {case_id}, {product} claim: {facts}."
                app = apps[(i + round_no) % len(apps)]
                event, handoff = self._call(app, "app", case_id, text, APP_INSTRUCTION)
                events.append(event)
                if judges and handoff:
                    judged, _ = self._call(judges[0], "judge", case_id, handoff, JUDGE_INSTRUCTION)
                    events.append(judged)
                if embedders:
                    embedded, _ = self._call(embedders[0], "embedding_query", case_id, text, None)
                    events.append(embedded)
        self._writer.write(self._engine, invocations=events)
        for event in events:
            summary.add(event)
        self._logger.info("sample_traffic_completed", extra={
            "calls": summary.calls, "errors": summary.errors, "use_case_id": DEMO_USE_CASE,
            "kpi_name": "sample_traffic_calls", "kpi_value": summary.calls})
        return summary

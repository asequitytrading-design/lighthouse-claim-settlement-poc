"""Grounded answers over the knowledge index: answer only from embedded knowledge, else refuse.

Retrieval alone cannot say "I don't know": vector search always returns the nearest chunks, and
raw similarity never gets near 0 (any English sentence scores ~0.4 against these chunks; the
hybrid score of the top hit is ~1.0 even for "best pizza in Rome"). Grounding is three checks,
each of which can only lower the score:

  1. Retrieval confidence  dense similarity of the best chunk, calibrated to 0..1:
                           (similarity - floor) / (ceiling - floor), clipped. Below the floor the
                           score is exactly 0 and the answer is refused WITHOUT a model call.
  2. Answerability         an approved chat model (through Unity Gateway, tagged) reads the
                           retrieved passages and either declares them insufficient or answers
                           from them, citing passage numbers and quoting the supporting sentences.
                           Catches look-alikes the embedding scores as relevant (pet insurance).
  3. Quote verification    deterministic: every quote must appear verbatim (whitespace and case
                           normalised) in a passage it cites. A model that invents support fails.

  grounding score = retrieval confidence of the best cited chunk x share of verified quotes
                    0 whenever any check refuses -> "I do not have this information."

The answer is guidance from the knowledge base for a human examiner. The prompt forbids deciding
coverage, eligibility, fraud, causation or benefit amounts, as everywhere in this POC.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..exceptions import ModelInvocationError
from .model_catalog import ModelEntry
from .observability import ModelInvocation, new_event_id
from .retrieval import EXPLORER_USE_CASE, Hit

NO_INFORMATION = "I do not have this information in the knowledge base."

GROUNDED_SYSTEM = """You answer a claims examiner's question using ONLY the numbered passages.
Rules:
- If the passages do not contain the answer, reply {"answerable": false}. Do not use outside knowledge.
  Passages about a different product or topic than the question asks do NOT contain the answer.
- Otherwise reply {"answerable": true, "answer": "<2-4 sentences>", "citations": [passage numbers],
  "quotes": ["<sentence copied exactly from a cited passage>", ...]}. Every fact in the answer must
  be supported by a quote.
- Never decide coverage, eligibility, exclusions, diagnosis, causation, fraud, claim validity or
  benefit amounts. If asked to, answer only with what the passages say about who decides.
Reply with one JSON object only."""


@dataclass(frozen=True)
class RetrievalConfidence:
    """Dense similarity -> 0..1. Calibrated 2026-11-27 on this index (RelevanceGate): off-topic
    questions 0.42-0.52, sample queries 0.53-0.72."""
    floor: float = 0.50
    ceiling: float = 0.70

    def score(self, similarity: float | None) -> float:
        if similarity is None:
            return 0.0
        return round(min(1.0, max(0.0, (similarity - self.floor) / (self.ceiling - self.floor))), 3)


@dataclass
class GroundedAnswer:
    question: str
    answer: str
    grounded: bool
    grounding_score: float               # 0 when refused
    retrieval_confidence: float          # check 1 (best chunk)
    refusal_reason: str | None = None    # which check refused
    citations: list[int] = field(default_factory=list)
    verified_quotes: list[str] = field(default_factory=list)
    unverified_quotes: list[str] = field(default_factory=list)
    cited_chunks: list[str] = field(default_factory=list)
    model_called: bool = False
    latency_ms: int | None = None

    @classmethod
    def refused(cls, question: str, reason: str, confidence: float, **extra: Any) -> "GroundedAnswer":
        return cls(question, NO_INFORMATION, False, 0.0, confidence, reason, **extra)


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[‘’“”`]", "'", text).lower().split()).strip(" .'\"")


def parse_reply(text: str) -> dict | None:
    """The model's JSON object, or None when there is none."""
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        reply = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
    return reply if isinstance(reply, dict) else None


def verify_quotes(quotes: list, citations: list[int], passages: list[Hit]) -> tuple[list[str], list[str]]:
    """Split quotes into those found verbatim in a cited passage and those that are not."""
    cited = [_normalise(passages[i - 1].text) for i in citations]
    verified, unverified = [], []
    for quote in quotes:
        q = _normalise(str(quote))
        (verified if q and len(q) >= 12 and any(q in p for p in cited) else unverified).append(str(quote))
    return verified, unverified


class GroundedAnswerer:
    """Answers from retrieved passages or refuses; records its model call as a usage event."""

    def __init__(self, entry: ModelEntry, chat_client: Any, logger: logging.Logger,
                 environment: str = "dev", confidence: RetrievalConfidence = RetrievalConfidence()):
        self._entry, self._chat, self._environment = entry, chat_client, environment
        self._logger = logger
        self.confidence = confidence
        self.events: list[ModelInvocation] = []

    def answer(self, question: str, passages: list[Hit], dense_hits: list[Hit]) -> GroundedAnswer:
        """`passages`: what the strategy retrieved, in rank order. `dense_hits`: a dense (ANN)
        search for the same question; its similarities are used for confidence because the
        strategy's own scores may be rank-fused."""
        best = self.confidence.score(max((h.score for h in dense_hits if h.score is not None), default=None))
        if best == 0.0 or not passages:
            return GroundedAnswer.refused(question, "retrieval: no chunk above the similarity floor "
                                          f"{self.confidence.floor}", best)
        started = time.perf_counter()
        reply, error = self._ask(question, passages)
        latency = round((time.perf_counter() - started) * 1000)
        common = {"model_called": True, "latency_ms": latency}
        if error:                                   # fail closed: no answer without the check
            return GroundedAnswer.refused(question, f"model call failed ({error})", best, **common)
        if reply is None:
            return GroundedAnswer.refused(question, "model: reply was not JSON", best, **common)
        if reply.get("answerable") is not True:
            return GroundedAnswer.refused(question, "model: passages do not contain the answer", best, **common)
        citations = list(dict.fromkeys(int(c) for c in reply.get("citations") or []
                                       if str(c).isdigit() and 1 <= int(c) <= len(passages)))
        quotes = reply.get("quotes") if isinstance(reply.get("quotes"), list) else []
        verified, unverified = verify_quotes(quotes, citations, passages)
        extra = {**common, "citations": citations, "verified_quotes": verified,
                 "unverified_quotes": unverified}
        if not citations or not verified:
            return GroundedAnswer.refused(question, "verification: no quote found in the cited passages",
                                          best, **extra)
        cited = [passages[i - 1] for i in citations]
        cited_chunks = [h.chunk_id for h in cited]
        cited_confidence = max((self.confidence.score(self._similarity(h, dense_hits)) for h in cited),
                               default=0.0)
        score = round(cited_confidence * len(verified) / len(quotes), 3)
        if score == 0.0:
            return GroundedAnswer.refused(question, "retrieval: cited chunks are below the similarity floor",
                                          best, **extra)
        return GroundedAnswer(question, str(reply.get("answer") or "").strip(), True, score, best,
                              cited_chunks=cited_chunks, **extra)

    @staticmethod
    def _similarity(passage: Hit, dense_hits: list[Hit]) -> float | None:
        """Dense similarity of a passage: its chunk's, or for a whole document (parent_document,
        no chunk index) its best chunk's."""
        scores = [d.score for d in dense_hits if d.score is not None and (
            d.chunk_id == passage.chunk_id or (passage.chunk_index is None and d.doc_id == passage.doc_id))]
        return max(scores, default=None)

    def _ask(self, question: str, passages: list[Hit]) -> tuple[dict | None, str | None]:
        event = ModelInvocation(event_id=new_event_id(), status="ok", purpose="grounded_answer",
                                model_key=self._entry.key, model_service=self._entry.model_service,
                                environment=self._environment, use_case_id=EXPLORER_USE_CASE,
                                activity="explorer")
        text = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(passages, 1))
        started = time.perf_counter()
        try:
            result = self._chat.complete_messages(
                [{"role": "system", "content": GROUNDED_SYSTEM},
                 {"role": "user", "content": f"Question: {question}\n\nPassages:\n{text}"}],
                tags=event.request_tags())
            event.input_tokens, event.output_tokens = result.prompt_tokens, result.completion_tokens
            event.served_model = result.served_model
            return parse_reply(result.text), None
        except ModelInvocationError as exc:
            event.status, event.error_category = "error", exc.category
            self._logger.warning("grounded_answer_failed", extra={
                "error_category": exc.category, "model_key": self._entry.key}, exc_info=exc)
            return None, exc.category
        finally:
            event.latency_ms = round((time.perf_counter() - started) * 1000)
            self.events.append(event)


def ground(answerer: GroundedAnswerer, report: Any) -> GroundedAnswer:
    """Grounded answer for a single-query report (retrieval.run_single): the strategy's results
    are the passages, the report's dense probe gives the confidence."""
    return answerer.answer(report.query.query, report.result.hits, report.dense_hits)

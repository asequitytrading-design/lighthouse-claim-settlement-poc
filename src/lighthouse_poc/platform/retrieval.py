"""Retrieval strategies over the knowledge index (Mosaic AI Vector Search), and their comparison.

Notebook 03 section 4 uses this to run synthetic sample queries with one selected strategy, or
to compare all strategies on the same queries with known answers (each sample query names the
document that answers it).

Strategies (all share one query embedding per query, computed through Unity Gateway, ADR-013):
  dense            ANN on the query embedding (the baseline)
  hybrid           Vector Search HYBRID: keyword (full-text) + vector results fused by rank
  filtered_hybrid  a rules-based product router turns the query into a metadata filter
                   (doc_id IN the product's documents + cross-product guidance), then hybrid
  parent_document  hybrid over chunks, grouped per document; returns whole documents
                   (small-to-big: precise matching, complete context)
  llm_rerank       hybrid candidates re-ordered by an approved chat model through the gateway
                   (Vector Search's built-in reranker is not enabled on Free Edition)

Usage events: each embedding batch and each rerank call is a model_invocation, each index query
a vector_query (use case `knowledge_search_explorer`); no query text is stored.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import PROJECT_ROOT
from ..exceptions import ConfigurationError, ModelInvocationError
from .model_catalog import ModelEntry
from .observability import EventWriter, ModelInvocation, VectorQuery, new_event_id

SAMPLE_QUERIES_FILE = PROJECT_ROOT / "data" / "knowledge" / "sample_queries.jsonl"
EXPLORER_USE_CASE = "knowledge_search_explorer"
INDEX_COLUMNS = ["chunk_id", "doc_id", "chunk_index", "text"]


# ---------------------------------------------------------------- data
@dataclass(frozen=True)
class SampleQuery:
    query_id: str
    query: str
    expected_doc_id: str | None = None
    style: str | None = None        # keyword | paraphrase | exact_term


def load_sample_queries(path: Path | None = None) -> list[SampleQuery]:
    path = Path(path) if path else SAMPLE_QUERIES_FILE
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read sample queries {path}") from exc
    queries = [SampleQuery(**r) for r in rows]
    if len({q.query_id for q in queries}) != len(queries):
        raise ConfigurationError(f"Duplicate query_id in {path.name}")
    return queries


@dataclass(frozen=True)
class Hit:
    rank: int
    doc_id: str
    chunk_id: str
    chunk_index: int | None
    score: float | None
    text: str


@dataclass
class RetrievalResult:
    strategy: str
    query: SampleQuery
    hits: list[Hit]
    latency_ms: int
    model_calls: int = 0           # model calls beyond the shared query embedding
    note: str = ""

    @property
    def doc_ranking(self) -> list[str]:
        """Distinct documents in rank order (metrics are document-level)."""
        seen: list[str] = []
        for hit in self.hits:
            if hit.doc_id not in seen:
                seen.append(hit.doc_id)
        return seen

    def reciprocal_rank(self) -> float:
        ranking = self.doc_ranking
        expected = self.query.expected_doc_id
        return 1.0 / (ranking.index(expected) + 1) if expected in ranking else 0.0

    def hit_at(self, k: int) -> bool:
        return self.query.expected_doc_id in self.doc_ranking[:k]


@dataclass(frozen=True)
class KnowledgeCorpus:
    """What the strategies need besides the index: document product lines and chunk texts."""
    product_lines: dict[str, str]                         # doc_id -> product_line
    chunks: dict[str, list[tuple[int, str]]]              # doc_id -> [(chunk_index, text)]
    titles: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, documents: list[dict], chunks: list[dict]) -> "KnowledgeCorpus":
        by_doc: dict[str, list[tuple[int, str]]] = {}
        for c in chunks:
            by_doc.setdefault(c["doc_id"], []).append((int(c["chunk_index"]), c["text"]))
        return cls(product_lines={d["doc_id"]: d["product_line"] for d in documents},
                   chunks={k: sorted(v) for k, v in by_doc.items()},
                   titles={d["doc_id"]: d.get("title", "") for d in documents})

    def document_text(self, doc_id: str) -> str:
        return "\n\n".join(text for _, text in self.chunks.get(doc_id, []))

    def docs_for(self, product_lines: set[str]) -> list[str]:
        return sorted(d for d, p in self.product_lines.items() if p in product_lines)


# ---------------------------------------------------------------- index access
class SearchBackend:
    """Thin wrapper over a Vector Search index: ANN and hybrid queries, timed and recorded."""

    def __init__(self, index: Any, index_name: str, embedding_model_key: str):
        self._index, self.index_name, self._model_key = index, index_name, embedding_model_key
        self.queries: list[VectorQuery] = []

    def _search(self, k: int, **kwargs) -> list[Hit]:
        started, status, hits = time.perf_counter(), "ok", []
        try:
            result = self._index.similarity_search(columns=INDEX_COLUMNS, num_results=k,
                                                   disable_notice=True, **kwargs)
            columns = [c["name"] for c in result.get("manifest", {}).get("columns", [])]
            for rank, row in enumerate(result.get("result", {}).get("data_array", []) or [], 1):
                r = dict(zip(columns, row))
                index = r.get("chunk_index")
                hits.append(Hit(rank, r["doc_id"], r["chunk_id"],
                                int(index) if index is not None else None, r.get("score"), r["text"]))
        except Exception:
            status = "error"
            raise
        finally:
            self.queries.append(VectorQuery(
                event_id=new_event_id(), status=status, index_name=self.index_name,
                embedding_model_key=self._model_key, num_results_requested=k,
                use_case_id=EXPLORER_USE_CASE, num_results=len(hits),
                latency_ms=round((time.perf_counter() - started) * 1000)))
        return hits

    def ann(self, vector: list[float], k: int, filters: dict | None = None) -> list[Hit]:
        return self._search(k, query_vector=vector, filters=filters)

    def hybrid(self, text: str, vector: list[float], k: int, filters: dict | None = None) -> list[Hit]:
        return self._search(k, query_vector=vector, query_text=text, query_type="HYBRID",
                            filters=filters)


class QueryEncoder:
    """Embeds queries through the gateway once (batched) and caches the vectors."""

    def __init__(self, entry: ModelEntry, client: Any, environment: str = "dev"):
        self._entry, self._client, self._environment = entry, client, environment
        self._cache: dict[str, list[float]] = {}
        self.events: list[ModelInvocation] = []

    def encode(self, texts: list[str]) -> None:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if not missing:
            return
        event = ModelInvocation(event_id=new_event_id(), status="ok", purpose="embedding_query",
                                model_key=self._entry.key, model_service=self._entry.model_service,
                                task="embeddings", environment=self._environment,
                                use_case_id=EXPLORER_USE_CASE, input_count=len(missing),
                                activity="explorer")
        started = time.perf_counter()
        try:
            result = self._client.embed(missing, tags=event.request_tags())
            event.input_tokens, event.served_model = result.prompt_tokens, result.served_model
            event.embedding_dimensions = result.dimensions
            self._cache.update(zip(missing, result.vectors))
        except ModelInvocationError as exc:
            event.status, event.error_category = "error", exc.category
            raise
        finally:
            event.latency_ms = round((time.perf_counter() - started) * 1000)
            self.events.append(event)

    def vector(self, text: str) -> list[float]:
        self.encode([text])
        return self._cache[text]


# ---------------------------------------------------------------- strategies
class RetrievalStrategy(Protocol):
    name: str
    summary: str

    def retrieve(self, query: SampleQuery, k: int) -> RetrievalResult: ...


class _Base:
    name = ""
    summary = ""

    def __init__(self, backend: SearchBackend, encoder: QueryEncoder):
        self._backend, self._encoder = backend, encoder

    def _timed(self, query: SampleQuery, run) -> RetrievalResult:
        vector = self._encoder.vector(query.query)          # embedding time is not counted
        started = time.perf_counter()
        hits, calls, note = run(vector)
        return RetrievalResult(self.name, query, hits, round((time.perf_counter() - started) * 1000),
                               calls, note)


class DenseStrategy(_Base):
    name = "dense"
    summary = "ANN on the query embedding only (semantic similarity). Baseline."

    def retrieve(self, query: SampleQuery, k: int) -> RetrievalResult:
        return self._timed(query, lambda v: (self._backend.ann(v, k), 0, ""))


class HybridStrategy(_Base):
    name = "hybrid"
    summary = ("Keyword (full-text) and vector results fused by rank. Helps exact terms, codes and "
               "field names that embeddings blur.")

    def retrieve(self, query: SampleQuery, k: int) -> RetrievalResult:
        return self._timed(query, lambda v: (self._backend.hybrid(query.query, v, k), 0, ""))


PRODUCT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("std", ("short-term disability", "short term disability", "std ", "last day worked")),
    ("ltd", ("long-term disability", "long term disability", "elimination period",
             "functional capacity")),
    ("life", ("life claim", "death", "passed away", "deceased", "beneficiar", "insured person")),
    ("accident", ("accident", "incident", "injury", "police")),
    ("critical_illness", ("critical illness", "specialist", "consultant", "illness", "diagnos")),
    ("dental", ("dental", "procedure", "treatment statement", "itemised", "pre-treatment",
                "tooth", "teeth")),
]


def route_products(text: str) -> set[str]:
    """Product lines a query mentions (rules on lower-cased text); empty = no filter."""
    lowered = f" {text.lower()} "
    return {product for product, words in PRODUCT_RULES if any(w in lowered for w in words)}


class FilteredHybridStrategy(_Base):
    name = "filtered_hybrid"
    summary = ("A rules-based product router turns the query into a metadata pre-filter "
               "(the product's documents + cross-product guidance), then hybrid search. "
               "Wrong or missing routing falls back to an unfiltered search.")

    def __init__(self, backend: SearchBackend, encoder: QueryEncoder, corpus: KnowledgeCorpus):
        super().__init__(backend, encoder)
        self._corpus = corpus

    def retrieve(self, query: SampleQuery, k: int) -> RetrievalResult:
        products = route_products(query.query)
        docs = self._corpus.docs_for(products | {"all"}) if products else []

        def run(vector):
            if not docs:
                return self._backend.hybrid(query.query, vector, k), 0, "no product detected: unfiltered"
            hits = self._backend.hybrid(query.query, vector, k, filters={"doc_id": docs})
            return hits, 0, f"filter: {', '.join(sorted(products))} (+ all)"
        return self._timed(query, run)


class ParentDocumentStrategy(_Base):
    name = "parent_document"
    summary = ("Hybrid search over small chunks, grouped by document (best chunk score), returning "
               "the whole document: precise matching, complete context for the model.")

    def __init__(self, backend: SearchBackend, encoder: QueryEncoder, corpus: KnowledgeCorpus,
                 candidates: int = 8):
        super().__init__(backend, encoder)
        self._corpus, self._candidates = corpus, candidates

    def retrieve(self, query: SampleQuery, k: int) -> RetrievalResult:
        def run(vector):
            chunks = self._backend.hybrid(query.query, vector, max(self._candidates, k))
            best: dict[str, Hit] = {}
            for hit in chunks:                          # rank order: the first hit per doc is its best
                best.setdefault(hit.doc_id, hit)
            docs = list(best.values())[:k]
            return [Hit(rank, h.doc_id, f"{h.doc_id} (document)", None, h.score,
                        self._corpus.document_text(h.doc_id) or h.text)
                    for rank, h in enumerate(docs, 1)], 0, f"{len(chunks)} chunks -> {len(best)} documents"
        return self._timed(query, run)


RERANK_SYSTEM = ("You rank passages for a claims examiner's question. Reply with JSON only: "
                 '{"order": [passage numbers, most relevant first]}. Include every passage number once.')


class LlmRerankStrategy(_Base):
    name = "llm_rerank"
    summary = ("Hybrid candidates re-ordered by an approved chat model through Unity Gateway "
               "(one extra model call per query: better ordering costs latency and tokens).")

    def __init__(self, backend: SearchBackend, encoder: QueryEncoder, entry: ModelEntry,
                 chat_client: Any, environment: str = "dev", candidates: int = 8):
        super().__init__(backend, encoder)
        self._entry, self._chat, self._environment = entry, chat_client, environment
        self._candidates = candidates
        self.events: list[ModelInvocation] = []

    @staticmethod
    def parse_order(text: str, count: int) -> list[int] | None:
        match = re.search(r"\{.*\}", text or "", re.S)
        try:
            order = json.loads(match.group(0))["order"] if match else None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        if not isinstance(order, list):
            return None
        valid = [int(i) for i in order if isinstance(i, (int, str)) and str(i).isdigit()
                 and 1 <= int(i) <= count]
        valid = list(dict.fromkeys(valid))
        return valid + [i for i in range(1, count + 1) if i not in valid] if valid else None

    def retrieve(self, query: SampleQuery, k: int) -> RetrievalResult:
        def run(vector):
            candidates = self._backend.hybrid(query.query, vector, max(self._candidates, k))
            if not candidates:
                return [], 0, "no candidates"
            passages = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(candidates, 1))
            event = ModelInvocation(event_id=new_event_id(), status="ok", purpose="rerank",
                                    model_key=self._entry.key, model_service=self._entry.model_service,
                                    environment=self._environment, use_case_id=EXPLORER_USE_CASE,
                                    activity="explorer")
            started, order, note = time.perf_counter(), None, ""
            try:
                result = self._chat.complete_messages(
                    [{"role": "system", "content": RERANK_SYSTEM},
                     {"role": "user", "content": f"Question: {query.query}\n\nPassages:\n{passages}"}],
                    tags=event.request_tags())
                event.input_tokens, event.output_tokens = result.prompt_tokens, result.completion_tokens
                event.served_model = result.served_model
                order = self.parse_order(result.text, len(candidates))
                note = "reranked" if order else "unparseable answer: hybrid order kept"
            except ModelInvocationError as exc:
                event.status, event.error_category = "error", exc.category
                note = f"rerank failed ({exc.category}): hybrid order kept"
            event.latency_ms = round((time.perf_counter() - started) * 1000)
            self.events.append(event)
            ordered = [candidates[i - 1] for i in order] if order else candidates
            hits = [Hit(rank, h.doc_id, h.chunk_id, h.chunk_index, h.score, h.text)
                    for rank, h in enumerate(ordered[:k], 1)]
            return hits, 1, note
        return self._timed(query, run)


# ---------------------------------------------------------------- comparison
@dataclass
class StrategySummary:
    strategy: str
    queries: int
    hit_at_1: float
    hit_at_3: float
    mrr: float
    avg_latency_ms: float
    extra_model_calls: int

    def as_row(self) -> dict:
        return {"strategy": self.strategy, "queries": self.queries,
                "hit@1": round(self.hit_at_1, 3), "hit@3": round(self.hit_at_3, 3),
                "MRR": round(self.mrr, 3), "avg_latency_ms": round(self.avg_latency_ms),
                "extra_model_calls": self.extra_model_calls}


def summarise(results: list[RetrievalResult]) -> list[StrategySummary]:
    by_strategy: dict[str, list[RetrievalResult]] = {}
    for r in results:
        if r.query.expected_doc_id:
            by_strategy.setdefault(r.strategy, []).append(r)
    summaries = []
    for name, rs in by_strategy.items():
        n = len(rs)
        summaries.append(StrategySummary(
            name, n, sum(r.hit_at(1) for r in rs) / n, sum(r.hit_at(3) for r in rs) / n,
            sum(r.reciprocal_rank() for r in rs) / n, sum(r.latency_ms for r in rs) / n,
            sum(r.model_calls for r in rs)))
    return summaries


def recommend(summaries: list[StrategySummary], allow_extra_calls: bool = True) -> StrategySummary:
    """Best quality first (MRR, then hit@1), then fewer extra model calls, then lower latency.
    allow_extra_calls=False: best among strategies that make no model call beyond the embedding."""
    pool = [s for s in summaries if allow_extra_calls or s.extra_model_calls == 0]
    if not pool:
        raise ValueError("No strategy results with expected documents to compare")
    return sorted(pool, key=lambda s: (-round(s.mrr, 3), -round(s.hit_at_1, 3),
                                       s.extra_model_calls, s.avg_latency_ms))[0]


def rank_matrix(results: list[RetrievalResult]) -> list[dict]:
    """Per query: the rank of the expected document under each strategy (None = not retrieved)."""
    rows: dict[str, dict] = {}
    for r in results:
        if not r.query.expected_doc_id:
            continue
        row = rows.setdefault(r.query.query_id, {"query_id": r.query.query_id, "style": r.query.style,
                                                 "query": r.query.query,
                                                 "expected_doc_id": r.query.expected_doc_id})
        ranking = r.doc_ranking
        row[r.strategy] = ranking.index(r.query.expected_doc_id) + 1 \
            if r.query.expected_doc_id in ranking else None
    return list(rows.values())


def result_rows(results: list[RetrievalResult], snippet: int = 160) -> list[dict]:
    rows = []
    for r in results:
        for h in r.hits:
            rows.append({"strategy": r.strategy, "query_id": r.query.query_id, "query": r.query.query,
                         "rank": h.rank, "doc_id": h.doc_id, "chunk": h.chunk_id,
                         "score": None if h.score is None else round(float(h.score), 4),
                         "expected": h.doc_id == r.query.expected_doc_id,
                         "text": " ".join(h.text.split())[:snippet], "note": r.note})
    return rows


class RetrievalExplorer:
    """Builds the strategies, runs queries, records usage events."""

    def __init__(self, backend: SearchBackend, encoder: QueryEncoder,
                 strategies: dict[str, RetrievalStrategy], logger: logging.Logger):
        self.backend, self.encoder, self.strategies, self._logger = backend, encoder, strategies, logger
        self.event_sources: list[Any] = []          # more objects with .events (grounded answerer)

    def run(self, strategy_names: list[str], queries: list[SampleQuery], k: int = 3) -> list[RetrievalResult]:
        unknown = set(strategy_names) - set(self.strategies)
        if unknown:
            raise ConfigurationError(f"Unknown retrieval strategies: {sorted(unknown)}")
        self.encoder.encode([q.query for q in queries])           # one embedding call for all
        results = [self.strategies[name].retrieve(q, k) for name in strategy_names for q in queries]
        self._logger.info("retrieval_explorer_run", extra={
            "strategies": strategy_names, "queries": len(queries), "top_k": k,
            "use_case_id": EXPLORER_USE_CASE, "kpi_name": "retrieval_queries",
            "kpi_value": len(results)})
        return results

    def flush_events(self, names, engine: Any, logger: logging.Logger) -> int:
        """Write the recorded model invocations and vector queries (then forget them)."""
        invocations = list(self.encoder.events)
        sources = [*self.strategies.values(), *self.event_sources]
        for source in sources:
            invocations += getattr(source, "events", [])
        written = EventWriter(names, logger).write(engine, invocations=invocations or None,
                                                   queries=self.backend.queries or None)
        self.encoder.events.clear()
        self.backend.queries.clear()
        for source in sources:
            getattr(source, "events", []).clear()
        return written


# ---------------------------------------------------------------- single query (showcase / negative tests)
@dataclass(frozen=True)
class RelevanceGate:
    """Is anything in the index relevant at all? Uses the DENSE top score, whatever the strategy:
    hybrid scores are rank-fused and normalised (the best hit scores ~1.0 even for an off-topic
    query), so they cannot say "nothing relevant". Calibrated 2026-11-27 on this index: the 15
    sample queries scored 0.53-0.72; off-topic questions (pizza, router, cricket, weather) 0.42-0.52.
    Out-of-scope insurance look-alikes (pet, car) scored 0.55-0.61: no score threshold catches
    those; the product router / scope rules must."""
    low: float = 0.50
    high: float = 0.55

    def verdict(self, score: float | None) -> str:
        if score is None:
            return "no results"
        if score < self.low:
            return "NO RELEVANT KNOWLEDGE"
        if score < self.high:
            return "BORDERLINE"
        return "RELEVANT"


@dataclass
class SingleQueryReport:
    query: SampleQuery
    strategy: str
    result: RetrievalResult
    dense_top_score: float | None
    dense_top_doc: str | None
    relevance: str
    routed_products: list[str]
    embedding_ms: int | None
    dense_hits: list[Hit] = field(default_factory=list)     # dense probe: similarity per chunk

    def expected_check(self) -> str:
        expected = self.query.expected_doc_id
        if not expected:
            return "no expected document (free-text query)"
        ranking = self.result.doc_ranking
        return (f"expected {expected}: rank {ranking.index(expected) + 1}" if expected in ranking
                else f"expected {expected}: NOT retrieved")


PROBE_K = 20          # the index holds 13 chunks: the probe scores all of them


def run_single(explorer: "RetrievalExplorer", text: str, strategy: str, k: int = 3,
               samples: list[SampleQuery] | None = None,
               gate: RelevanceGate = RelevanceGate()) -> SingleQueryReport:
    """One query through one strategy, plus a dense relevance check and, when the text is one of
    the sample queries, the expected-document check."""
    text = text.strip()
    if not text:
        raise ConfigurationError("The query is empty")
    known = {q.query: q for q in (samples or [])}
    query = known.get(text) or SampleQuery("custom", text)
    events_before = len(explorer.encoder.events)
    [result] = explorer.run([strategy], [query], k=k)
    new_events = explorer.encoder.events[events_before:]
    vector = explorer.encoder.vector(text)
    probe = explorer.backend.ann(vector, PROBE_K)          # relevance probe (dense scores)
    top = probe[:1]
    return SingleQueryReport(
        query=query, strategy=strategy, result=result,
        dense_top_score=float(top[0].score) if top and top[0].score is not None else None,
        dense_top_doc=top[0].doc_id if top else None,
        relevance=gate.verdict(float(top[0].score) if top and top[0].score is not None else None),
        routed_products=sorted(route_products(text)),
        embedding_ms=new_events[-1].latency_ms if new_events else None, dense_hits=probe)


def build_explorer(index: Any, index_name: str, corpus: KnowledgeCorpus, embed_entry: ModelEntry,
                   embed_client: Any, rerank_entry: ModelEntry, rerank_client: Any,
                   logger: logging.Logger, environment: str = "dev") -> RetrievalExplorer:
    backend = SearchBackend(index, index_name, embed_entry.key)
    encoder = QueryEncoder(embed_entry, embed_client, environment)
    strategies: dict[str, RetrievalStrategy] = {
        "dense": DenseStrategy(backend, encoder),
        "hybrid": HybridStrategy(backend, encoder),
        "filtered_hybrid": FilteredHybridStrategy(backend, encoder, corpus),
        "parent_document": ParentDocumentStrategy(backend, encoder, corpus),
        "llm_rerank": LlmRerankStrategy(backend, encoder, rerank_entry, rerank_client, environment),
    }
    return RetrievalExplorer(backend, encoder, strategies, logger)

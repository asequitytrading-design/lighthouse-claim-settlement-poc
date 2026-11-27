"""End-to-end verification of the AI Marketplace platform setup (notebook 02 and the CLI).

Steps, each reported as PASS / WARN / FAIL with a short detail:
  1 workspace   identity, catalog, SQL warehouse
  2 migrations  Alembic upgrade to head; applied revision == head
  3 objects     every platform table and view exists
  4 catalog     config/model_catalog.yaml synced; current rows == YAML entries
  5 gateway     every approved/candidate model answers through Unity Gateway (embeddings: the
                catalog's dimension); each call carries request tags and is recorded
  6 knowledge   synthetic documents chunked, embedded through the gateway, merged
  7 vectors     Vector Search endpoint + Delta Sync index synced; a gateway-embedded query
                returns the expected document
  8 dashboard   usage views return today's events (gateway rows appear later: system tables lag)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import ModelRoute, Profile
from ..gateway import GatewayChatClient, GatewayEmbeddingClient, build_auth
from .knowledge import (KnowledgeStore, VectorIndexManager, chunk_document, load_documents,
                        search)
from .migrations import SchemaMigrator
from .model_catalog import ModelCatalogSync, ModelEntry, load_model_catalog
from .naming import TABLE_SCHEMAS, PlatformNames
from .observability import EventWriter, ModelInvocation, VectorQuery, new_event_id

VIEWS = ["v_model_catalog_current", "v_gateway_usage", "v_model_usage_daily", "v_eval_trend"]
PROBE_USE_CASE = "platform_setup_verification"
QUERY = ("Which documents should be in a life claim file, and what makes a beneficiary record "
         "conflicting?")
EXPECTED_DOC = "kb-life-evidence"


@dataclass
class Check:
    step: str
    status: str          # PASS | WARN | FAIL | SKIP
    detail: str

    @property
    def ok(self) -> bool:
        return self.status in ("PASS", "WARN", "SKIP")


@dataclass
class VerificationReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, step: str, status: str, detail: str) -> Check:
        check = Check(step, status, detail)
        self.checks.append(check)
        return check

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    def rows(self) -> list[dict[str, str]]:
        return [{"step": c.step, "status": c.status, "detail": c.detail} for c in self.checks]


class SetupVerifier:
    """Runs the verification steps; engine, clients and factories are injectable for tests."""

    def __init__(self, profile: Profile, names: PlatformNames, engine: Any, logger: logging.Logger,
                 environment: str = "dev", vector_client: Any = None,
                 chat_factory: Callable[[ModelEntry], Any] | None = None,
                 embed_factory: Callable[[ModelEntry], Any] | None = None):
        self._profile, self._names, self._engine, self._logger = profile, names, engine, logger
        self._environment = environment
        self._vector_client = vector_client
        self._chat_factory = chat_factory or self._gateway_chat
        self._embed_factory = embed_factory or self._gateway_embed
        self._writer = EventWriter(names, logger)
        self._entries: list[ModelEntry] = []

    # ------------------------------------------------------------------ helpers
    def _query(self, sql: str, params: dict | None = None) -> list[tuple]:
        from sqlalchemy import text
        with self._engine.connect() as connection:
            return [tuple(r) for r in connection.execute(text(sql), params or {}).fetchall()]

    def _route(self, entry: ModelEntry, params: dict | None = None) -> ModelRoute:
        return ModelRoute(alias=entry.key, upstream_model=entry.model_service,
                          params=params or {}, task=entry.task)

    def _gateway_chat(self, entry: ModelEntry):
        # reasoning models (qwen3.5) spend tokens before answering: leave room for that
        route = self._route(entry, {"temperature": 0, "max_tokens": 1024})
        return GatewayChatClient(self._profile.base_url_for(route), route, self._profile.gateway.kind,
                                 build_auth(self._profile.gateway.auth), self._logger,
                                 timeout_seconds=120, max_retries=1)

    def _gateway_embed(self, entry: ModelEntry):
        route = self._route(entry)
        return GatewayEmbeddingClient(self._profile.base_url_for(route), route,
                                      self._profile.gateway.kind,
                                      build_auth(self._profile.gateway.auth), self._logger,
                                      timeout_seconds=120, max_retries=1)

    def _event(self, entry: ModelEntry, purpose: str) -> ModelInvocation:
        return ModelInvocation(event_id=new_event_id(), status="ok", purpose=purpose,
                               model_key=entry.key, model_service=entry.model_service,
                               task=entry.task, environment=self._environment,
                               use_case_id=PROBE_USE_CASE)

    def _embedding_entry(self) -> ModelEntry:
        key = self._profile.platform.embedding_model if self._profile.platform else "embed-default"
        for entry in self._entries:
            if entry.key == key and entry.allows("embedding"):
                return entry
        raise LookupError(f"Embedding model {key} is not an approved catalog entry")

    # ------------------------------------------------------------------ steps
    def run(self, migrate: bool = True, vectors: bool = True,
            index_timeout_s: int = 1800) -> VerificationReport:
        report = VerificationReport()
        steps = [("1 workspace", self.step_workspace),
                 ("2 migrations", lambda: self.step_migrations(migrate)),
                 ("3 objects", self.step_objects),
                 ("4 catalog", self.step_catalog),
                 ("5 gateway", self.step_gateway),
                 ("6 knowledge", self.step_knowledge),
                 ("7 vectors", (lambda: self.step_vectors(index_timeout_s)) if vectors else None),
                 ("8 dashboard", self.step_dashboard)]
        blocked = False
        for name, step in steps:
            if step is None:
                report.add(name, "SKIP", "disabled for this run")
                continue
            if blocked:
                report.add(name, "SKIP", "an earlier required step failed")
                continue
            try:
                status, detail = step()
            except Exception as exc:  # each step reports its own failure; later steps are skipped
                self._logger.exception("platform_verification_step_failed", extra={
                    "step": name, "error_type": type(exc).__name__})
                cause = f" (cause: {exc.__cause__})" if exc.__cause__ else ""
                status, detail = "FAIL", f"{type(exc).__name__}: {exc}{cause}"[:600]
            report.add(name, status, detail)
            blocked = blocked or (status == "FAIL" and name[0] in "1234")
        self._logger.info("platform_verification_completed", extra={
            "passed": report.passed, "checks": len(report.checks),
            "failed": sum(not c.ok for c in report.checks),
            "kpi_name": "platform_verification_pass", "kpi_value": int(report.passed)})
        return report

    def step_workspace(self) -> tuple[str, str]:
        user, catalog = self._query("SELECT current_user(), current_catalog()")[0]
        found = self._query(f"SHOW CATALOGS LIKE '{self._names.catalog}'")
        if not found:
            return "FAIL", f"catalog {self._names.catalog} not found or not visible to {user}"
        return "PASS", f"user {user.split('@')[0]}@…, catalog {self._names.catalog}, warehouse session ok"

    def step_migrations(self, migrate: bool) -> tuple[str, str]:
        migrator = SchemaMigrator(self._names, self._logger, engine=self._engine)
        run_id = migrator.upgrade("head") if migrate else None
        current, heads = migrator.current(), migrator.heads()
        if current != heads:
            return "FAIL", f"applied {current or 'base'} but head is {heads}"
        return "PASS", f"at {', '.join(current)}" + (f" (migration run {run_id})" if run_id else "")

    def step_objects(self) -> tuple[str, str]:
        schemas = ", ".join(f"'{s}'" for s in self._names.schemas)
        rows = self._query(f"SELECT table_name FROM {self._names.catalog}.information_schema.tables "
                           f"WHERE table_schema IN ({schemas})")
        present = {r[0] for r in rows}
        missing = sorted((set(TABLE_SCHEMAS) | set(VIEWS)) - present)
        if missing:
            return "FAIL", f"missing: {missing}"
        return "PASS", f"{len(TABLE_SCHEMAS)} tables and {len(VIEWS)} views in {', '.join(self._names.schemas)}"

    def step_catalog(self) -> tuple[str, str]:
        self._entries = load_model_catalog()
        counts = ModelCatalogSync(self._names, self._logger).sync(self._engine, self._entries)
        current = self._query(f"SELECT count(*) FROM {self._names.table('model_catalog')} WHERE is_current")[0][0]
        if current != len(self._entries):
            return "FAIL", f"{current} current rows for {len(self._entries)} YAML entries"
        return "PASS", (f"{current} models current ({counts.get('opened', 0)} new/changed this sync); "
                        f"approved: {', '.join(e.key for e in self._entries if e.status == 'approved')}")

    def step_gateway(self) -> tuple[str, str]:
        events, failures, notes = [], [], []
        for entry in [e for e in self._entries if e.status in ("approved", "candidate")]:
            event = self._event(entry, "probe")
            started = time.perf_counter()
            try:
                if entry.task == "chat":
                    result = self._chat_factory(entry).complete_messages(
                        [{"role": "user", "content": "Reply with the single word OK."}],
                        tags=event.request_tags())
                    event.input_tokens, event.output_tokens = result.prompt_tokens, result.completion_tokens
                    event.served_model = result.served_model
                else:
                    result = self._embed_factory(entry).embed(["synthetic claim file"],
                                                             tags=event.request_tags())
                    event.input_count, event.embedding_dimensions = 1, result.dimensions
                    event.input_tokens, event.served_model = result.prompt_tokens, result.served_model
                    if result.dimensions != entry.embedding_dimensions:
                        raise ValueError(f"dimension {result.dimensions} != catalog "
                                         f"{entry.embedding_dimensions}")
                notes.append(entry.key)
            except Exception as exc:
                event.status = "error"
                event.error_category = getattr(exc, "category", type(exc).__name__)
                (failures if entry.status == "approved" else notes).append(
                    f"{entry.key} ({event.error_category})")
            event.latency_ms = round((time.perf_counter() - started) * 1000)
            events.append(event)
        self._writer.write(self._engine, invocations=events)
        if failures:
            return "FAIL", f"approved models failing: {failures}; ok: {notes}"
        return "PASS", f"{len(events)} probes recorded with request tags; ok: {notes}"

    def step_knowledge(self) -> tuple[str, str]:
        entry = self._embedding_entry()
        docs = load_documents()
        chunks = [c for d in docs for c in chunk_document(d)]
        event = self._event(entry, "embedding_index")
        started = time.perf_counter()
        embeddings = self._embed_factory(entry).embed([c.text for c in chunks], tags=event.request_tags())
        event.latency_ms = round((time.perf_counter() - started) * 1000)
        event.input_count, event.embedding_dimensions = len(chunks), embeddings.dimensions
        event.input_tokens, event.served_model = embeddings.prompt_tokens, embeddings.served_model
        store = KnowledgeStore(self._names, self._logger)
        from sqlalchemy import text
        with self._engine.connect() as connection:
            for statement in (store.document_merge(docs),
                              store.chunk_merge(chunks, embeddings.vectors, entry.key, entry.model_service),
                              store.stale_chunk_delete(chunks)):
                connection.execute(text(statement.replace(":", "\\:")))
        self._writer.write(self._engine, invocations=[event])
        stored = self._query(f"SELECT count(*) FROM {self._names.table('document_chunk')} "
                             "WHERE embedding IS NOT NULL")[0][0]
        self._chunk_count = stored
        return "PASS", (f"{len(docs)} documents, {len(chunks)} chunks embedded with {entry.key} "
                        f"({embeddings.dimensions}-d, one gateway call); {stored} chunks stored")

    def step_vectors(self, timeout_s: int) -> tuple[str, str]:
        entry = self._embedding_entry()
        manager = VectorIndexManager(self._names, self._logger, client=self._vector_client)
        requested = self._profile.platform.vector_search_endpoint if self._profile.platform else "lighthouse-vs"
        endpoint = manager.ensure_endpoint(requested)
        index = manager.ensure_index(endpoint, entry.embedding_dimensions)
        manager.sync_and_wait(index, expected_rows=getattr(self, "_chunk_count", 1), timeout_s=timeout_s,
                              require_sync=not manager.created_index)
        event = self._event(entry, "embedding_query")
        query_vector = self._embed_factory(entry).embed([QUERY], tags=event.request_tags()).vectors[0]
        started = time.perf_counter()
        hits = search(index, query_vector, num_results=3)
        latency = round((time.perf_counter() - started) * 1000)
        found = [h.get("doc_id") for h in hits]
        query = VectorQuery(event_id=event.event_id, status="ok" if hits else "error",
                            index_name=self._names.vector_index, embedding_model_key=entry.key,
                            num_results_requested=3, use_case_id=PROBE_USE_CASE,
                            num_results=len(hits), latency_ms=latency)
        event.input_count, event.embedding_dimensions = 1, len(query_vector)
        self._writer.write(self._engine, invocations=[event], queries=[query])
        if EXPECTED_DOC not in found:
            return "WARN", f"endpoint {endpoint}: top results {found} did not include {EXPECTED_DOC}"
        return "PASS", f"endpoint {endpoint}, index {self._names.vector_index}: top results {found}"

    def step_dashboard(self) -> tuple[str, str]:
        obs = f"{self._names.catalog}.{self._names.observability}"
        today = self._query(
            f"SELECT coalesce(sum(calls), 0), coalesce(sum(gateway_confirmed), 0) FROM {obs}.v_model_usage_daily "
            f"WHERE use_case_id = :use_case AND event_date = current_date()", {"use_case": PROBE_USE_CASE})[0]
        if not today[0]:
            return "FAIL", "v_model_usage_daily has no events for today's probes"
        lag = "" if today[1] else " (gateway usage rows arrive later: system tables lag; re-run step 8)"
        return ("PASS" if today[1] else "WARN",
                f"{today[0]} events today, {today[1]} matched in system.ai_gateway.usage{lag}")

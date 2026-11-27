"""Knowledge base on Mosaic AI Vector Search with embeddings computed through Unity Gateway.

Flow (ADR-013): synthetic documents -> chunks -> embeddings via the gateway (embed-default) ->
MERGE into <kn>.document_chunk (change data feed on) -> Delta Sync index with a self-managed
embedding column -> queries embed the question through the gateway and search by vector.

Self-managed embeddings keep every embedding call on Unity Gateway (usage, limits, one audit
trail); Vector Search's managed embeddings would call a serving endpoint directly. Free Edition
allows one Vector Search endpoint (one unit) and Delta Sync indexes only, which this design fits.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import PROJECT_ROOT
from ..exceptions import ConfigurationError, LighthouseError
from .naming import PlatformNames
from .sql import float_array, literal

KNOWLEDGE_FILE = PROJECT_ROOT / "data" / "knowledge" / "examiner_guidance.jsonl"
INDEX_COLUMNS = ["chunk_id", "doc_id", "chunk_index", "text"]


class KnowledgeError(LighthouseError):
    """The knowledge base or vector index could not be built or queried."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    doc_type: str
    product_line: str
    version: str
    text: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


def load_documents(path: Path | None = None) -> list[Document]:
    path = Path(path) if path else KNOWLEDGE_FILE
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read knowledge documents {path}") from exc
    docs = [Document(**{k: str(r[k]) for k in
                        ("doc_id", "title", "doc_type", "product_line", "version", "text")})
            for r in rows]
    if len({d.doc_id for d in docs}) != len(docs):
        raise ConfigurationError(f"Duplicate doc_id in {path.name}")
    return docs


def chunk_document(doc: Document, max_chars: int = 450) -> list[Chunk]:
    """Paragraph-preserving chunks up to max_chars (a paragraph longer than that is split)."""
    pieces: list[str] = []
    for paragraph in [p.strip() for p in doc.text.split("\n\n") if p.strip()]:
        while len(paragraph) > max_chars:
            cut = paragraph.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        pieces.append(paragraph)
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return [Chunk(f"{doc.doc_id}-{i:03d}", doc.doc_id, i, text) for i, text in enumerate(chunks)]


class KnowledgeStore:
    """Writes documents and embedded chunks to the knowledge tables via a SQL warehouse."""

    def __init__(self, names: PlatformNames, logger: logging.Logger):
        self._names, self._logger = names, logger

    def document_merge(self, docs: list[Document]) -> str:
        table = self._names.table("source_document")
        rows = ",\n  ".join("(" + ", ".join(literal(v) for v in (
            d.doc_id, d.title, d.doc_type, d.product_line, d.version, d.content_hash)) + ")"
                            for d in docs)
        return (f"MERGE INTO {table} t USING (SELECT * FROM VALUES {rows} AS s(doc_id, title, "
                "doc_type, product_line, version, content_hash)) s ON t.doc_id = s.doc_id "
                "WHEN MATCHED AND t.content_hash <> s.content_hash THEN UPDATE SET "
                "title = s.title, doc_type = s.doc_type, product_line = s.product_line, "
                "version = s.version, content_hash = s.content_hash, ingested_at = current_timestamp() "
                "WHEN NOT MATCHED THEN INSERT (doc_id, title, doc_type, product_line, version, "
                "source_uri, content_hash, classification, status, ingested_at) VALUES (s.doc_id, "
                "s.title, s.doc_type, s.product_line, s.version, 'repo:data/knowledge', "
                "s.content_hash, 'synthetic', 'active', current_timestamp())")

    def chunk_merge(self, chunks: list[Chunk], vectors: list[list[float]], model_key: str,
                    model_service: str) -> str:
        if len(chunks) != len(vectors):
            raise KnowledgeError(f"{len(chunks)} chunks but {len(vectors)} embeddings")
        table = self._names.table("document_chunk")
        rows = ",\n  ".join("(" + ", ".join([
            literal(c.chunk_id), literal(c.doc_id), str(c.chunk_index), literal(c.text),
            str(len(c.text)), float_array(v), literal(c.content_hash)]) + ")"
                            for c, v in zip(chunks, vectors))
        dims = len(vectors[0]) if vectors else 0
        return (f"MERGE INTO {table} t USING (SELECT * FROM VALUES {rows} AS s(chunk_id, doc_id, "
                "chunk_index, text, char_count, embedding, content_hash)) s ON t.chunk_id = s.chunk_id "
                "WHEN MATCHED AND (t.content_hash <> s.content_hash OR "
                f"t.embedding_model_service <> {literal(model_service)}) THEN UPDATE SET "
                "text = s.text, char_count = s.char_count, embedding = s.embedding, "
                f"embedding_model_key = {literal(model_key)}, "
                f"embedding_model_service = {literal(model_service)}, embedding_dimensions = {dims}, "
                "content_hash = s.content_hash, embedded_at = current_timestamp() "
                "WHEN NOT MATCHED THEN INSERT (chunk_id, doc_id, chunk_index, text, char_count, "
                "embedding, embedding_model_key, embedding_model_service, embedding_dimensions, "
                "content_hash, embedded_at) VALUES (s.chunk_id, s.doc_id, s.chunk_index, s.text, "
                f"s.char_count, s.embedding, {literal(model_key)}, {literal(model_service)}, {dims}, "
                "s.content_hash, current_timestamp())")

    def stale_chunk_delete(self, chunks: list[Chunk]) -> str:
        """Chunks of these documents that no longer exist (a document got shorter)."""
        table = self._names.table("document_chunk")
        docs = ", ".join(sorted({literal(c.doc_id) for c in chunks}))
        keep = ", ".join(literal(c.chunk_id) for c in chunks)
        return f"DELETE FROM {table} WHERE doc_id IN ({docs}) AND chunk_id NOT IN ({keep})"


class VectorIndexManager:
    """Endpoint and Delta Sync index lifecycle; the Vector Search client is injectable."""

    def __init__(self, names: PlatformNames, logger: logging.Logger, client: Any = None):
        self._names, self._logger = names, logger
        self._client = client
        self.created_index = False   # True when ensure_index built it now (its first build syncs)

    def _vs(self):
        if self._client is None:
            self._client = vector_search_client()
        return self._client

    def ensure_endpoint(self, name: str) -> str:
        """The named endpoint; on Free Edition (one endpoint) reuse whichever already exists."""
        endpoints = self._vs().list_endpoints().get("endpoints", []) or []
        names = [e.get("name") for e in endpoints]
        if name in names:
            return name
        if endpoints:
            self._logger.warning("vector_endpoint_reused", extra={
                "requested": name, "reused": names[0],
                "reason": "workspace already has an endpoint (Free Edition allows one)"})
            return names[0]
        self._logger.info("vector_endpoint_creating", extra={"endpoint": name})
        self._vs().create_endpoint_and_wait(name=name, endpoint_type="STANDARD")
        return name

    def ensure_index(self, endpoint: str, dimensions: int) -> Any:
        index_name = self._names.vector_index
        try:
            return self._vs().get_index(endpoint_name=endpoint, index_name=index_name)
        except Exception as exc:  # the client raises a generic Exception for "not found"
            if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
                raise KnowledgeError(f"Cannot read vector index {index_name}") from exc
        self._logger.info("vector_index_creating", extra={"index": index_name, "endpoint": endpoint})
        self.created_index = True
        return self._vs().create_delta_sync_index(
            endpoint_name=endpoint, index_name=index_name,
            source_table_name=self._names.table("document_chunk"),
            pipeline_type="TRIGGERED", primary_key="chunk_id",
            embedding_dimension=dimensions, embedding_vector_column="embedding")

    def sync_and_wait(self, index: Any, expected_rows: int, timeout_s: int = 1800,
                      poll_s: int = 20, sleep: Callable[[float], None] = time.sleep,
                      require_sync: bool = True) -> dict:
        """Wait until the index holds expected_rows rows, triggering one sync first when needed.

        A newly created index builds from the whole table (require_sync=False). While its build
        pipeline runs, the API answers "not ready to sync": that means keep waiting.
        """
        deadline = time.monotonic() + timeout_s
        triggered = not require_sync
        while time.monotonic() < deadline:
            status = index.describe().get("status") or {}
            state = str(status.get("detailed_state", ""))
            if "FAILED" in state:
                raise KnowledgeError(f"Vector index failed: {state} {status.get('message', '')}")
            if status.get("ready"):
                if int(status.get("indexed_row_count") or 0) >= expected_rows and triggered:
                    return status
                if not triggered:
                    try:
                        index.sync()
                        triggered = True
                    except Exception as exc:
                        message = str(exc).lower()
                        if "not ready to sync" in message or "state running" in message:
                            pass                      # build still running: poll again
                        elif "already" in message or "in progress" in message:
                            triggered = True          # a sync is already running
                        else:
                            raise KnowledgeError("Vector index sync failed") from exc
            sleep(poll_s)
        raise KnowledgeError(f"Vector index not ready with {expected_rows} rows after {timeout_s} s")


def vector_search_client():
    """Notebook/job: runtime auth. Elsewhere (CLI, CI) the client accepts only a token or SP
    secret, so pass the bearer token databricks-sdk unified auth produced (OAuth included)."""
    import os

    from databricks.vector_search.client import VectorSearchClient
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        return VectorSearchClient(disable_notice=True)
    from databricks.sdk import WorkspaceClient
    config = WorkspaceClient().config
    header = config.authenticate().get("Authorization", "")
    if not header.startswith("Bearer "):
        raise ConfigurationError("databricks-sdk auth did not produce a bearer token for Vector Search")
    return VectorSearchClient(workspace_url=config.host, personal_access_token=header[7:],
                              disable_notice=True)


def search(index: Any, query_vector: list[float], num_results: int = 3) -> list[dict]:
    """Similarity search by vector; returns rows as dicts with a score."""
    result = index.similarity_search(query_vector=query_vector, columns=INDEX_COLUMNS,
                                     num_results=num_results)
    columns = [c["name"] for c in result.get("manifest", {}).get("columns", [])]
    return [dict(zip(columns, row)) for row in result.get("result", {}).get("data_array", [])]


def new_event_id() -> str:
    return str(uuid.uuid4())

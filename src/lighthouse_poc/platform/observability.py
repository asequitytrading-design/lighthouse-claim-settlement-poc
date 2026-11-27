"""Usage events for the AI Marketplace dashboards (ADR-013).

Each model call made by a marketplace application gets an event ID. The ID is sent to Unity
Gateway as request tag `lh_event_id` (with use case, purpose and environment) and stored in
<obs>.model_invocation with domain context the gateway cannot know: prompt version, evaluation
run, synthetic case ID. The dashboard view joins both, preferring the gateway's token and latency
values. Never store prompt text, responses, claim narratives or health details here.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

from .naming import PlatformNames
from .sql import literal


def new_event_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class ModelInvocation:
    event_id: str
    status: str                         # ok | error
    purpose: str                        # app | judge | embedding_index | embedding_query | probe
    model_key: str
    model_service: str
    task: str = "chat"
    event_time: datetime = field(default_factory=utc_now)
    environment: str | None = None
    use_case_id: str | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    app_version: str | None = None
    git_commit: str | None = None
    trace_id: str | None = None
    run_id: str | None = None
    case_id: str | None = None
    error_category: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_count: int | None = None
    embedding_dimensions: int | None = None
    served_model: str | None = None

    def request_tags(self) -> dict[str, str]:
        """Tags for the Databricks-Ai-Gateway-Request-Tags header (non-secret identifiers)."""
        tags = {"lh_event_id": self.event_id, "lh_purpose": self.purpose,
                "lh_use_case": self.use_case_id, "lh_environment": self.environment,
                "lh_prompt_version": self.prompt_version}
        return {k: v for k, v in tags.items() if v}


@dataclass
class VectorQuery:
    event_id: str
    status: str
    index_name: str
    embedding_model_key: str
    num_results_requested: int
    event_time: datetime = field(default_factory=utc_now)
    use_case_id: str | None = None
    num_results: int | None = None
    latency_ms: int | None = None
    error_category: str | None = None


def _insert(table: str, rows: list[Any]) -> str:
    columns = [f.name for f in fields(rows[0])]
    values = ",\n  ".join("(" + ", ".join(literal(v) for v in asdict(r).values()) + ")" for r in rows)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES {values}"


class EventWriter:
    """Appends usage events through a SQL warehouse engine (batched per call)."""

    def __init__(self, names: PlatformNames, logger: logging.Logger):
        self._names, self._logger = names, logger

    def invocation_insert(self, rows: list[ModelInvocation]) -> str:
        return _insert(self._names.table("model_invocation"), rows)

    def vector_query_insert(self, rows: list[VectorQuery]) -> str:
        return _insert(self._names.table("vector_query"), rows)

    def write(self, engine: Any, invocations: list[ModelInvocation] | None = None,
              queries: list[VectorQuery] | None = None) -> int:
        from sqlalchemy import text
        statements = []
        if invocations:
            statements.append(self.invocation_insert(invocations))
        if queries:
            statements.append(self.vector_query_insert(queries))
        with engine.connect() as connection:
            for statement in statements:
                connection.execute(text(statement.replace(":", "\\:")))
        written = len(invocations or []) + len(queries or [])
        self._logger.info("usage_events_written", extra={
            "invocations": len(invocations or []), "vector_queries": len(queries or []),
            "kpi_name": "usage_events_written", "kpi_value": written})
        return written

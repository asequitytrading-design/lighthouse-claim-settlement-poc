"""Writes the model calls of a run to <obs>.model_invocation (ADR-013).

`tracked_run` is what try, eval, the evaluation job and the notebooks wrap around their work:

    with tracked_run(profile, logger, activity="eval", prompt_version="3") as tracker:
        ...                                  # every gateway call is tagged and recorded
        tracker.update(run_id=run.info.run_id)

On exit the recorded calls are written in one INSERT. Recording happens only on the Databricks
profile (Unity Gateway + a platform section) and can be switched off with
LIGHTHOUSE_RECORD_USAGE=false. Writing usage must never lose evaluation evidence, so a failed
write is logged as an error (`usage_events_write_failed`) and reported, not raised.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from ..config import Profile
from ..usage import CallRecord, UsageTracker, activate, context_from_profile
from .observability import EventWriter, ModelInvocation


def should_record(profile: Profile) -> bool:
    return (profile.platform is not None and profile.gateway.kind == "unity_gateway"
            and os.getenv("LIGHTHOUSE_RECORD_USAGE", "true").lower() not in ("0", "false", "no"))


def to_invocation(record: CallRecord) -> ModelInvocation:
    c = record.context
    return ModelInvocation(
        event_id=record.event_id, status=record.status, purpose=c.purpose or "app",
        model_key=record.model_key, model_service=record.model_service or "", task=record.task,
        event_time=record.event_time, environment=c.environment, use_case_id=c.use_case_id,
        prompt_name=c.prompt_name, prompt_version=c.prompt_version, app_version=c.app_version,
        git_commit=c.git_commit, trace_id=record.trace_id, run_id=c.run_id, case_id=c.case_id,
        error_category=record.error_category, latency_ms=record.latency_ms,
        input_tokens=record.input_tokens, output_tokens=record.output_tokens,
        input_count=record.input_count, embedding_dimensions=record.embedding_dimensions,
        served_model=record.served_model, activity=c.activity)


@dataclass(frozen=True)
class FlushResult:
    status: str          # written | skipped | error
    written: int
    detail: str


class PlatformWriter:
    """Appends rows to platform tables after checking the schemas are at the Alembic head.

    Used for call events (model_invocation), evaluation results (eval_run, eval_case_result) and
    review decisions (release_decision). Never raises: platform records must not lose the run's
    MLflow evidence, so failures are logged as errors and reported in the FlushResult."""

    def __init__(self, profile: Profile, logger: logging.Logger,
                 engine_factory: Callable[[], Any] | None = None):
        self._profile, self._logger = profile, logger
        self._engine_factory = engine_factory

    def _names(self):
        from .runtime import platform_names
        return platform_names(self._profile)

    def _engine(self, names):
        if self._engine_factory:
            return self._engine_factory()
        from .runtime import warehouse_engine
        return warehouse_engine(self._profile, names, self._logger)

    def write(self, rows_by_table: dict[str, list], what: str) -> FlushResult:
        """INSERT each table's rows (dataclasses whose fields are the columns)."""
        rows_by_table = {t: r for t, r in rows_by_table.items() if r}
        count = sum(len(r) for r in rows_by_table.values())
        if not count:
            return FlushResult("skipped", 0, f"no {what}")
        try:
            from sqlalchemy import text

            from .migrations import SchemaMigrator
            from .observability import insert_statement
            names = self._names()
            engine = self._engine(names)
            try:
                migrator = SchemaMigrator(names, self._logger, engine=engine)
                current, heads = migrator.current(), migrator.heads()
                if current != heads:
                    detail = (f"platform schemas of prefix '{names.prefix}' are at {current or 'base'}, "
                              f"head is {heads}: run notebook 02/04 or `lighthouse platform db "
                              f"upgrade`; {count} {what} not written")
                    self._logger.warning("platform_records_skipped", extra={"reason": detail})
                    return FlushResult("skipped", 0, detail)
                with engine.connect() as connection:
                    for table, rows in rows_by_table.items():
                        statement = insert_statement(names.table(table), rows)
                        connection.execute(text(statement.replace(":", "\\:")))
            finally:
                if hasattr(engine, "dispose"):
                    engine.dispose()
            tables = ", ".join(f"{names.table(t)} ({len(r)})" for t, r in rows_by_table.items())
            self._logger.info("platform_records_written", extra={
                "what": what, "kpi_name": "platform_records_written", "kpi_value": count})
            return FlushResult("written", count, f"{count} {what} -> {tables}")
        except Exception as exc:
            self._logger.error("platform_records_write_failed", exc_info=True, extra={
                "error_type": type(exc).__name__, "error_category": "platform_records",
                "what": what, "rows": count})
            return FlushResult("error", 0, f"{type(exc).__name__}: {exc}"[:300])


class UsageRecorder(PlatformWriter):
    """Writes a run's call records to model_invocation."""

    def flush(self, records: list[CallRecord]) -> FlushResult:
        if not records:
            return FlushResult("skipped", 0, "no calls recorded")
        return self.write({"model_invocation": [to_invocation(r) for r in records]}, "call events")


@contextmanager
def tracked_run(profile: Profile, logger: logging.Logger, activity: str,
                recorder: UsageRecorder | None = None, **context: Any) -> Iterator[UsageTracker]:
    """Tag every gateway call in the block with the run's context and record it."""
    tracker = UsageTracker(context_from_profile(profile, activity=activity, **context),
                           recording=should_record(profile))
    with activate(tracker):
        try:
            yield tracker
        finally:
            if tracker.recording:
                result = (recorder or UsageRecorder(profile, logger)).flush(tracker.drain())
                tracker.last_flush = result
                logger.info("usage_run_recorded", extra={
                    "activity": activity, "status": result.status, "events": result.written,
                    "detail": result.detail})

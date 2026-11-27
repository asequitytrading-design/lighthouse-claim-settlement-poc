"""Usage attribution for every model call (ADR-013): request tags and call records.

Every gateway client asks the usage tracker for the tags of a call and reports the outcome:

  tags   sent as the Databricks-Ai-Gateway-Request-Tags header (Unity Gateway only), so each
         row of system.ai_gateway.usage names its use case, purpose, activity, environment,
         model key, prompt version and MLflow run, even when no application event is written
  record one CallRecord per call, kept by a *recording* tracker and written by the platform
         (ai_observability.model_invocation) at the end of a run (lighthouse_poc.platform.usage_recorder)

Context comes from three levels, most specific last:
  1. the profile (platform.use_case, platform.environment): the default for every client
  2. the active run (`activate(tracker)` in try / eval / jobs / notebooks): activity, prompt
     version, MLflow run ID. Process-wide on purpose: MLflow evaluates cases in worker threads.
  3. the current case (`case_scope(case_id)`): set by the assistant inside the worker thread

Never put prompt text, responses, claim narratives or health details in tags or records.
"""
from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

TAG_NAMES = {"use_case_id": "lh_use_case", "purpose": "lh_purpose", "environment": "lh_environment",
             "activity": "lh_activity", "prompt_version": "lh_prompt_version", "run_id": "lh_run_id"}

# Activities: what kind of work made the call (dashboard filter "activity").
ACTIVITIES = {"app", "try", "eval", "demo", "explorer", "verify", "notebook"}


@dataclass(frozen=True)
class UsageContext:
    use_case_id: str | None = None
    purpose: str | None = None              # app | judge | embedding_index | embedding_query | rerank | grounded_answer | probe
    environment: str | None = None          # dev | ci | prod
    activity: str | None = None             # see ACTIVITIES
    prompt_name: str | None = None
    prompt_version: str | None = None
    run_id: str | None = None
    case_id: str | None = None              # synthetic case ID only
    app_version: str | None = None
    git_commit: str | None = None

    def merged(self, **overrides: Any) -> "UsageContext":
        """A copy with every non-empty override applied (None keeps the current value)."""
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown usage context fields: {sorted(unknown)}")
        return replace(self, **{k: str(v) for k, v in overrides.items() if v not in (None, "")})

    def tags(self) -> dict[str, str]:
        return {tag: getattr(self, name) for name, tag in TAG_NAMES.items() if getattr(self, name)}


def context_from_profile(profile: Any, **overrides: Any) -> UsageContext:
    """Profile defaults: platform.use_case / platform.environment (Databricks profile only)."""
    platform = getattr(profile, "platform", None)
    base = UsageContext(use_case_id=getattr(platform, "use_case", None),
                        environment=getattr(platform, "environment", None), purpose="app",
                        activity="app")
    return base.merged(**overrides)


@dataclass
class CallRecord:
    """Outcome of one model call (no prompt or response text)."""
    event_id: str
    context: UsageContext
    model_key: str
    model_service: str | None
    task: str = "chat"
    status: str = "ok"
    error_category: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_count: int | None = None
    embedding_dimensions: int | None = None
    served_model: str | None = None
    trace_id: str | None = None
    event_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


_CASE: ContextVar[UsageContext | None] = ContextVar("lighthouse_usage_case", default=None)


@contextmanager
def case_scope(**fields_: Any) -> Iterator[None]:
    """Per-case context (case ID, prompt version) for calls made in this thread / task."""
    current = _CASE.get() or UsageContext()
    token = _CASE.set(current.merged(**fields_))
    try:
        yield
    finally:
        _CASE.reset(token)


class UsageTracker:
    """Tags calls and (when recording) keeps their records; thread-safe."""

    def __init__(self, context: UsageContext | None = None, recording: bool = False):
        self.context = context or UsageContext()
        self.recording = recording
        self.last_flush: Any = None          # set by the platform recorder after writing
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()

    def update(self, **fields_: Any) -> None:
        """Change the run context, e.g. once the MLflow run ID is known."""
        with self._lock:
            self.context = self.context.merged(**fields_)

    def call_context(self, purpose: str | None = None) -> UsageContext:
        context = self.context
        case = _CASE.get()
        if case:
            context = context.merged(**{f.name: getattr(case, f.name) for f in fields(case)})
        return context.merged(purpose=purpose) if purpose else context

    def begin(self, model_key: str, explicit: Mapping[str, Any] | None = None,
              purpose: str | None = None) -> tuple[str, UsageContext, dict[str, str], bool]:
        """Event ID, context and tags for one call. A caller that passes its own lh_event_id
        records the event itself (verification, demo traffic, retrieval explorer): owned=True."""
        explicit = {k: str(v) for k, v in (explicit or {}).items() if v not in (None, "")}
        owned = "lh_event_id" in explicit
        event_id = explicit.get("lh_event_id") or str(uuid.uuid4())
        context = self.call_context(purpose)
        tags = {**context.tags(), "lh_model_key": model_key, "lh_event_id": event_id, **explicit}
        return event_id, context, tags, owned

    def finish(self, record: CallRecord, owned: bool) -> None:
        if self.recording and not owned:
            with self._lock:
                self._records.append(record)

    def judge_tags(self, judge_model_key: str) -> dict[str, str]:
        """Tags for LLM judge calls made inside MLflow evaluation (no per-call hook there)."""
        return {**self.context.merged(purpose="judge", case_id=None).tags(),
                "lh_model_key": judge_model_key}

    def drain(self) -> list[CallRecord]:
        with self._lock:
            records, self._records = self._records, []
        return records


_ACTIVE: list[UsageTracker] = []
_ACTIVE_LOCK = threading.Lock()


def active_tracker() -> UsageTracker | None:
    with _ACTIVE_LOCK:
        return _ACTIVE[-1] if _ACTIVE else None


@contextmanager
def activate(tracker: UsageTracker) -> Iterator[UsageTracker]:
    """Make `tracker` the one every gateway client uses until the block ends (process-wide)."""
    with _ACTIVE_LOCK:
        _ACTIVE.append(tracker)
    try:
        yield tracker
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.remove(tracker)

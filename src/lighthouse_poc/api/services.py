"""Services behind the demo API (no HTTP here): chat, playground, models, usage, evals, health.

Usage attribution: every chat turn runs inside tracked_run (activity `app`, use case from
settings), every playground call inside tracked_run (activity `playground`), so the dashboard
shows the demo's calls per model like any other application. The usage tracker is process-wide
(ADR-015), so chat turns and playground calls are serialised with one lock: correct attribution
for a single-user demo. A multi-user deployment passes a tracker per request instead.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from ..config import Profile
from ..exceptions import LighthouseError
from ..platform.usage_recorder import FlushResult, UsageRecorder, tracked_run
from ..sessions.store import Message, SessionStore, _now, history_for_agent
from .settings import AppSettings

APP_ROLES = {"app"}
LISTED_STATUSES = {"approved", "candidate"}


class BackgroundRecorder(UsageRecorder):
    """Writes a turn's call events in a background thread, so the response is not held open
    by the SQL warehouse insert (a few seconds). Failures are logged by UsageRecorder."""

    def flush(self, records):
        if not records:
            return FlushResult("skipped", 0, "no calls recorded")
        threading.Thread(target=super().flush, args=(records,), name="usage-flush", daemon=True).start()
        return FlushResult("queued", 0, f"{len(records)} call events queued")


@dataclass
class CachedQuery:
    expires: float
    rows: list[dict]


class AppServices:
    def __init__(self, profile: Profile, settings: AppSettings, store: SessionStore,
                 logger: logging.Logger, copilot_factory: Callable[[], Any] | None = None,
                 model_factory: Callable[[str], Any] | None = None,
                 query: Callable[[str], list[dict]] | None = None, host: str | None = None):
        self.profile, self.settings, self.store, self.logger = profile, settings, store, logger
        self._copilot_factory = copilot_factory
        self._copilot = None
        self._model_factory = model_factory
        self._query_fn = query
        self._engine = None
        self._lock = threading.Lock()                # one tracked turn at a time (see module doc)
        self._cache: dict[str, CachedQuery] = {}
        self.host = host or (profile.gateway.base_url or "").split("/ai-gateway")[0]

    # ------------------------------------------------------------ chat
    def copilot(self):
        if self._copilot is None:
            if self._copilot_factory is None:
                from ..agents.claims_copilot.agent import build_copilot
                self._copilot = build_copilot(self.profile, self.logger)
            else:
                self._copilot = self._copilot_factory()
        return self._copilot

    def chat(self, message: str, model: str | None = None,
             session_id: str | None = None) -> Iterator[dict]:
        """One copilot turn as event dicts (session, model_call, tool_call, ..., answer, usage, done)."""
        message = (message or "").strip()
        if not message:
            yield {"type": "error", "category": "validation", "message": "empty message"}
            return
        session = self.store.get(session_id) if session_id else None
        if session is None:
            session = self.store.create(message[:60], model)
        history = history_for_agent(session)
        self.store.append(session.session_id, Message("user", message, _now()))
        yield {"type": "session", "session_id": session.session_id, "title": session.title}
        answer, meta = None, {"tools": [], "model": model}
        with self._lock, tracked_run(self.profile, self.logger, "app", recorder=self._recorder(),
                                     use_case_id=self.settings.use_case):
            for event in self.copilot().run(message, history=history, model_key=model):
                data = event.as_dict()
                if event.type == "tool_result":
                    meta["tools"].append({k: data.get(k) for k in ("name", "status", "summary", "latency_ms")})
                elif event.type == "answer":
                    answer = data.get("text")
                    meta["citations"] = data.get("citations", [])
                elif event.type == "usage":
                    meta["usage"] = {k: v for k, v in data.items() if k != "type"}
                elif event.type == "guardrail":
                    meta.setdefault("guardrails", []).append(data.get("note"))
                yield data
        if answer:
            self.store.append(session.session_id, Message("assistant", answer, _now(), meta))

    # ------------------------------------------------------------ playground
    def playground(self, models: list[str], prompt: str, system: str | None = None) -> list[dict]:
        from ..agents.claims_copilot.agent import price_function
        price = price_function(self.profile)
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        results = []
        with self._lock, tracked_run(self.profile, self.logger, "playground", recorder=self._recorder(),
                                     use_case_id=self.settings.use_case):
            for key in models:
                started = time.perf_counter()
                try:
                    completion = self._model(key).complete_messages(messages)
                    results.append({
                        "model": key, "status": "ok", "text": completion.text,
                        "latency_ms": round((time.perf_counter() - started) * 1000),
                        "input_tokens": completion.prompt_tokens, "output_tokens": completion.completion_tokens,
                        "served_model": completion.served_model,
                        "cost_usd": price(key, completion.prompt_tokens or 0, completion.completion_tokens or 0)})
                except LighthouseError as exc:
                    results.append({"model": key, "status": "error",
                                    "error": getattr(exc, "category", type(exc).__name__),
                                    "latency_ms": round((time.perf_counter() - started) * 1000)})
        return results

    def _recorder(self) -> UsageRecorder:
        return BackgroundRecorder(self.profile, self.logger)

    def _model(self, key: str):
        if self._model_factory:
            return self._model_factory(key)
        from ..gateway import build_chat_model
        return build_chat_model(self.profile, key, self.logger)

    # ------------------------------------------------------------ models
    def models(self) -> list[dict]:
        from ..agents.claims_copilot.agent import DEFAULT_MODEL, price_function
        from ..platform.model_catalog import load_model_catalog
        price = price_function(self.profile)
        listed = []
        for entry in load_model_catalog():
            if not (set(entry.roles) & APP_ROLES) or entry.status not in LISTED_STATUSES:
                continue
            if entry.key not in self.profile.models:
                continue
            listed.append({"key": entry.key, "model_service": entry.model_service,
                           "family": entry.family, "status": entry.status,
                           "input_usd_per_1m": price(entry.key, 1_000_000, 0),
                           "output_usd_per_1m": price(entry.key, 0, 1_000_000),
                           "default": entry.key == DEFAULT_MODEL})
        return listed

    # ------------------------------------------------------------ usage and evals (Unity Catalog)
    def _query(self, sql: str, ttl: float = 60.0) -> list[dict]:
        cached = self._cache.get(sql)
        if cached and cached.expires > time.monotonic():
            return cached.rows
        rows = self._query_fn(sql) if self._query_fn else self._warehouse(sql)
        self._cache[sql] = CachedQuery(time.monotonic() + ttl, rows)
        return rows

    def _warehouse(self, sql: str) -> list[dict]:
        from sqlalchemy import text

        from ..platform.runtime import platform_names, warehouse_engine
        if self._engine is None:
            self._engine = warehouse_engine(self.profile, platform_names(self.profile), self.logger)
        with self._engine.connect() as connection:
            return [dict(r._mapping) for r in connection.execute(text(sql))]

    def _obs(self) -> str:
        from ..platform.runtime import platform_names
        names = platform_names(self.profile)
        return f"{names.catalog}.{names.observability}"

    def usage_summary(self, hours: int = 24, use_case: str | None = None) -> dict:
        hours = max(1, min(int(hours), 24 * 90))
        where = f"event_time >= current_timestamp() - INTERVAL {hours} HOURS"
        if use_case:
            where += f" AND use_case_id = '{_safe(use_case)}'"
        by_model = self._query(f"""
            SELECT model_key, count(*) AS calls, sum(is_error) AS errors,
                   percentile_approx(latency_ms, 0.5) AS p50_ms, percentile_approx(latency_ms, 0.95) AS p95_ms,
                   sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
                   round(sum(total_cost_usd), 6) AS cost_usd
            FROM {self._obs()}.v_model_call_cost WHERE {where}
            GROUP BY model_key ORDER BY calls DESC""")
        by_activity = self._query(f"""
            SELECT activity, purpose, count(*) AS calls, round(sum(total_cost_usd), 6) AS cost_usd
            FROM {self._obs()}.v_model_call_cost WHERE {where}
            GROUP BY activity, purpose ORDER BY calls DESC""")
        timeline = self._query(f"""
            SELECT date_trunc('HOUR', event_time) AS hour, model_key, count(*) AS calls,
                   round(sum(total_cost_usd), 6) AS cost_usd
            FROM {self._obs()}.v_model_call_cost WHERE {where}
            GROUP BY 1, 2 ORDER BY 1""")
        return {"hours": hours, "by_model": by_model, "by_activity": by_activity, "timeline": timeline,
                "dashboard_url": self.settings.dashboard_url}

    def usage_recent(self, limit: int = 50) -> list[dict]:
        limit = max(1, min(int(limit), 500))
        return self._query(f"""
            SELECT event_time, activity, purpose, model_key, status, latency_ms, input_tokens,
                   output_tokens, round(total_cost_usd, 6) AS cost_usd, run_id, case_id
            FROM {self._obs()}.v_model_call_cost ORDER BY event_time DESC LIMIT {limit}""", ttl=15)

    def eval_runs(self, limit: int = 20) -> list[dict]:
        rows = self._query(f"""
            SELECT run_id, experiment_id, prompt_name, prompt_version, model_key, judge_model_key,
                   dataset_name, cases, combined_score, verdict, started_at, finished_at
            FROM {self._obs()}.eval_run ORDER BY finished_at DESC LIMIT {max(1, min(int(limit), 200))}""")
        return [{**r, "mlflow_url": self._mlflow_url(r)} for r in rows]

    def eval_run(self, run_id: str) -> dict:
        run_id = _safe(run_id)
        runs = self._query(f"SELECT * FROM {self._obs()}.eval_run WHERE run_id = '{run_id}'")
        cases = self._query(f"""
            SELECT case_id, criterion, kind, value, score, passed, trace_id
            FROM {self._obs()}.eval_case_result WHERE run_id = '{run_id}'
            ORDER BY case_id, kind, criterion""")
        if not runs:
            return {}
        return {"run": {**runs[0], "mlflow_url": self._mlflow_url(runs[0])}, "cases": cases}

    def _mlflow_url(self, row: dict) -> str | None:
        if not (self.host and row.get("experiment_id")):
            return None
        return f"{self.host}/ml/experiments/{row['experiment_id']}/runs/{row['run_id']}"

    # ------------------------------------------------------------ warm-up (before a demo)
    def warmup(self) -> dict:
        """Load the copilot and touch every backend once (cold calls take up to ~8 s; Genie and
        the SQL warehouse can take longer on first use). Calls are tracked (activity app)."""
        steps = {}

        def step(name, fn):
            started = time.perf_counter()
            try:
                detail = fn()
                steps[name] = {"status": "ok", "ms": round((time.perf_counter() - started) * 1000), "detail": detail}
            except Exception as exc:              # report, never fail the warm-up as a whole
                steps[name] = {"status": "error", "ms": round((time.perf_counter() - started) * 1000),
                               "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}

        with self._lock, tracked_run(self.profile, self.logger, "app", use_case_id=self.settings.use_case):
            step("copilot", lambda: sorted(getattr(self.copilot(), "tools", {})))
            tools = getattr(self._copilot, "tools", {}) if self._copilot is not None else {}
            if "get_claim_case" in tools:
                step("claim_case", lambda: tools["get_claim_case"].run({"case_id": "CSET-STD-001"}).summary)
            if "search_guidance" in tools:
                step("guidance", lambda: tools["search_guidance"].run({"query": "employer statement last day worked"}).summary)
            if "ask_portfolio" in tools:
                step("genie", lambda: tools["ask_portfolio"].run({"question": "How many claims are there?"}).summary)
        step("warehouse", lambda: f"{len(self.usage_recent(5))} recent calls")
        return {"steps": steps, "ready": all(s["status"] == "ok" for s in steps.values())}

    # ------------------------------------------------------------ health
    def health(self) -> dict:
        checks = {"profile": self.profile.name, "session_store": self.store.describe(),
                  "use_case": self.settings.use_case}
        try:
            from ..gateway import build_auth
            build_auth(self.profile.gateway.auth).headers()
            checks["workspace_auth"] = "ok"
        except Exception as exc:
            checks["workspace_auth"] = f"error: {type(exc).__name__}"
        checks["copilot"] = "ready" if self._copilot is not None else "not loaded (loads on first chat)"
        return checks


def _safe(value: str) -> str:
    """Identifier-ish values only in SQL literals (use case, run ID)."""
    cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch in "_-.")
    if cleaned != str(value):
        raise ValueError("invalid identifier")
    return cleaned

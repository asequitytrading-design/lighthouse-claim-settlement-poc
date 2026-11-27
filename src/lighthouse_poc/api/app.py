"""lighthouse-api: the demo's HTTP layer (FastAPI) over src/lighthouse_poc (docs/demo-thin-ui.md).

Routes (all JSON under /api; chat streams Server-Sent Events):
  GET  /api/health                      workspace auth, session store, copilot state
  POST /api/warmup                      load the copilot and touch every backend (before a demo)
  GET  /api/models                      approved/candidate app models from the governed catalog
  POST /api/chat                        {message, model?, session_id?} -> SSE AgentEvents
  GET  /api/sessions | /api/sessions/{id}   conversations and history; DELETE to remove
  POST /api/playground                  {models[], prompt, system?} -> one result per model
  GET  /api/usage/summary?hours=&use_case=  per model, per activity, hourly timeline (UC views)
  GET  /api/usage/recent?limit=         latest calls (no prompt or response text)
  GET  /api/evals/runs | /api/evals/runs/{run_id}   evaluation runs and per-case scores (UC)
The built UI (apps/lighthouse_demo/web/dist) is served at / when present.

Run locally:  uvicorn lighthouse_poc.api.app:create_app --factory --port 8000
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .services import AppServices
from .settings import AppSettings


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    model: str | None = None
    session_id: str | None = None


class PlaygroundRequest(BaseModel):
    models: list[str] = Field(min_length=1, max_length=4)
    prompt: str = Field(min_length=1, max_length=8000)
    system: str | None = Field(default=None, max_length=8000)


def _sse(events) -> Any:
    for event in events:
        yield f"event: {event.get('type', 'message')}\ndata: {json.dumps(event, default=str)}\n\n"


def default_services(settings: AppSettings) -> AppServices:
    from ..config import load_profile
    from ..logging_config import configure_logging
    from ..notebook_support import model_service_env
    from ..sessions.store import SqliteSessionStore
    logger = configure_logging()
    for key, value in model_service_env().items():      # UNITY_GATEWAY_*_MODEL from the catalog
        os.environ.setdefault(key, value)
    profile = load_profile(settings.profile)
    if settings.session_store == "postgres":
        from ..sessions.postgres import PostgresSessionStore
        store = PostgresSessionStore(settings.pg_dsn)
    else:
        store = SqliteSessionStore(settings.sqlite_path)
    return AppServices(profile, settings, store, logger)


def create_app(services: AppServices | None = None, settings: AppSettings | None = None) -> FastAPI:
    settings = settings or (services.settings if services else AppSettings.from_env())
    services = services or default_services(settings)
    app = FastAPI(title="Lighthouse demo API", version="1.0")
    log = services.logger if services else logging.getLogger(__name__)

    @app.get("/api/health")
    def health():
        return services.health()

    @app.post("/api/warmup")
    def warmup():
        return services.warmup()

    @app.get("/api/models")
    def models():
        return {"models": services.models()}

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        return StreamingResponse(_sse(services.chat(request.message, request.model, request.session_id)),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/sessions")
    def sessions():
        return {"sessions": [{"session_id": s.session_id, "title": s.title, "updated_at": s.updated_at,
                              "model": s.model} for s in services.store.list()]}

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str):
        found = services.store.get(session_id)
        if found is None:
            raise HTTPException(404, "session not found")
        return {"session_id": found.session_id, "title": found.title, "model": found.model,
                "messages": [m.__dict__ for m in found.messages]}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str):
        services.store.delete(session_id)
        return {"deleted": session_id}

    @app.post("/api/playground")
    def playground(request: PlaygroundRequest):
        unknown = [m for m in request.models if m not in services.profile.models]
        if unknown:
            raise HTTPException(400, f"unknown models: {unknown}")
        return {"results": services.playground(request.models, request.prompt, request.system)}

    @app.get("/api/usage/summary")
    def usage_summary(hours: int = 24, use_case: str | None = None):
        try:
            return services.usage_summary(hours, use_case)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/usage/recent")
    def usage_recent(limit: int = 50):
        return {"calls": services.usage_recent(limit)}

    @app.get("/api/evals/runs")
    def eval_runs(limit: int = 20):
        return {"runs": services.eval_runs(limit)}

    @app.get("/api/evals/runs/{run_id}")
    def eval_run(run_id: str):
        try:
            found = services.eval_run(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not found:
            raise HTTPException(404, "run not found")
        return found

    dist = settings.web_dist
    if dist.is_dir() and (dist / "index.html").exists():
        if (dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = dist / path
            return FileResponse(candidate if path and candidate.is_file() else dist / "index.html")
    else:
        log.info("web_ui_not_built", extra={"web_dist": str(dist)})
    return app

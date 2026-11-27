"""Build the session store and semantic cache for a mode, falling back to SQLite when
Postgres/Lakebase is unreachable (the app keeps serving; health reports the fallback)."""
from __future__ import annotations

import logging
from typing import Any

from .cache import MemorySemanticCache
from .store import SqliteSessionStore


def make_store(mode: str, sqlite_path: str, logger: logging.Logger) -> tuple[Any, Any, str]:
    """(store, cache, effective mode description)."""
    if mode in ("postgres", "lakebase"):
        try:
            from .postgres import Connector, PgSemanticCache, PostgresSessionStore
            connector = Connector.from_env(mode)
            return PostgresSessionStore(connector), PgSemanticCache(connector), mode
        except Exception as exc:
            logger.error("session_store_fallback", exc_info=True, extra={
                "store": mode, "error_type": type(exc).__name__, "error_category": "session_store"})
            return (SqliteSessionStore(sqlite_path), MemorySemanticCache(),
                    f"sqlite (fallback: {mode} unavailable, {type(exc).__name__})")
    return SqliteSessionStore(sqlite_path), MemorySemanticCache(), "sqlite"

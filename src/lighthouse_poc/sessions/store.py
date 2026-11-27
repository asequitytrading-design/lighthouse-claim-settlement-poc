"""Chat sessions for the demo app: conversations, messages and per-message metadata.

Backends (selected by LIGHTHOUSE_SESSION_STORE, see api/settings.py):
  sqlite    local development and testing on Databricks Apps (the app disk is ephemeral there:
            history is lost on restart, which is fine for testing)
  postgres  local Postgres + pgvector, or Lakebase for the demo (phase 5, same SQL)

Stored: message text the user typed and the assistant's final answer, plus metadata (tool names,
statuses, citations, usage). Synthetic data only; no secrets.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Message:
    role: str                    # user | assistant
    content: str
    created_at: str
    meta: dict = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    model: str | None = None
    messages: list[Message] = field(default_factory=list)


class SessionStore(Protocol):
    def create(self, title: str, model: str | None = None) -> Session: ...
    def get(self, session_id: str) -> Session | None: ...
    def list(self, limit: int = 50) -> list[Session]: ...
    def append(self, session_id: str, message: Message) -> None: ...
    def delete(self, session_id: str) -> None: ...
    def describe(self) -> str: ...


class SqliteSessionStore:
    def __init__(self, path: str | Path):
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS chat_session (
                    session_id TEXT PRIMARY KEY, title TEXT NOT NULL, model TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chat_message (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chat_message_session ON chat_message(session_id, id);""")

    def describe(self) -> str:
        return f"sqlite:{self._path}"

    def create(self, title: str, model: str | None = None) -> Session:
        session = Session(uuid.uuid4().hex, title[:80] or "New chat", _now(), _now(), model)
        with self._lock:
            self._conn.execute("INSERT INTO chat_session VALUES (?, ?, ?, ?, ?)",
                               (session.session_id, session.title, model, session.created_at,
                                session.updated_at))
            self._conn.commit()
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            row = self._conn.execute("SELECT session_id, title, model, created_at, updated_at "
                                     "FROM chat_session WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            messages = self._conn.execute("SELECT role, content, created_at, meta FROM chat_message "
                                          "WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
        return Session(row[0], row[1], row[3], row[4], row[2],
                       [Message(m[0], m[1], m[2], json.loads(m[3] or "{}")) for m in messages])

    def list(self, limit: int = 50) -> list[Session]:
        with self._lock:
            rows = self._conn.execute("SELECT session_id, title, model, created_at, updated_at "
                                      "FROM chat_session ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [Session(r[0], r[1], r[3], r[4], r[2]) for r in rows]

    def append(self, session_id: str, message: Message) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO chat_message (session_id, role, content, meta, created_at) "
                               "VALUES (?, ?, ?, ?, ?)",
                               (session_id, message.role, message.content,
                                json.dumps(message.meta, default=str), message.created_at))
            self._conn.execute("UPDATE chat_session SET updated_at = ? WHERE session_id = ?",
                               (message.created_at, session_id))
            self._conn.commit()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chat_message WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM chat_session WHERE session_id = ?", (session_id,))
            self._conn.commit()


def history_for_agent(session: Session, max_messages: int = 12) -> list[dict]:
    """Recent user/assistant turns as chat messages (text only) for the agent's context."""
    return [{"role": m.role, "content": m.content} for m in session.messages[-max_messages:]
            if m.role in ("user", "assistant") and m.content]

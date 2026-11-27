"""Postgres backend for chat sessions and the semantic cache: local Postgres or Lakebase.

Same SQL everywhere (pgvector for the cache). Connections:
  local     LIGHTHOUSE_PG_DSN, e.g. postgresql://lighthouse:...@localhost:5434/lighthouse_demo
  Lakebase  LIGHTHOUSE_SESSION_STORE=lakebase: host/database/user from the app's database resource
            (PGHOST, PGDATABASE, PGUSER, PGPORT) or LIGHTHOUSE_LAKEBASE_INSTANCE; the password is a
            short-lived OAuth token from Databricks (refreshed before it expires), sslmode=require.
One connection per operation (the demo is low-traffic; tokens rotate without a pool to drain).
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from ..exceptions import ConfigurationError
from .store import Message, Session, _now

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_session (
    session_id TEXT PRIMARY KEY, title TEXT NOT NULL, model TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chat_message (
    id BIGSERIAL PRIMARY KEY, session_id TEXT NOT NULL REFERENCES chat_session ON DELETE CASCADE,
    role TEXT NOT NULL, content TEXT NOT NULL, meta JSONB, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chat_message_session ON chat_message(session_id, id);
"""

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_cache (
    id TEXT PRIMARY KEY, namespace TEXT NOT NULL, embedding vector({dim}) NOT NULL,
    payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    hits INTEGER NOT NULL DEFAULT 0, last_hit_at TIMESTAMPTZ);
CREATE INDEX IF NOT EXISTS semantic_cache_ns ON semantic_cache(namespace);
"""


class Connector:
    """psycopg connections from a DSN, or to Lakebase with a refreshed OAuth token."""

    def __init__(self, dsn: str | None = None, lakebase: dict | None = None,
                 token_provider: Callable[[], str] | None = None, schema: str = "lighthouse_app"):
        """schema: the app's own Postgres schema (an app identity may not create in public);
        the pgvector type is found through search_path (extension installed in public)."""
        if not dsn and not lakebase:
            raise ConfigurationError("Postgres store needs LIGHTHOUSE_PG_DSN or a Lakebase database")
        if not schema.replace("_", "").isalnum():
            raise ConfigurationError(f"Invalid Postgres schema {schema!r}")
        self._dsn, self._lakebase, self._token_provider = dsn, lakebase, token_provider
        self.schema = schema
        self._schema_ready = False
        self._token: tuple[str, float] | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, mode: str, env: dict | None = None) -> "Connector":
        env = os.environ if env is None else env
        if mode == "postgres":
            return cls(dsn=env.get("LIGHTHOUSE_PG_DSN"))
        host = env.get("PGHOST")
        if not host:
            raise ConfigurationError("Lakebase: bind a database resource to the app (PGHOST missing)")
        return cls(lakebase={"host": host, "port": int(env.get("PGPORT", "5432")),
                             "dbname": env.get("PGDATABASE", "databricks_postgres"),
                             "user": env.get("PGUSER") or env.get("DATABRICKS_CLIENT_ID"),
                             "instance": env.get("LIGHTHOUSE_LAKEBASE_INSTANCE")},
                   token_provider=None)

    def _password(self) -> str:
        with self._lock:
            if self._token and self._token[1] > time.monotonic():
                return self._token[0]
            token = (self._token_provider or _databricks_token(self._lakebase.get("instance")))()
            self._token = (token, time.monotonic() + 40 * 60)        # tokens live ~1 h
            return token

    @contextmanager
    def connect(self) -> Iterator[Any]:
        import psycopg
        if self._dsn:
            connection = psycopg.connect(self._dsn, autocommit=True)
        else:
            cfg = {k: v for k, v in self._lakebase.items() if k != "instance"}
            connection = psycopg.connect(**cfg, password=self._password(), sslmode="require",
                                         autocommit=True)
        try:
            if not self._schema_ready:
                connection.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                self._schema_ready = True
            connection.execute(f"SET search_path TO {self.schema}, public")
            yield connection
        finally:
            connection.close()

    def describe(self) -> str:
        if self._dsn:
            return "postgres:" + self._dsn.split("@")[-1]          # no credentials
        return f"lakebase:{self._lakebase['host']}/{self._lakebase['dbname']}"


def _databricks_token(instance: str | None) -> Callable[[], str]:
    def token() -> str:
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient()
        if instance:
            credential = client.database.generate_database_credential(
                request_id=str(uuid.uuid4()), instance_names=[instance])
            return credential.token
        headers = client.config.authenticate()                     # app identity OAuth token
        return headers["Authorization"].split(" ", 1)[1]
    return token


class PostgresSessionStore:
    def __init__(self, connector: Connector):
        self._connector = connector
        with connector.connect() as connection:
            connection.execute(SCHEMA)

    def describe(self) -> str:
        return self._connector.describe()

    def create(self, title: str, model: str | None = None) -> Session:
        session = Session(uuid.uuid4().hex, title[:80] or "New chat", _now(), _now(), model)
        with self._connector.connect() as c:
            c.execute("INSERT INTO chat_session VALUES (%s, %s, %s, %s, %s)",
                      (session.session_id, session.title, model, session.created_at, session.updated_at))
        return session

    def get(self, session_id: str) -> Session | None:
        with self._connector.connect() as c:
            row = c.execute("SELECT session_id, title, model, created_at, updated_at FROM chat_session "
                            "WHERE session_id = %s", (session_id,)).fetchone()
            if row is None:
                return None
            messages = c.execute("SELECT role, content, created_at, meta FROM chat_message "
                                 "WHERE session_id = %s ORDER BY id", (session_id,)).fetchall()
        return Session(row[0], row[1], row[3], row[4], row[2],
                       [Message(m[0], m[1], m[2], m[3] or {}) for m in messages])

    def list(self, limit: int = 50) -> list[Session]:
        with self._connector.connect() as c:
            rows = c.execute("SELECT session_id, title, model, created_at, updated_at FROM chat_session "
                             "ORDER BY updated_at DESC LIMIT %s", (limit,)).fetchall()
        return [Session(r[0], r[1], r[3], r[4], r[2]) for r in rows]

    def append(self, session_id: str, message: Message) -> None:
        with self._connector.connect() as c:
            c.execute("INSERT INTO chat_message (session_id, role, content, meta, created_at) "
                      "VALUES (%s, %s, %s, %s, %s)",
                      (session_id, message.role, message.content, json.dumps(message.meta, default=str),
                       message.created_at))
            c.execute("UPDATE chat_session SET updated_at = %s WHERE session_id = %s",
                      (message.created_at, session_id))

    def delete(self, session_id: str) -> None:
        with self._connector.connect() as c:
            c.execute("DELETE FROM chat_session WHERE session_id = %s", (session_id,))


class PgSemanticCache:
    """pgvector cache: nearest entry in a namespace with cosine similarity >= threshold."""

    def __init__(self, connector: Connector, dim: int = 1024, ttl_hours: int = 24):
        self._connector, self._ttl = connector, ttl_hours
        with connector.connect() as connection:
            has_vector = connection.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
            if not has_vector:   # needs owner rights: on Lakebase the instance owner creates it once
                connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(CACHE_SCHEMA.format(dim=int(dim)))

    def describe(self) -> str:
        return "pgvector:" + self._connector.describe()

    def lookup(self, namespace: str, vector: list[float], threshold: float) -> dict | None:
        literal = "[" + ",".join(f"{v:.7f}" for v in vector) + "]"
        with self._connector.connect() as c:
            row = c.execute(
                "SELECT id, payload, 1 - (embedding <=> %s::vector) AS similarity FROM semantic_cache "
                "WHERE namespace = %s AND created_at > now() - make_interval(hours => %s) "
                "ORDER BY embedding <=> %s::vector LIMIT 1",
                (literal, namespace, self._ttl, literal)).fetchone()
            if row is None or row[2] < threshold:
                return None
            c.execute("UPDATE semantic_cache SET hits = hits + 1, last_hit_at = now() WHERE id = %s", (row[0],))
        return {"payload": row[1], "similarity": float(row[2])}

    def store(self, namespace: str, vector: list[float], payload: dict) -> None:
        literal = "[" + ",".join(f"{v:.7f}" for v in vector) + "]"
        with self._connector.connect() as c:
            c.execute("INSERT INTO semantic_cache (id, namespace, embedding, payload) VALUES (%s, %s, %s::vector, %s)",
                      (uuid.uuid4().hex, namespace, literal, json.dumps(payload, default=str)))

    def stats(self) -> dict:
        with self._connector.connect() as c:
            row = c.execute("SELECT count(*), coalesce(sum(hits), 0) FROM semantic_cache").fetchone()
        return {"entries": int(row[0]), "hits": int(row[1]), "backend": self.describe()}

    def clear(self, namespace: str | None = None) -> int:
        with self._connector.connect() as c:
            cursor = c.execute("DELETE FROM semantic_cache" + (" WHERE namespace = %s" if namespace else ""),
                               (namespace,) if namespace else None)
            return cursor.rowcount

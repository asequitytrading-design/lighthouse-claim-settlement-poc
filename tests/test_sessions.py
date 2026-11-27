"""Session stores and the semantic cache (Postgres tests need LIGHTHOUSE_TEST_PG_DSN or the local
demo-pg container; they are skipped otherwise, e.g. in CI)."""
import json
import os
import uuid

import pytest

from lighthouse_poc.sessions.cache import MemorySemanticCache, cosine
from lighthouse_poc.sessions.store import Message, SqliteSessionStore, _now

DSN = os.environ.get("LIGHTHOUSE_TEST_PG_DSN",
                     "postgresql://lighthouse:lighthouse-local-only@localhost:5434/lighthouse_demo")


def _pg_available() -> bool:
    try:
        import psycopg
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except Exception:
        return False


def _check_store(store):
    session = store.create("Prepare CSET-LTD-002", "claims-alt-qwen-next")
    store.append(session.session_id, Message("user", "hello", _now()))
    store.append(session.session_id, Message("assistant", "hi", _now(), {"tools": [{"name": "x"}]}))
    loaded = store.get(session.session_id)
    assert [m.role for m in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[1].meta == {"tools": [{"name": "x"}]}
    assert any(s.session_id == session.session_id for s in store.list())
    store.delete(session.session_id)
    assert store.get(session.session_id) is None


def test_sqlite_store_round_trip():
    _check_store(SqliteSessionStore(":memory:"))


def test_memory_cache_hits_only_similar_questions_in_the_namespace():
    cache = MemorySemanticCache()
    cache.store("guidance:a", [1.0, 0.0, 0.0], {"status": "ok"})
    assert cache.lookup("guidance:a", [0.99, 0.05, 0.0], 0.95)["payload"] == {"status": "ok"}
    assert cache.lookup("guidance:a", [0.5, 0.8, 0.0], 0.95) is None             # different question
    assert cache.lookup("guidance:b", [1.0, 0.0, 0.0], 0.95) is None             # other index/model
    assert cache.stats()["hits"] == 1 and cache.clear() == 1
    assert round(cosine([1, 0], [0, 1]), 3) == 0.0


def test_guidance_search_serves_repeated_questions_from_the_cache():
    from tests.test_claims_copilot_tools import _Embedder, _vs
    from lighthouse_poc.agents.tools.databricks_tools import GuidanceSearchTool
    cache = MemorySemanticCache()
    tool = GuidanceSearchTool(_vs([0.7]), _Embedder(), cache=cache, cache_namespace="guidance:test")
    first = tool.run({"query": "police report missing"})
    second = tool.run({"query": "police report missing"})
    assert first.status == second.status == "ok" and "cache" not in first.data
    assert second.data["cache"]["hit"] and second.summary.endswith("cache hit")
    assert second.content == first.content and cache.stats()["entries"] == 1


@pytest.mark.skipif(not _pg_available(), reason="no local Postgres (docker compose up -d demo-pg)")
def test_postgres_store_and_pgvector_cache():
    from lighthouse_poc.sessions.postgres import Connector, PgSemanticCache, PostgresSessionStore
    connector = Connector(dsn=DSN)
    _check_store(PostgresSessionStore(connector))
    cache = PgSemanticCache(connector)                          # 1024-d, like the app
    namespace = f"test:{uuid.uuid4().hex}"

    def vec(*head):
        return list(head) + [0.0] * (1024 - len(head))
    cache.store(namespace, vec(1.0, 0.0), {"status": "ok", "content": {"q": 1}})
    hit = cache.lookup(namespace, vec(0.99, 0.05), 0.95)
    assert hit["payload"]["content"] == {"q": 1} and hit["similarity"] > 0.95
    assert cache.lookup(namespace, vec(0.0, 1.0), 0.95) is None
    assert cache.clear(namespace) == 1
    assert "localhost:5434" in connector.describe() and "lighthouse-local-only" not in connector.describe()

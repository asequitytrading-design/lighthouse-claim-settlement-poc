"""Semantic cache for guidance answers (never case-specific answers: another claimant's facts
must never be served for a "similar" question).

Key: the question's embedding inside a namespace (index + embedding model); a hit is the nearest
entry with cosine similarity >= threshold (0.95 by default) younger than the TTL. A hit skips the
vector search and the grounding model call. Backends: PgSemanticCache (local Postgres / Lakebase,
pgvector) and MemorySemanticCache (SQLite mode, per process).
"""
from __future__ import annotations

import math
import threading
import time
from typing import Protocol


class SemanticCache(Protocol):
    def lookup(self, namespace: str, vector: list[float], threshold: float) -> dict | None: ...
    def store(self, namespace: str, vector: list[float], payload: dict) -> None: ...
    def stats(self) -> dict: ...
    def clear(self, namespace: str | None = None) -> int: ...
    def describe(self) -> str: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class MemorySemanticCache:
    def __init__(self, ttl_seconds: float = 24 * 3600, max_entries: int = 500):
        self._entries: list[dict] = []
        self._ttl, self._max = ttl_seconds, max_entries
        self._hits = 0
        self._lock = threading.Lock()

    def describe(self) -> str:
        return "memory"

    def lookup(self, namespace: str, vector: list[float], threshold: float) -> dict | None:
        now = time.monotonic()
        with self._lock:
            live = [e for e in self._entries if e["namespace"] == namespace and now - e["at"] < self._ttl]
            best = max(live, key=lambda e: cosine(e["vector"], vector), default=None)
            if best is None:
                return None
            similarity = cosine(best["vector"], vector)
            if similarity < threshold:
                return None
            best["hits"] += 1
            self._hits += 1
            return {"payload": best["payload"], "similarity": similarity}

    def store(self, namespace: str, vector: list[float], payload: dict) -> None:
        with self._lock:
            self._entries.append({"namespace": namespace, "vector": vector, "payload": payload,
                                  "at": time.monotonic(), "hits": 0})
            del self._entries[:-self._max]

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._entries), "hits": self._hits, "backend": "memory"}

    def clear(self, namespace: str | None = None) -> int:
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if namespace and e["namespace"] != namespace]
            return before - len(self._entries)

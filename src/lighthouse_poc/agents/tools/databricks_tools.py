"""Agent tools backed by Databricks managed MCP servers (synthetic claims data, guidance, Genie).

  get_claim_case    UC function <prefix>ai_claims.get_claim_case via the UC-functions MCP server
  search_guidance   the knowledge index via the Vector Search MCP server; the query is embedded
                    through Unity Gateway (tracked) because the index uses self-managed
                    embeddings; passages below the similarity floor are not returned, so the
                    model gets "no relevant guidance" instead of off-topic text (grounding,
                    see platform/grounding.py)
  ask_portfolio     the claims Genie space via the Genie MCP server (polls until complete)

The MCP tool names are discovered once (tools/list) and mapped to these stable names.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..core.tools import ToolResult, ToolSpec, timed
from .mcp_client import McpError, McpHttpClient

NO_GUIDANCE = "No relevant guidance in the knowledge base for this question."


def _discover(client: McpHttpClient, suffix: str) -> str:
    for tool in client.list_tools():
        if tool.name.endswith(suffix) or tool.name.startswith(suffix):
            return tool.name
    raise McpError(f"no MCP tool matching {suffix!r} at {client.url}", "mcp_tool_missing")


class ClaimCaseTool:
    spec = ToolSpec(
        "get_claim_case",
        "Fetch one synthetic claim case file by case ID (e.g. CSET-LTD-002): product line, status, "
        "dates, a short summary and every document with its status (received, missing, "
        "conflicting). Use it before saying anything about a specific claim.",
        {"type": "object", "properties": {"case_id": {"type": "string",
                                                      "description": "Claim case ID, e.g. CSET-LTD-002"}},
         "required": ["case_id"]})

    def __init__(self, client: McpHttpClient):
        self._client = client
        self._tool: str | None = None

    def run(self, arguments: dict) -> ToolResult:
        def call() -> ToolResult:
            case_id = str(arguments.get("case_id", "")).strip()
            if not case_id:
                return ToolResult(self.spec.name, "error", {"error": "case_id is required"}, "no case ID")
            self._tool = self._tool or _discover(self._client, "__get_claim_case")
            text = self._client.call_tool(self._tool, {"case_id": case_id})
            rows = json.loads(text).get("rows") or [[None]]
            case = json.loads(rows[0][0]) if rows[0][0] else {"error": "case not found"}
            if "error" in case:
                return ToolResult(self.spec.name, "empty", case, f"{case_id}: not found")
            docs = case.get("documents") or []
            missing = sum(d.get("status") == "missing" for d in docs)
            return ToolResult(self.spec.name, "ok", case,
                              f"{case['case_id']}: {len(docs)} documents, {missing} missing",
                              data={"case_id": case["case_id"]})
        return timed(self.spec.name, call)


class GuidanceSearchTool:
    spec = ToolSpec(
        "search_guidance",
        "Search the claims examiner guidance (evidence checklists per product, handoff format). "
        "Returns passages with their source document; returns nothing when the guidance does "
        "not cover the question.",
        {"type": "object", "properties": {"query": {"type": "string",
                                                    "description": "What to look up, in plain words"}},
         "required": ["query"]})

    # floor 0.52: measured 2026-11-27, off-topic questions 0.42-0.52, sample questions 0.53-0.72
    def __init__(self, client: McpHttpClient, embedder: Any, floor: float = 0.52, k: int = 3,
                 answerer: Any = None, cache: Any = None, cache_namespace: str = "guidance",
                 cache_threshold: float = 0.95):
        """answerer: platform.grounding.GroundedAnswerer. When set, the passages must actually
        answer the question (model check + verbatim quotes), else the tool returns no guidance:
        retrieval alone lets look-alikes through (pet insurance scores like a claims question)."""
        self._client, self._embedder, self._floor, self._k = client, embedder, floor, k
        self._answerer = answerer
        self._cache, self._namespace, self._threshold = cache, cache_namespace, cache_threshold
        self._tool: str | None = None

    def run(self, arguments: dict) -> ToolResult:
        def call() -> ToolResult:
            query = str(arguments.get("query", "")).strip()
            if not query:
                return ToolResult(self.spec.name, "error", {"error": "query is required"}, "no query")
            vector = self._embedder.embed([query]).vectors[0]
            cached = self._cached(vector)
            if cached is not None:
                return cached
            result = self._search(query, vector)
            if self._cache is not None and result.status in ("ok", "empty"):
                try:
                    self._cache.store(self._namespace, vector, {
                        "status": result.status, "content": result.content,
                        "summary": result.summary, "data": result.data})
                except Exception:          # the cache must never break the answer
                    pass
            return result
        return timed(self.spec.name, call)

    def _cached(self, vector: list[float]) -> ToolResult | None:
        if self._cache is None:
            return None
        try:
            hit = self._cache.lookup(self._namespace, vector, self._threshold)
        except Exception:
            return None
        if hit is None:
            return None
        p = hit["payload"]
        return ToolResult(self.spec.name, p["status"], p["content"], f"{p['summary']} · cache hit",
                          data={**p.get("data", {}), "cache": {"hit": True,
                                                              "similarity": round(hit["similarity"], 3)}})

    def _search(self, query: str, vector: list[float]) -> ToolResult:
        self._tool = self._tool or _discover(self._client, "__document_chunk_index")
        hits = sorted(json.loads(self._client.call_tool(
            self._tool, {"query_vector": vector, "num_results": self._k})),
            key=lambda h: -float(h.get("score", 0)))
        passages = [{"chunk_id": h["chunk_id"], "doc_id": h["doc_id"], "text": h["text"],
                     "score": round(float(h["score"]), 3)}
                    for h in hits if float(h.get("score", 0)) >= self._floor][: self._k]
        # the server may return more than num_results: keep the best k
        if not passages:
            best = max((float(h.get("score", 0)) for h in hits), default=0.0)
            return ToolResult(self.spec.name, "empty", {"result": NO_GUIDANCE},
                              f"no passage above {self._floor} (best {best:.2f})",
                              data={"best_score": round(best, 3)})
        if self._answerer is None:
            return ToolResult(self.spec.name, "ok", {"passages": passages},
                              f"{len(passages)} passages ({', '.join(p['doc_id'] for p in passages)})",
                              data={"citations": passages})
        return self._grounded(query, passages)

    def _grounded(self, query: str, passages: list[dict]) -> ToolResult:
        from ...platform.retrieval import Hit
        # chunk_index 0: each passage is a chunk scored by its own (dense) similarity
        hits = [Hit(i, p["doc_id"], p["chunk_id"], 0, p["score"], p["text"]) for i, p in enumerate(passages, 1)]
        grounded = self._answerer.answer(query, hits, hits)
        if not grounded.grounded:
            return ToolResult(self.spec.name, "empty", {"result": NO_GUIDANCE},
                              f"not answered by the guidance ({grounded.refusal_reason})",
                              data={"grounding_score": 0.0, "refusal_reason": grounded.refusal_reason})
        cited = [p for p in passages if p["chunk_id"] in grounded.cited_chunks]
        return ToolResult(self.spec.name, "ok",
                          {"passages": cited, "verified_quotes": grounded.verified_quotes,
                           "grounded_summary": grounded.answer},
                          f"{len(cited)} cited passages, grounding {grounded.grounding_score:.2f}",
                          data={"citations": cited, "grounding_score": grounded.grounding_score,
                                "verified_quotes": grounded.verified_quotes})


class PortfolioTool:
    spec = ToolSpec(
        "ask_portfolio",
        "Answer aggregate questions across all synthetic claims (counts, lists, statuses, missing "
        "documents by product) by asking the claims Genie space, which writes and runs SQL on "
        "Unity Catalog. Not for a single claim's details (use get_claim_case).",
        {"type": "object", "properties": {"question": {"type": "string",
                                                       "description": "The aggregate question"}},
         "required": ["question"]})

    def __init__(self, client: McpHttpClient, poll_seconds: float = 2.0, max_wait_seconds: float = 60.0,
                 sleep=time.sleep):
        self._client, self._poll, self._max_wait, self._sleep = client, poll_seconds, max_wait_seconds, sleep
        self._query_tool: str | None = None
        self._poll_tool: str | None = None

    def run(self, arguments: dict) -> ToolResult:
        def call() -> ToolResult:
            question = str(arguments.get("question", "")).strip()
            if not question:
                return ToolResult(self.spec.name, "error", {"error": "question is required"}, "no question")
            if not self._query_tool:
                names = [t.name for t in self._client.list_tools()]
                self._query_tool = next(n for n in names if n.startswith("query_space"))
                self._poll_tool = next(n for n in names if n.startswith("poll_response"))
            reply = json.loads(self._client.call_tool(self._query_tool, {"query": question}))
            waited = 0.0
            while reply.get("status") not in ("COMPLETED", "FAILED", "CANCELLED") and waited < self._max_wait:
                self._sleep(self._poll)
                waited += self._poll
                reply = json.loads(self._client.call_tool(self._poll_tool, {
                    "conversation_id": reply["conversationId"], "message_id": reply["messageId"]}))
            if reply.get("status") != "COMPLETED":
                return ToolResult(self.spec.name, "error", {"error": f"Genie status {reply.get('status')}"},
                                  f"Genie {reply.get('status', 'timeout')}", error_category="genie")
            content = reply.get("content") or {}
            text = " ".join(content.get("textAttachments") or [])
            queries = content.get("queryAttachments") or []
            sql = queries[0].get("query") if queries else None
            rows = _rows(queries[0]) if queries else []
            return ToolResult(self.spec.name, "ok", {"answer": text, "rows": rows[:20]},
                              f"{len(rows)} rows" + (" (SQL)" if sql else ""),
                              data={"sql": sql, "rows": rows[:20], "columns": _columns(queries[0]) if queries else [],
                                    "suggested_questions": content.get("suggestedQuestions") or []})
        return timed(self.spec.name, call)


def _columns(attachment: dict) -> list[str]:
    schema = ((attachment.get("statement_response") or {}).get("manifest") or {}).get("schema") or {}
    return [c.get("name") for c in schema.get("columns") or []]


def _rows(attachment: dict) -> list[list]:
    result = (attachment.get("statement_response") or {}).get("result") or {}
    return [[v.get("string_value") for v in row.get("values", [])] for row in result.get("data_array") or []]

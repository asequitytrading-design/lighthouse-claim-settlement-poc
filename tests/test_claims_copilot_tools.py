"""Claims copilot phase 1: synthetic claims data, migration 0005, MCP client and tools (offline)."""
import json
import logging
from types import SimpleNamespace

import pytest

from lighthouse_poc.agents.claims_copilot.synthetic_claims import CHECKLISTS, GOLDEN, generate
from lighthouse_poc.agents.tools.databricks_tools import (NO_GUIDANCE, ClaimCaseTool,
                                                          GuidanceSearchTool, PortfolioTool)
from lighthouse_poc.agents.tools.mcp_client import McpError, McpHttpClient
from lighthouse_poc.platform.genie import GenieSpaces, serialized_space, space_title
from lighthouse_poc.platform.migrations import SchemaMigrator
from lighthouse_poc.platform.naming import PlatformNames

LOG = logging.getLogger("test")
NAMES = PlatformNames("workspace", "dev_alex_")


# ---------------------------------------------------------------- data and migration
def test_synthetic_claims_are_deterministic_and_cover_every_product():
    cases, docs = generate()
    again, _ = generate()
    assert cases == again and len(cases) == 40
    assert {c.product_line for c in cases} == set(CHECKLISTS)
    assert {g[0] for g in GOLDEN} <= {c.case_id for c in cases}
    assert all(c.synthetic for c in cases)
    assert len({(d.case_id, d.document_type) for d in docs}) == len(docs)      # merge keys unique
    assert all(d.received_date is None for d in docs if d.doc_status == "missing")


def test_revision_0005_creates_the_claims_schema_tables_function_and_data():
    sql = SchemaMigrator(NAMES, LOG).offline_sql(start="0004_gateway_usage_copy")
    assert "CREATE SCHEMA IF NOT EXISTS workspace.dev_alex_ai_claims" in sql
    assert "workspace.dev_alex_ai_claims.claim_case" in sql and "claim_document" in sql
    assert "FUNCTION workspace.dev_alex_ai_claims.get_claim_case" in sql
    assert "'CSET-LTD-002'" in sql and "WHEN NOT MATCHED BY SOURCE THEN DELETE" in sql
    assert "HAVING count(c.case_id) > 0" in sql                      # unknown ID -> "case not found"
    assert NAMES.sql_names()["clm"] == "workspace.dev_alex_ai_claims"


# ---------------------------------------------------------------- MCP client
class _Http:
    """Fake httpx client: answers each JSON-RPC call from a handler."""

    def __init__(self, handler, status=200):
        self.handler, self.status, self.calls = handler, status, []

    def post(self, url, json=None, headers=None):
        self.calls.append((url, json, headers))
        body = self.handler(json["method"], json["params"])
        return SimpleNamespace(status_code=self.status, json=lambda: body)


class _Auth:
    def headers(self):
        return {"Authorization": "Bearer t"}


def _client(handler, status=200):
    http = _Http(handler, status)
    return McpHttpClient("https://h/", "/api/2.0/mcp/functions/c/s", _Auth(), LOG, http=http), http


def test_mcp_client_lists_and_calls_tools_with_auth():
    def handler(method, params):
        if method == "tools/list":
            return {"result": {"tools": [{"name": "c__s__get_claim_case", "inputSchema": {}}]}}
        return {"result": {"content": [{"type": "text", "text": '{"rows": [["x"]]}'}]}}
    client, http = _client(handler)
    assert [t.name for t in client.list_tools()] == ["c__s__get_claim_case"]
    assert client.call_tool("c__s__get_claim_case", {"case_id": "A"}) == '{"rows": [["x"]]}'
    assert http.calls[0][0] == "https://h/api/2.0/mcp/functions/c/s"
    assert http.calls[1][2]["Authorization"] == "Bearer t"


def test_mcp_errors_are_typed():
    client, _ = _client(lambda m, p: {"error": {"code": -32602, "message": "bad"}})
    with pytest.raises(McpError, match="-32602"):
        client.call_tool("t", {})
    client, _ = _client(lambda m, p: {}, status=403)
    with pytest.raises(McpError) as exc:
        client.list_tools()
    assert exc.value.category == "permission"


# ---------------------------------------------------------------- tools
CASE = {"case_id": "CSET-LTD-002", "documents": [{"document_type": "x", "status": "missing"},
                                                  {"document_type": "y", "status": "received"}]}


def _uc(case_json):
    def handler(method, params):
        if method == "tools/list":
            return {"result": {"tools": [{"name": "workspace__s__get_claim_case"}]}}
        return {"result": {"content": [{"type": "text",
                                        "text": json.dumps({"columns": ["output"], "rows": [[case_json]]})}]}}
    return _client(handler)[0]


def test_claim_case_tool_returns_the_case_or_not_found():
    result = ClaimCaseTool(_uc(json.dumps(CASE))).run({"case_id": "CSET-LTD-002"})
    assert result.status == "ok" and result.summary == "CSET-LTD-002: 2 documents, 1 missing"
    missing = ClaimCaseTool(_uc(json.dumps({"error": "case not found", "case_id": "Z"}))).run({"case_id": "Z"})
    assert missing.status == "empty" and "not found" in missing.summary
    assert ClaimCaseTool(_uc("{}")).run({}).status == "error"


class _Embedder:
    def embed(self, texts):
        return SimpleNamespace(vectors=[[0.1] * 4 for _ in texts])


def _vs(scores):
    hits = [{"chunk_id": f"c{i}", "doc_id": f"d{i}", "text": f"t{i}", "score": s} for i, s in enumerate(scores)]

    def handler(method, params):
        if method == "tools/list":
            return {"result": {"tools": [{"name": "workspace__kn__document_chunk_index"}]}}
        assert "query_text" not in params["arguments"] and params["arguments"]["query_vector"]
        return {"result": {"content": [{"type": "text", "text": json.dumps(hits)}]}}
    return _client(handler)[0]


def test_guidance_search_keeps_the_best_k_above_the_floor():
    result = GuidanceSearchTool(_vs([0.55, 0.70, 0.40, 0.60, 0.58]), _Embedder()).run({"query": "q"})
    assert result.status == "ok" and [p["score"] for p in result.content["passages"]] == [0.7, 0.6, 0.58]
    assert result.data["citations"][0]["doc_id"] == "d1"


def test_guidance_search_refuses_off_topic_questions():
    result = GuidanceSearchTool(_vs([0.49, 0.45]), _Embedder()).run({"query": "best pizza in Rome"})
    assert result.status == "empty" and result.content == {"result": NO_GUIDANCE}
    assert result.data["best_score"] == 0.49


def test_portfolio_tool_polls_genie_until_complete():
    replies = iter([
        {"status": "EXECUTING_QUERY", "conversationId": "c", "messageId": "m"},
        {"status": "COMPLETED", "content": {
            "textAttachments": ["There are **5** claims."],
            "queryAttachments": [{"query": "SELECT 5", "statement_response": {
                "manifest": {"schema": {"columns": [{"name": "claims"}]}},
                "result": {"data_array": [{"values": [{"string_value": "5"}]}]}}}],
            "suggestedQuestions": ["next?"]}}])

    def handler(method, params):
        if method == "tools/list":
            return {"result": {"tools": [{"name": "query_space_x"}, {"name": "poll_response_x"}]}}
        return {"result": {"content": [{"type": "text", "text": json.dumps(next(replies))}]}}
    result = PortfolioTool(_client(handler)[0], sleep=lambda s: None).run({"question": "how many?"})
    assert result.status == "ok" and result.content["answer"] == "There are **5** claims."
    assert result.data["sql"] == "SELECT 5" and result.data["rows"] == [["5"]]
    assert result.data["columns"] == ["claims"]


def test_tool_failures_become_error_results_not_exceptions():
    client, _ = _client(lambda m, p: {}, status=500)
    result = ClaimCaseTool(client).run({"case_id": "A"})
    assert result.status == "error" and result.error_category == "mcp_http"


# ---------------------------------------------------------------- Genie space
def test_genie_space_definition_and_lookup_picks_the_oldest_by_title_prefix():
    space = json.loads(serialized_space(NAMES))
    assert [t["identifier"] for t in space["data_sources"]["tables"]] == [
        "workspace.dev_alex_ai_claims.claim_case", "workspace.dev_alex_ai_claims.claim_document"]
    title = space_title(NAMES)
    spaces = [SimpleNamespace(space_id="new", title=f"{title} 2026-11-27 12:15", create_time="2026-11-27T12:15"),
              SimpleNamespace(space_id="old", title=title, create_time="2026-11-27T12:00"),
              SimpleNamespace(space_id="x", title="Bakehouse", create_time="2026-11-01")]
    client = SimpleNamespace(genie=SimpleNamespace(
        list_spaces=lambda page_token=None: SimpleNamespace(spaces=spaces, next_page_token=None),
        create_space=lambda **kw: pytest.fail("must not create when one exists")))
    assert GenieSpaces(client, LOG).ensure(NAMES, "wh").space_id == "old"
    assert GenieSpaces(client, LOG).ensure(NAMES, "wh", configured_id="pinned").space_id == "pinned"

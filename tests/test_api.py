"""Demo API (FastAPI) with fake copilot, models and warehouse (offline)."""
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lighthouse_poc.agents.core.agent import AgentEvent
from lighthouse_poc.api.app import create_app
from lighthouse_poc.api.services import AppServices
from lighthouse_poc.api.settings import AppSettings
from lighthouse_poc.config import load_profile
from lighthouse_poc.exceptions import ModelInvocationError
from lighthouse_poc.sessions.store import SqliteSessionStore

LOG = logging.getLogger("test")


class _Copilot:
    def __init__(self):
        self.calls = []

    def run(self, message, history=None, model_key=None):
        self.calls.append((message, history, model_key))
        yield AgentEvent("tool_call", {"id": "c0", "name": "get_claim_case", "arguments": {"case_id": "A"}})
        yield AgentEvent("tool_result", {"id": "c0", "name": "get_claim_case", "status": "ok",
                                         "summary": "A: 2 documents", "latency_ms": 5, "data": {}})
        yield AgentEvent("answer", {"text": f"answer to {message}", "citations": [{"doc_id": "kb"}]})
        yield AgentEvent("usage", {"model": model_key, "model_calls": 2, "cost_usd": 0.0001})
        yield AgentEvent("done", {})


class _Model:
    def __init__(self, key):
        self.key = key

    def complete_messages(self, messages, tags=None):
        if self.key == "claims-alt-llama":
            raise ModelInvocationError("busy", "rate_limit")
        return SimpleNamespace(text=f"{self.key}: hi", prompt_tokens=10, completion_tokens=5, served_model="s")


def _client(queries=None):
    profile = load_profile("databricks", env={"DATABRICKS_HOST": "https://h", "UNITY_GATEWAY_SMALL_MODEL": "system.ai.s",
                                              "UNITY_GATEWAY_MEDIUM_MODEL": "system.ai.m",
                                              "UNITY_GATEWAY_JUDGE_MODEL": "system.ai.j",
                                              "LIGHTHOUSE_PLATFORM_PREFIX": "dev_alex_",
                                              "LIGHTHOUSE_RECORD_USAGE": "false"})
    seen = []

    def query(sql):
        seen.append(sql)
        return (queries or {}).get(next((k for k in (queries or {}) if k in sql), ""), [])
    copilot = _Copilot()
    services = AppServices(profile, AppSettings(session_store="sqlite", sqlite_path=":memory:"),
                           SqliteSessionStore(":memory:"), LOG, copilot_factory=lambda: copilot,
                           model_factory=_Model, query=query)
    return TestClient(create_app(services)), copilot, seen


def _events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.fixture(autouse=True)
def _no_usage_recording(monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_RECORD_USAGE", "false")


def test_chat_streams_events_and_keeps_the_conversation():
    client, copilot, _ = _client()
    first = _events(client.post("/api/chat", json={"message": "Prepare A", "model": "claims-small"}))
    assert [e["type"] for e in first] == ["session", "tool_call", "tool_result", "answer", "usage", "done"]
    session_id = first[0]["session_id"]
    second = _events(client.post("/api/chat", json={"message": "and B?", "session_id": session_id}))
    assert second[0]["session_id"] == session_id
    assert copilot.calls[1][1] == [{"role": "user", "content": "Prepare A"},
                                   {"role": "assistant", "content": "answer to Prepare A"}]
    history = client.get(f"/api/sessions/{session_id}").json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant", "user", "assistant"]
    assert history["messages"][1]["meta"]["tools"][0]["name"] == "get_claim_case"
    assert client.get("/api/sessions").json()["sessions"][0]["session_id"] == session_id
    assert client.delete(f"/api/sessions/{session_id}").status_code == 200
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_chat_rejects_empty_messages():
    client, _, _ = _client()
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_models_list_app_models_with_prices():
    client, _, _ = _client()
    models = {m["key"]: m for m in client.get("/api/models").json()["models"]}
    assert {"claims-small", "claims-medium", "claims-alt-qwen-next", "claims-alt-llama"} <= set(models)
    assert "claims-judge" not in models and models["claims-alt-qwen-next"]["default"]
    assert models["claims-small"]["input_usd_per_1m"] == 0.07


def test_playground_compares_models_and_reports_failures():
    client, _, _ = _client()
    results = client.post("/api/playground", json={"models": ["claims-small", "claims-alt-llama"],
                                                   "prompt": "hi"}).json()["results"]
    assert results[0]["status"] == "ok" and results[0]["text"] == "claims-small: hi"
    assert results[0]["cost_usd"] == round((10 * 0.07 + 5 * 0.30) / 1e6, 6)
    assert results[1] == {"model": "claims-alt-llama", "status": "error", "error": "rate_limit",
                          "latency_ms": results[1]["latency_ms"]}
    assert client.post("/api/playground", json={"models": ["nope"], "prompt": "x"}).status_code == 400


def test_usage_and_evals_read_unity_catalog_views():
    client, _, seen = _client({"GROUP BY model_key": [{"model_key": "claims-small", "calls": 3}],
                               "FROM workspace.dev_alex_ai_observability.eval_run ORDER BY":
                                   [{"run_id": "r1", "experiment_id": "7", "verdict": "READY FOR HUMAN REVIEW"}]})
    summary = client.get("/api/usage/summary?hours=6").json()
    assert summary["by_model"][0]["calls"] == 3 and "INTERVAL 6 HOURS" in seen[0]
    assert "dev_alex_ai_observability.v_model_call_cost" in seen[0]
    runs = client.get("/api/evals/runs").json()["runs"]
    assert runs[0]["mlflow_url"] == "https://h/ml/experiments/7/runs/r1"
    assert client.get("/api/usage/summary?use_case=x' OR 1=1 --").status_code == 400   # no SQL injection
    assert client.get("/api/evals/runs/missing").status_code == 404


def test_health_reports_store_and_copilot_state():
    client, _, _ = _client()
    health = client.get("/api/health").json()
    assert health["session_store"] == "sqlite::memory:" and "not loaded" in health["copilot"]


def test_warmup_reports_each_step_without_failing():
    client, _, _ = _client()
    result = client.post("/api/warmup").json()
    assert result["steps"]["copilot"]["status"] == "ok" and "warehouse" in result["steps"]

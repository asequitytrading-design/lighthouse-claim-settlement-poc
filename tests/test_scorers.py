"""Judge binding and deterministic scorer tests (no network, no judge calls)."""
import json
import logging
import os
from pathlib import Path

import pytest

from lighthouse_poc.config import load_profile, scoped_environment
from lighthouse_poc.exceptions import ConfigurationError
from lighthouse_poc.gateway import build_chat_model
from lighthouse_poc.scorers import (ERROR_KEY, build_judge_binding, build_scorers,
                                    expected_fields_match, non_decision_boundary,
                                    output_schema_valid)

LOGGER = logging.getLogger("test")
GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden_eval_dataset.jsonl"


def local_profile():
    return load_profile("local", env={})


def test_kong_judge_binding_uses_route_and_header_key(monkeypatch):
    monkeypatch.setenv("KONG_API_KEY", "kong-secret")
    binding = build_judge_binding(local_profile())
    assert binding.model_uri == "openai:/gemma4-e4b-32k:latest"
    assert binding.environment["OPENAI_API_BASE"] == "http://localhost:8100/ai/claims-judge"
    assert binding.environment["OPENAI_API_KEY"] != "kong-secret"
    assert binding.extra_headers == {"apikey": "kong-secret"}
    assert binding.inference_params == {"temperature": 0, "reasoning_effort": "none"}
    # Lineage and repr must never expose the credential.
    assert "kong-secret" not in json.dumps(binding.describe())
    assert "kong-secret" not in repr(binding)


def test_alternate_judge_uses_its_own_route(monkeypatch):
    monkeypatch.setenv("KONG_API_KEY", "k")
    binding = build_judge_binding(local_profile(), "claims-judge-alt")
    assert binding.model_uri == "openai:/qwen3.5-9b-32k:latest"
    assert binding.environment["OPENAI_API_BASE"].endswith("/ai/claims-medium")


def test_unity_gateway_judge_uses_bearer_token_as_api_key(monkeypatch):
    env = {"DATABRICKS_HOST": "https://x.cloud.databricks.com",
           "UNITY_GATEWAY_JUDGE_MODEL": "system.ai.judge-model"}
    profile = load_profile("databricks", env=env)

    class FakeDatabricksAuth:
        def headers(self):
            return {"Authorization": "Bearer dbx-token"}

    monkeypatch.setattr("lighthouse_poc.scorers.build_auth", lambda auth: FakeDatabricksAuth())
    binding = build_judge_binding(profile)
    assert binding.model_uri == "openai:/system.ai.judge-model"
    assert binding.environment == {
        "OPENAI_API_BASE": "https://x.cloud.databricks.com/ai-gateway/mlflow/v1",
        "OPENAI_API_KEY": "dbx-token"}
    assert binding.extra_headers == {}


def test_databricks_managed_judge_needs_no_gateway():
    binding = build_judge_binding(load_profile("databricks", env={}), "databricks-managed-judge")
    assert binding.model_uri == "databricks"
    assert binding.environment == {}


def test_judge_configuration_errors(monkeypatch):
    monkeypatch.setenv("KONG_API_KEY", "k")
    with pytest.raises(ConfigurationError, match="mock"):
        build_judge_binding(local_profile(), "mock")
    with pytest.raises(ConfigurationError, match="upstream_model"):
        build_judge_binding(local_profile(), "claims-small")  # pinned by Kong, name unknown
    with pytest.raises(ConfigurationError, match="judge"):
        build_chat_model(load_profile("databricks", env={}), "databricks-managed-judge", LOGGER)


def test_build_scorers_modes(monkeypatch):
    monkeypatch.setenv("KONG_API_KEY", "k")
    scorers, judge, manifest = build_scorers(local_profile(), "deterministic", None, LOGGER)
    assert judge is None and len(scorers) == 3
    scorers, judge, manifest = build_scorers(local_profile(), "llm-judges", None, LOGGER)
    names = [s.name for s in scorers]
    assert names[:3] == ["output_schema_valid", "non_decision_boundary", "expected_fields_match"]
    assert {"correctness", "safety", "expectations_guidelines"} <= set(names)
    assert manifest["judge_model"] == "claims-judge"
    assert "k" not in manifest.values()


def test_judge_environment_is_restored(monkeypatch):
    monkeypatch.setenv("KONG_API_KEY", "k")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    binding = build_judge_binding(local_profile())
    with binding.activated():
        assert os.environ["OPENAI_API_BASE"].endswith("/claims-judge")
    assert "OPENAI_API_BASE" not in os.environ


def test_scoped_environment_restores_existing(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", "10")
    with scoped_environment(local_profile().evaluation.environment()):
        assert os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] == "2"
        assert os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] == "1"
        assert os.environ["MLFLOW_GENAI_EVAL_LLM_TIMEOUT"] == "600"
    assert os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] == "10"


def test_golden_records_are_compatible_with_correctness_judge():
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        expectations = json.loads(line)["expectations"]
        assert "expected_response" not in expectations  # Correctness rejects it with expected_facts
        assert expectations["expected_facts"] and expectations["guidelines"]
        assert expectations["expected_fields"]["benefit_determination"] == "Not determined"


def call(scorer_obj, **kwargs):
    return scorer_obj(**kwargs)


def test_deterministic_gates_fail_on_error_marker_and_decisions():
    good = {"claim_id": "C", "product_segment": "dental", "summary": "s",
            "key_facts": [], "evidence_received": [], "evidence_to_confirm": [],
            "inconsistencies": [], "recommended_examiner_actions": [], "risk_flags": [],
            "benefit_determination": "Not determined"}
    expectations = {"expected_fields": {"product_segment": "dental",
                                        "benefit_determination": "Not determined"}}
    assert call(output_schema_valid, outputs=good) is True
    assert call(non_decision_boundary, outputs=good) is True
    assert call(expected_fields_match, outputs=good, expectations=expectations) is True

    errored = {ERROR_KEY: "OutputValidationError", "claim_id": "C"}
    assert call(output_schema_valid, outputs=errored) is False
    assert call(non_decision_boundary, outputs=errored) is False

    decided = dict(good, benefit_determination="Approved")
    assert call(non_decision_boundary, outputs=decided) is False
    assert call(expected_fields_match, outputs=decided, expectations=expectations) is False

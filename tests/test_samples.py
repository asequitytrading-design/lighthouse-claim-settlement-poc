"""Sample use case (claim intake triage) and the evaluation dataset loader (offline)."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from lighthouse_poc.exceptions import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "claim_intake_triage"
REQUEST_TYPES = {"new_claim", "document_submission", "status_inquiry", "complaint",
                 "decision_request", "out_of_scope", "unclear"}
PRODUCTS = {"short_term_disability", "long_term_disability", "life", "accident",
            "critical_illness", "dental", "unknown"}


def load_script():
    spec = importlib.util.spec_from_file_location("load_eval_dataset",
                                                  ROOT / "scripts" / "load_eval_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def golden():
    return [json.loads(line) for line in
            (SAMPLE / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def test_golden_set_is_balanced_and_consistent_with_prompt():
    records = golden()
    system = (SAMPLE / "system_prompt_v1.txt").read_text(encoding="utf-8")
    ids = [r["tags"]["case_id"] for r in records]
    assert len(ids) == len(set(ids)) == 11
    assert sum(r["tags"]["scenario"] == "positive" for r in records) == 5
    assert sum(r["tags"]["scenario"].startswith("negative") for r in records) == 6
    for record in records:
        fields = record["expectations"]["expected_fields"]
        assert set(record["inputs"]) == {"claimant_message"}
        assert fields["request_type"] in REQUEST_TYPES and fields["product_line"] in PRODUCTS
        assert fields["benefit_determination"] == "Not determined"
        assert isinstance(fields["requires_human_review"], bool)
        assert record["expectations"]["guidelines"]
    for value in REQUEST_TYPES | PRODUCTS:
        assert f'"{value}"' in system  # every expected label is offered by the prompt


def test_user_prompt_variable_matches_dataset_inputs():
    assert "{{claimant_message}}" in (SAMPLE / "user_prompt.txt").read_text(encoding="utf-8")


def test_sample_contains_no_real_identifiers():
    text = (SAMPLE / "golden.jsonl").read_text(encoding="utf-8")
    assert "000-000-000" in text  # the identifier case uses an obviously fake number
    assert "@" not in text


def test_read_records_validates_shape(tmp_path):
    read_records = load_script().read_records
    assert len(read_records(SAMPLE / "golden.jsonl")) == 11
    cases = {"bad.jsonl": "{not json\n", "missing.jsonl": '{"inputs": {"a": 1}}\n',
             "empty.jsonl": "\n"}
    for name, content in cases.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigurationError):
            read_records(path)


def test_kong_bearer_mapping_runs_before_body_shims_and_keeps_key_auth():
    service = yaml.safe_load((ROOT / "kong" / "kong.yaml").read_text())["services"][0]
    plugins = {p["name"]: p["config"] for p in service["plugins"]}
    assert plugins["key-auth"]["key_names"] == ["apikey"]
    first, second = plugins["pre-function"]["access"]
    assert "authorization" in first and 'clear_header("authorization")' in first
    assert "response_format" in second and 'reasoning_effort = "none"' in second


def test_mlflow_service_allows_ui_gateway_to_reach_kong():
    mlflow = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["mlflow"]
    env = mlflow["environment"]
    assert "--artifacts-destination /mlflow-artifacts" in mlflow["command"]
    assert "--default-artifact-root" not in mlflow["command"]
    assert env["MLFLOW_GATEWAY_API_BASE_ALLOWED_SCHEMES"] == "http,https"
    assert env["MLFLOW_CRYPTO_KEK_PASSPHRASE"].startswith("${MLFLOW_CRYPTO_KEK_PASSPHRASE:?")

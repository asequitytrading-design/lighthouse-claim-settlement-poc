"""Unit tests for environment profiles, gateway client and output parsing (no network)."""
import json
import logging
from types import SimpleNamespace

import pytest

from lighthouse_poc.agent import ClaimSettlementAssistant, extract_json_object
from lighthouse_poc.config import load_dotenv, load_profile
from lighthouse_poc.exceptions import (ConfigurationError, ModelInvocationError,
                                       OutputValidationError)
from lighthouse_poc.gateway import (DatabricksSdkAuth, EnvHeaderAuth, GatewayChatClient,
                                    MockClaimModel, build_chat_model)

LOGGER = logging.getLogger("test")


# ---------------------------------------------------------------- profiles
def test_local_profile_resolves_defaults_and_routes():
    profile = load_profile("local", env={})
    route = profile.model("claims-small")
    assert profile.tracking_uri == "http://localhost:5000"
    assert profile.base_url_for(route) == "http://localhost:8100/ai/claims-small"
    assert route.request_model() == ""  # Kong pins the physical model per route
    assert route.params["reasoning_effort"] == "none"
    assert profile.model().alias == "claims-medium"


def test_env_overrides_default():
    profile = load_profile("local", env={"KONG_PROXY_URL": "http://kong:8000/ai"})
    assert profile.base_url_for(profile.model("claims-judge")) == "http://kong:8000/ai/claims-judge"


def test_databricks_profile_loads_without_gateway_vars_but_fails_on_use():
    profile = load_profile("databricks", env={})
    assert profile.prompt_name == "workspace.lighthouse.claim_settlement_review"
    assert isinstance(build_chat_model(profile, "mock", LOGGER), MockClaimModel)
    with pytest.raises(ConfigurationError, match="UNITY_GATEWAY_MEDIUM_MODEL"):
        profile.model("claims-medium").request_model()
    with pytest.raises(ConfigurationError, match="DATABRICKS_HOST"):
        profile.base_url_for(profile.model("claims-medium"))


def test_databricks_profile_resolves_with_vars():
    env = {"DATABRICKS_HOST": "https://x.cloud.databricks.com",
           "UNITY_GATEWAY_MEDIUM_MODEL": "system.ai.some-model", "LIGHTHOUSE_SCHEMA": "poc"}
    profile = load_profile("databricks", env=env)
    route = profile.model("claims-medium")
    assert profile.base_url_for(route) == "https://x.cloud.databricks.com/ai-gateway/mlflow/v1"
    assert route.request_model() == "system.ai.some-model"
    assert profile.prompt_name == "workspace.poc.claim_settlement_review"


def test_unknown_profile_and_model_are_configuration_errors():
    with pytest.raises(ConfigurationError, match="not found"):
        load_profile("nope", env={})
    with pytest.raises(ConfigurationError, match="not defined"):
        load_profile("local", env={}).model("gpt-unknown")


def test_invalid_profile_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("environment: x\nmlflow: {tracking_uri: a, experiment: b}\n"
                   "registry: {prompt_name: p}\ngateway: {kind: nginx, base_url: u}\nmodels: {}\n")
    with pytest.raises(ConfigurationError, match="gateway.kind"):
        load_profile(str(bad), env={})


def test_load_dotenv_does_not_override(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nA=from_file\nB='quoted'\n")
    env = {"A": "existing"}
    load_dotenv(env_file, env)
    assert env == {"A": "existing", "B": "quoted"}


# ---------------------------------------------------------------- auth
def test_env_header_auth_reads_secret_per_request():
    env = {"KONG_API_KEY": "k1"}
    auth = EnvHeaderAuth("KONG_API_KEY", "apikey", env=env)
    assert auth.headers() == {"apikey": "k1"}
    env["KONG_API_KEY"] = "k2"
    assert auth.headers() == {"apikey": "k2"}
    with pytest.raises(ConfigurationError):
        EnvHeaderAuth("MISSING", "apikey", env={}).headers()


def test_databricks_sdk_auth_uses_unified_config():
    tokens = iter(["t1", "t2"])
    cfg = SimpleNamespace(authenticate=lambda: {"Authorization": f"Bearer {next(tokens)}"})
    auth = DatabricksSdkAuth(lambda: cfg)
    options = auth.client_options()          # validates once (consumes t1)
    assert callable(options["api_key"]) and "default_headers" not in options
    assert options["api_key"]() == "t2"      # refreshed on every call


def test_client_options_keep_credentials_off_call_arguments():
    header = EnvHeaderAuth("KONG_API_KEY", "apikey", env={"KONG_API_KEY": "k1"})
    assert header.client_options()["default_headers"] == {"apikey": "k1"}
    bearer = EnvHeaderAuth("TOKEN", "Authorization", "Bearer ", env={"TOKEN": "b1"})
    options = bearer.client_options()
    assert options["api_key"]() == "b1" and "default_headers" not in options
    with pytest.raises(ConfigurationError):
        EnvHeaderAuth("MISSING", "Authorization", "Bearer ", env={}).client_options()


def test_real_sdk_client_carries_kong_key_in_default_headers(monkeypatch):
    monkeypatch.setenv("KONG_API_KEY", "kong-secret")
    profile = load_profile("local", env={})
    client = build_chat_model(profile, "claims-small", LOGGER)
    assert client._client.default_headers["apikey"] == "kong-secret"


# ---------------------------------------------------------------- gateway client
class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def response(content, model="qwen3.5-9b-32k:latest"):
    return SimpleNamespace(model=model,
                           choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def make_client(completions, alias="claims-medium"):
    profile = load_profile("local", env={})
    route = profile.model(alias)
    auth = EnvHeaderAuth("KONG_API_KEY", "apikey", env={"KONG_API_KEY": "secret"})
    return GatewayChatClient(profile.base_url_for(route), route, "kong", auth, LOGGER,
                             client=fake_client(completions))


def test_gateway_request_shape_and_completion():
    completions = FakeCompletions(response('{"a": 1}'))
    result = make_client(completions).complete("sys", "user")
    call = completions.calls[0]
    assert call["model"] == ""
    assert call["temperature"] == 0
    assert call["extra_body"] == {"reasoning_effort": "none"}
    # Credentials must never be call arguments: MLflow traces record them.
    assert call["extra_headers"] == {"X-Correlation-ID": result.correlation_id}
    assert "secret" not in repr(call)
    assert result.served_model == "qwen3.5-9b-32k:latest"
    assert (result.prompt_tokens, result.completion_tokens) == (10, 5)


def test_gateway_errors_are_typed_and_categorised():
    import httpx2
    import openai
    request = httpx2.Request("POST", "http://localhost:8100/ai/claims-medium/chat/completions")
    err = openai.AuthenticationError("401", response=httpx2.Response(401, request=request),
                                     body=None)
    with pytest.raises(ModelInvocationError) as info:
        make_client(FakeCompletions(error=err)).complete("s", "u")
    assert info.value.category == "auth"
    assert isinstance(info.value.__cause__, openai.AuthenticationError)


def test_gateway_keeps_only_text_parts_of_structured_content():
    content = [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
               {"type": "text", "text": '{"a": '}, {"type": "text", "text": "1}"}]
    result = make_client(FakeCompletions(response(content))).complete("s", "u")
    assert result.text == '{"a": 1}'


def test_gateway_reasoning_only_content_is_empty_error():
    with pytest.raises(ModelInvocationError) as info:
        make_client(FakeCompletions(response([{"type": "reasoning", "summary": []}]))).complete("s", "u")
    assert info.value.category == "empty_response"


def test_gateway_empty_content_is_an_error():
    with pytest.raises(ModelInvocationError) as info:
        make_client(FakeCompletions(response("  "))).complete("s", "u")
    assert info.value.category == "empty_response"


# ---------------------------------------------------------------- output contract
def test_extract_json_tolerates_fences_only():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object(' {"a": 1} ') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json_object("Here you go: {\"a\": 1}")
    with pytest.raises(ValueError):
        extract_json_object("[1, 2]")


class StaticRegistry:
    def load(self, name, version):
        return SimpleNamespace(format=lambda: "system prompt")


class StaticModel:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system_prompt, user_message):
        from lighthouse_poc.gateway import Completion
        return Completion(text=self.payload, duration_ms=1, served_model="fake")


def valid_output(**overrides):
    output = {key: [] for key in ClaimSettlementAssistant.LIST_FIELDS}
    output.update(claim_id="C-1", product_segment="dental", summary="s",
                  benefit_determination="Not determined")
    output.update(overrides)
    return output


def assistant(payload):
    return ClaimSettlementAssistant(StaticRegistry(), StaticModel(payload), LOGGER,
                                    "p", "1", "0.4.0")


def test_assistant_accepts_fenced_valid_output():
    payload = "```json\n" + json.dumps(valid_output()) + "\n```"
    assert assistant(payload).predict("C-1", "dental", "n", [], "c")["claim_id"] == "C-1"


@pytest.mark.parametrize("bad", [
    valid_output(benefit_determination="Approved"),
    valid_output(claim_id="OTHER"),
    valid_output(key_facts="not a list"),
    {"claim_id": "C-1"},
])
def test_assistant_rejects_contract_violations(bad):
    with pytest.raises(OutputValidationError):
        assistant(json.dumps(bad)).predict("C-1", "dental", "n", [], "c")

"""Bundle/code consistency and Databricks-path helpers (offline)."""
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lighthouse_poc.config import apply_env_overrides, load_profile
from lighthouse_poc.exceptions import ConfigurationError, PromptRegistryError
from lighthouse_poc.prompt_registry import MLflowPromptRegistry
from lighthouse_poc.runtime import publish_task_value

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("test")
RESOURCES = yaml.safe_load((ROOT / "resources" / "lighthouse.yml").read_text())
JOB = RESOURCES["resources"]["jobs"]["lighthouse_prompt_eval"]
TASKS = {task["task_key"]: task for task in JOB["tasks"]}


def env_args(task):
    params = task["spark_python_task"]["parameters"]
    return [params[i + 1] for i, p in enumerate(params) if p == "--env"]


# ---------------------------------------------------------------- bundle consistency
def test_task_scripts_exist_and_use_databricks_profile():
    for task in TASKS.values():
        script = (ROOT / "resources" / task["spark_python_task"]["python_file"]).resolve()
        assert script.is_file(), script
        params = task["spark_python_task"]["parameters"]
        assert params[params.index("--profile") + 1] == "databricks"
        assert task["environment_key"] in {e["environment_key"] for e in JOB["environments"]}


def test_job_parameter_defaults_name_real_profile_models():
    profile = load_profile("databricks", env={})
    defaults = {p["name"]: p["default"] for p in JOB["parameters"]}
    assert profile.model(defaults["app_model"]).kind == "mock"  # first run needs no model access
    assert profile.model(defaults["judge_model"]).kind == "mlflow_uri"
    assert defaults["scorer_mode"] in {"deterministic", "llm-judges"}


def test_env_overrides_cover_every_profile_variable():
    profile_text = (ROOT / "config" / "profiles" / "databricks.yaml").read_text()
    profile_vars = set(re.findall(r"\$\{([A-Z_]+)", profile_text))
    passed = {pair.split("=", 1)[0] for pair in env_args(TASKS["evaluate"])}
    # Everything the profile reads must be supplied by the job, except GIT_COMMIT (script)
    # which is extra lineage.
    assert profile_vars <= passed, profile_vars - passed
    assert passed - profile_vars <= {"GIT_COMMIT"}
    assert not any(k in passed for k in ("DATABRICKS_TOKEN", "KONG_API_KEY"))  # no secrets


def test_evaluate_consumes_registered_prompt_version():
    params = TASKS["evaluate"]["spark_python_task"]["parameters"]
    assert params[params.index("--prompt-version") + 1] == (
        "{{tasks.register_prompt.values.prompt_version}}")
    register = TASKS["register_prompt"]["spark_python_task"]["parameters"]
    assert register[register.index("--task-value-key") + 1] == "prompt_version"
    assert "--if-changed" in register


# ---------------------------------------------------------------- env overrides
def test_apply_env_overrides_sets_values_and_returns_names_only():
    env = {}
    assert apply_env_overrides(["A=1", "B=x=y", "EMPTY="], env) == ["A", "B", "EMPTY"]
    assert env == {"A": "1", "B": "x=y", "EMPTY": ""}
    for bad in (["noequals"], ["1BAD=x"], ["=x"]):
        with pytest.raises(ConfigurationError):
            apply_env_overrides(bad, {})


def test_empty_override_counts_as_unset_for_profiles():
    env = {"UNITY_GATEWAY_MEDIUM_MODEL": "", "DATABRICKS_HOST": "https://h"}
    with pytest.raises(ConfigurationError, match="UNITY_GATEWAY_MEDIUM_MODEL"):
        load_profile("databricks", env=env).model("claims-medium").request_model()


# ---------------------------------------------------------------- registry idempotency
class FakePromptApi:
    def __init__(self, latest=None, error=None):
        self.latest, self.error, self.registered = latest, error, []

    def load_prompt(self, name_or_uri):
        if self.error:
            raise self.error
        if self.latest is None:
            from mlflow.exceptions import MlflowException
            from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
            raise MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)
        return self.latest

    def register_prompt(self, name, template, commit_message, tags):
        self.registered.append(template)
        return SimpleNamespace(version=len(self.registered) + 1)


@pytest.fixture
def prompt_api(monkeypatch):
    def install(api):
        monkeypatch.setattr("mlflow.genai.load_prompt", api.load_prompt)
        monkeypatch.setattr("mlflow.genai.register_prompt", api.register_prompt)
        return api
    return install


def test_register_if_changed_reuses_identical_template(prompt_api):
    api = prompt_api(FakePromptApi(latest=SimpleNamespace(version=4, template="same")))
    ref, created = MLflowPromptRegistry(LOGGER).register_if_changed("c.s.p", "same", "note")
    assert (ref.version, created, api.registered) == ("4", False, [])


def test_register_if_changed_creates_version_for_new_or_changed(prompt_api):
    api = prompt_api(FakePromptApi(latest=SimpleNamespace(version=4, template="old")))
    _, created = MLflowPromptRegistry(LOGGER).register_if_changed("c.s.p", "new", "note")
    assert created and api.registered == ["new"]
    api = prompt_api(FakePromptApi(latest=None))
    _, created = MLflowPromptRegistry(LOGGER).register_if_changed("c.s.p", "first", "note")
    assert created and api.registered == ["first"]


def test_registry_lookup_errors_are_not_treated_as_missing(prompt_api):
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import PERMISSION_DENIED
    prompt_api(FakePromptApi(error=MlflowException("denied", error_code=PERMISSION_DENIED)))
    with pytest.raises(PromptRegistryError):
        MLflowPromptRegistry(LOGGER).register_if_changed("c.s.p", "t", "note")


# ---------------------------------------------------------------- task values
def test_publish_task_value_fails_loudly_outside_databricks(monkeypatch):
    monkeypatch.setitem(sys.modules, "databricks.sdk.runtime", None)  # import fails
    with pytest.raises(ConfigurationError, match="task value"):
        publish_task_value("prompt_version", "3", LOGGER)


def test_publish_task_value_uses_dbutils(monkeypatch):
    calls = []
    fake = SimpleNamespace(dbutils=SimpleNamespace(jobs=SimpleNamespace(
        taskValues=SimpleNamespace(set=lambda key, value: calls.append((key, value))))))
    monkeypatch.setitem(sys.modules, "databricks.sdk.runtime", fake)
    publish_task_value("prompt_version", "3", LOGGER)
    assert calls == [("prompt_version", "3")]

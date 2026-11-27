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
from lighthouse_poc.runtime import ensure_databricks_host, publish_task_value

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
    required = set(re.findall(r"\$\{([A-Z_]+)\}", profile_text))   # no ":-default"
    passed = {pair.split("=", 1)[0] for pair in env_args(TASKS["evaluate"])}
    # Every variable without a default must be supplied by the job; DATABRICKS_HOST is resolved
    # at runtime from databricks-sdk auth, GIT_COMMIT is extra lineage for the script.
    # Defaulted platform variables (ADR-012) are not used by the eval job.
    assert required - {"DATABRICKS_HOST"} <= passed, required - passed
    assert "DATABRICKS_HOST" not in passed  # ${workspace.host} is empty with CLI-profile auth
    assert passed - profile_vars <= {"GIT_COMMIT"}
    assert not any(k in passed for k in ("DATABRICKS_TOKEN", "KONG_API_KEY"))  # no secrets


def test_evaluate_scores_registered_version_against_champion():
    params = TASKS["evaluate"]["spark_python_task"]["parameters"]
    assert TASKS["evaluate"]["spark_python_task"]["python_file"].endswith("scripts/lighthouse.py")
    command = params[params.index("eval"):]
    assert command[command.index("--prompt") + 1] == "{{tasks.register_prompt.values.prompt_version}}"
    assert command[command.index("--baseline") + 1] == "@champion"
    assert "--baseline-optional" in command                  # first deploy: no champion yet
    assert command[command.index("--scorer-mode") + 1] == "{{job.parameters.scorer_mode}}"
    # Global options (profile, --env) come before the subcommand.
    assert params.index("--profile") < params.index("eval")
    assert max(i for i, p in enumerate(params) if p == "--env") < params.index("eval")
    from lighthouse_poc.cli import build_parser
    # Substitute what Databricks fills in: job parameter defaults and the task value.
    values = {f"{{{{job.parameters.{p['name']}}}}}": p["default"] for p in JOB["parameters"]}
    values["{{tasks.register_prompt.values.prompt_version}}"] = "3"
    args = build_parser().parse_args([values.get(p, p) for p in params])
    assert args.scorer_mode == "deterministic" and args.model == "mock" and args.prompt == "3"
    assert args.command == "eval" and args.profile == "databricks" and args.baseline_optional
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
class Page(list):
    def __init__(self, items, token=None):
        super().__init__(items)
        self.token = token


class FakePromptApi:
    """Fakes MlflowClient.search_prompt_versions plus mlflow.genai load/register."""

    def __init__(self, templates=None, pages=None, error=None):
        self.templates = templates or {}          # version -> template
        self.pages = pages                        # optional explicit paging
        self.error, self.registered, self.searched = error, [], []

    def search_prompt_versions(self, name, max_results=None, page_token=None):
        self.searched.append(page_token)
        if self.error:
            raise self.error
        if not self.templates:
            from mlflow.exceptions import MlflowException
            from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
            raise MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)
        if self.pages:
            return self.pages[page_token]
        return Page([SimpleNamespace(version=str(v)) for v in self.templates])

    def load_prompt(self, name_or_uri):
        version = int(name_or_uri.rsplit("/", 1)[1])
        return SimpleNamespace(version=version, template=self.templates[version])

    def register_prompt(self, name, template, commit_message, tags):
        self.registered.append(template)
        return SimpleNamespace(version=max(self.templates, default=0) + len(self.registered))


@pytest.fixture
def prompt_api(monkeypatch):
    def install(api):
        monkeypatch.setattr("mlflow.MlflowClient", lambda: api)
        monkeypatch.setattr("mlflow.genai.load_prompt", api.load_prompt)
        monkeypatch.setattr("mlflow.genai.register_prompt", api.register_prompt)
        return api
    return install


def test_register_if_changed_reuses_identical_latest_template(prompt_api):
    # Versions arrive unordered; the highest one is compared.
    api = prompt_api(FakePromptApi(templates={1: "old", 4: "same", 2: "older"}))
    ref, created = MLflowPromptRegistry(LOGGER).register_if_changed("c.s.p", "same", "note")
    assert (ref.version, created, api.registered) == ("4", False, [])


def test_register_if_changed_creates_version_for_new_or_changed(prompt_api):
    api = prompt_api(FakePromptApi(templates={4: "old"}))
    _, created = MLflowPromptRegistry(LOGGER).register_if_changed("c.s.p", "new", "note")
    assert created and api.registered == ["new"]
    api = prompt_api(FakePromptApi(templates={}))
    _, created = MLflowPromptRegistry(LOGGER).register_if_changed("c.s.p", "first", "note")
    assert created and api.registered == ["first"]


def test_latest_follows_pagination(prompt_api):
    pages = {None: Page([SimpleNamespace(version="1")], token="t2"),
             "t2": Page([SimpleNamespace(version="7")])}
    api = prompt_api(FakePromptApi(templates={1: "a", 7: "b"}, pages=pages))
    assert MLflowPromptRegistry(LOGGER).latest("c.s.p").version == 7
    assert api.searched == [None, "t2"]


def test_versions_newest_first_and_empty_when_missing(prompt_api):
    prompt_api(FakePromptApi(templates={2: "b", 10: "c", 1: "a"}))
    assert [v.version for v in MLflowPromptRegistry(LOGGER).versions("c.s.p")] == ["10", "2", "1"]
    prompt_api(FakePromptApi(templates={}))
    assert MLflowPromptRegistry(LOGGER).versions("c.s.p") == []


def test_load_alias_errors_are_typed(monkeypatch):
    def missing(name_or_uri):
        assert name_or_uri == "prompts:/c.s.p@champion"
        raise RuntimeError("NOT_FOUND")
    monkeypatch.setattr("mlflow.genai.load_prompt", missing)
    with pytest.raises(PromptRegistryError, match="@champion"):
        MLflowPromptRegistry(LOGGER).load_alias("c.s.p", "champion")


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


# ---------------------------------------------------------------- workspace host
def test_ensure_databricks_host_uses_sdk_when_unset():
    env = {}
    ensure_databricks_host(LOGGER, env, lambda: SimpleNamespace(host="https://h.example/"))
    assert env == {"DATABRICKS_HOST": "https://h.example"}
    env = {"DATABRICKS_HOST": "https://explicit"}
    ensure_databricks_host(LOGGER, env, lambda: SimpleNamespace(host="https://other"))
    assert env["DATABRICKS_HOST"] == "https://explicit"


def test_ensure_databricks_host_errors_are_typed():
    def broken():
        raise RuntimeError("no auth")
    with pytest.raises(ConfigurationError):
        ensure_databricks_host(LOGGER, {}, broken)
    with pytest.raises(ConfigurationError):
        ensure_databricks_host(LOGGER, {}, lambda: SimpleNamespace(host=None))

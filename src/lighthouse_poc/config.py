"""Environment profiles: one code path, configuration decides local vs Databricks.

A profile is a YAML file under ``config/profiles``. String values may reference
environment variables as ``${VAR}`` (required) or ``${VAR:-default}``. Unset required
variables only fail when the value is actually needed, so a Databricks profile can be
loaded to run the mock model without Unity Gateway variables present.
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from .exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = PROJECT_ROOT / "config" / "profiles"
PROFILE_ENV_VAR = "LIGHTHOUSE_PROFILE"
GATEWAY_KINDS = {"kong", "unity_gateway"}
AUTH_TYPES = {"none", "api_key_header", "bearer_env", "databricks_sdk"}
MODEL_KINDS = {"gateway", "mock", "mlflow_uri"}
MODEL_TASKS = {"chat", "embeddings"}

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _Unresolved(str):
    """Placeholder for a value whose required environment variable is unset."""

    def __new__(cls, raw: str, missing: tuple[str, ...]):
        obj = super().__new__(cls, raw)
        obj.missing = missing
        return obj


def _interpolate(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {k: _interpolate(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, env) for v in value]
    if not isinstance(value, str):
        return value
    missing: list[str] = []

    def substitute(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        resolved = env.get(name) or default
        if resolved is None:
            missing.append(name)
            return match.group(0)
        return resolved

    result = _VAR.sub(substitute, value)
    return _Unresolved(value, tuple(missing)) if missing else result


def require(value: str | None, what: str) -> str:
    """Return a resolved config value or raise a ConfigurationError naming the variable."""
    if isinstance(value, _Unresolved):
        raise ConfigurationError(
            f"{what} needs environment variable(s) {', '.join(value.missing)} (from '{value}')")
    if value is None or value == "":
        raise ConfigurationError(f"{what} is not configured")
    return value


@dataclass(frozen=True)
class AuthConfig:
    type: str = "none"
    header: str | None = None
    env: str | None = None


@dataclass(frozen=True)
class GatewayConfig:
    kind: str
    base_url: str
    auth: AuthConfig
    timeout_seconds: float = 120.0
    max_retries: int = 1


@dataclass(frozen=True)
class ModelRoute:
    """A logical model name and how this environment's gateway serves it."""
    alias: str
    kind: str = "gateway"
    route: str | None = None
    upstream_model: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    model_uri: str | None = None  # kind=mlflow_uri only, e.g. "databricks" managed judge
    task: str = "chat"            # chat | embeddings

    def request_model(self) -> str:
        """Value for the OpenAI ``model`` field. Empty lets a pinning gateway choose."""
        if self.upstream_model is None:
            return ""
        return require(self.upstream_model, f"upstream_model for '{self.alias}'")


@dataclass(frozen=True)
class EvaluationConfig:
    """Concurrency limits for mlflow.genai.evaluate (local models need low values)."""
    max_workers: int | None = None
    max_scorer_workers: int | None = None
    judge_timeout_seconds: int | None = None

    def environment(self) -> dict[str, str]:
        env = {}
        if self.max_workers:
            env["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(self.max_workers)
        if self.max_scorer_workers:
            env["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = str(self.max_scorer_workers)
        if self.judge_timeout_seconds:
            env["MLFLOW_GENAI_EVAL_LLM_TIMEOUT"] = str(self.judge_timeout_seconds)
        return env


@dataclass(frozen=True)
class PlatformConfig:
    """Unity Catalog data platform of the AI Marketplace (Databricks profile only)."""
    catalog: str
    schema_prefix: str = ""
    warehouse_id: str | None = None
    vector_search_endpoint: str = "lighthouse-vs"
    embedding_model: str = "embed-default"


@dataclass(frozen=True)
class Profile:
    name: str
    environment: str
    tracking_uri: str
    experiment: str
    prompt_name: str
    gateway: GatewayConfig
    models: Mapping[str, ModelRoute]
    default_app_model: str
    default_judge_model: str
    source: Path
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    registry_uri: str | None = None
    platform: PlatformConfig | None = None

    def model(self, alias: str | None = None) -> ModelRoute:
        alias = alias or self.default_app_model
        try:
            return self.models[alias]
        except KeyError:
            raise ConfigurationError(
                f"Model '{alias}' is not defined in profile '{self.name}'. "
                f"Available: {sorted(self.models)}") from None

    def base_url_for(self, route: ModelRoute) -> str:
        base = require(self.gateway.base_url, "gateway.base_url").rstrip("/")
        return f"{base}/{route.route}" if route.route else base

    def lineage(self, route: ModelRoute) -> dict[str, str]:
        """Log-safe description of where a model call goes (no secrets)."""
        if route.kind == "mock":
            return {"environment": self.environment, "gateway": "none",
                    "model_alias": route.alias, "gateway_route": "mock"}
        if route.kind == "mlflow_uri":
            return {"environment": self.environment, "gateway": "mlflow",
                    "model_alias": route.alias, "gateway_route": str(route.model_uri)}
        return {"environment": self.environment, "gateway": self.gateway.kind,
                "model_alias": route.alias,
                "gateway_route": route.route or str(route.upstream_model)}


def _section(data: Mapping, key: str, source: Path) -> Mapping:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Profile {source.name} is missing the '{key}' section")
    return value


def _parse(name: str, data: Mapping, source: Path) -> Profile:
    mlflow_cfg = _section(data, "mlflow", source)
    registry = _section(data, "registry", source)
    gateway = _section(data, "gateway", source)
    auth = gateway.get("auth") or {"type": "none"}
    gateway_kind = gateway.get("kind")
    if gateway_kind not in GATEWAY_KINDS:
        raise ConfigurationError(f"gateway.kind must be one of {sorted(GATEWAY_KINDS)}")
    if auth.get("type") not in AUTH_TYPES:
        raise ConfigurationError(f"gateway.auth.type must be one of {sorted(AUTH_TYPES)}")

    models: dict[str, ModelRoute] = {}
    for alias, spec in _section(data, "models", source).items():
        spec = spec or {}
        kind = spec.get("kind", "gateway")
        if kind not in MODEL_KINDS:
            raise ConfigurationError(f"models.{alias}.kind must be one of {sorted(MODEL_KINDS)}")
        if kind == "gateway" and not (spec.get("route") or "upstream_model" in spec):
            raise ConfigurationError(f"models.{alias} needs a 'route' or 'upstream_model'")
        if kind == "mlflow_uri" and not spec.get("model_uri"):
            raise ConfigurationError(f"models.{alias} (kind mlflow_uri) needs 'model_uri'")
        task = spec.get("task", "chat")
        if task not in MODEL_TASKS:
            raise ConfigurationError(f"models.{alias}.task must be one of {sorted(MODEL_TASKS)}")
        models[alias] = ModelRoute(alias=alias, kind=kind, route=spec.get("route"),
                                   upstream_model=spec.get("upstream_model"),
                                   params=dict(spec.get("params") or {}),
                                   model_uri=spec.get("model_uri"), task=task)

    defaults = data.get("defaults") or {}
    evaluation = data.get("evaluation") or {}
    platform = data.get("platform")
    platform_config = None
    if platform is not None:
        if not isinstance(platform, Mapping) or not platform.get("catalog"):
            raise ConfigurationError(f"Profile {source.name}: platform.catalog is required")
        platform_config = PlatformConfig(
            catalog=str(platform["catalog"]),
            schema_prefix=str(platform.get("schema_prefix") or ""),
            warehouse_id=str(platform["warehouse_id"]) if platform.get("warehouse_id") else None,
            vector_search_endpoint=str(platform.get("vector_search_endpoint") or "lighthouse-vs"),
            embedding_model=str(platform.get("embedding_model") or "embed-default"))
    profile = Profile(
        name=name,
        environment=str(data.get("environment", name)),
        tracking_uri=mlflow_cfg.get("tracking_uri"),
        registry_uri=mlflow_cfg.get("registry_uri"),
        experiment=mlflow_cfg.get("experiment"),
        prompt_name=registry.get("prompt_name"),
        gateway=GatewayConfig(
            kind=gateway_kind, base_url=gateway.get("base_url"),
            auth=AuthConfig(type=auth["type"], header=auth.get("header"), env=auth.get("env")),
            timeout_seconds=float(gateway.get("timeout_seconds", 120)),
            max_retries=int(gateway.get("max_retries", 1))),
        models=models,
        default_app_model=defaults.get("app_model", "mock"),
        default_judge_model=defaults.get("judge_model", ""),
        source=source,
        evaluation=EvaluationConfig(
            max_workers=int(evaluation["max_workers"]) if evaluation.get("max_workers") else None,
            max_scorer_workers=(int(evaluation["max_scorer_workers"])
                                if evaluation.get("max_scorer_workers") else None),
            judge_timeout_seconds=(int(evaluation["judge_timeout_seconds"])
                                   if evaluation.get("judge_timeout_seconds") else None)),
        platform=platform_config,
    )
    profile.model(profile.default_app_model)  # fail fast on a dangling default
    if profile.default_judge_model:
        profile.model(profile.default_judge_model)
    return profile


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def apply_env_overrides(pairs: list[str] | None, env: dict | None = None) -> list[str]:
    """Apply ``KEY=VALUE`` overrides (e.g. from job parameters) to the environment.

    Serverless job tasks cannot set environment variables portably, so bundles pass
    profile inputs as ``--env`` arguments. Only non-secret values belong here; secrets
    come from workspace auth or secret scopes. Returns the names set (never values).
    """
    env = os.environ if env is None else env
    names = []
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not _ENV_NAME.match(key):
            raise ConfigurationError(f"--env expects KEY=VALUE, got '{key}'")
        env[key] = value
        names.append(key)
    return names


@contextmanager
def scoped_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set environment variables for a block, then restore the previous values."""
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_dotenv(path: Path | None = None, env: dict | None = None) -> None:
    """Load KEY=VALUE lines into the environment without overriding existing values."""
    path = path or PROJECT_ROOT / ".env"
    env = os.environ if env is None else env
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_profile(name: str | None = None, env: Mapping[str, str] | None = None,
                 profile_dir: Path | None = None) -> Profile:
    """Load a profile by name (or path). Falls back to $LIGHTHOUSE_PROFILE, then 'local'.

    The profile directory defaults to the repository's config/profiles and can be
    overridden with $LIGHTHOUSE_PROFILE_DIR (e.g. for an installed wheel).
    """
    env = os.environ if env is None else env
    profile_dir = profile_dir or Path(env.get("LIGHTHOUSE_PROFILE_DIR") or PROFILE_DIR)
    name = name or env.get(PROFILE_ENV_VAR) or "local"
    source = Path(name) if name.endswith((".yaml", ".yml")) else profile_dir / f"{name}.yaml"
    if not source.is_file():
        available = sorted(p.stem for p in profile_dir.glob("*.yaml"))
        raise ConfigurationError(f"Profile '{name}' not found. Available: {available}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Profile {source.name} is not valid YAML") from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"Profile {source.name} must be a mapping")
    return _parse(source.stem, _interpolate(raw, env), source)

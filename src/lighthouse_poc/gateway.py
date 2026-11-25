"""OpenAI-compatible AI gateway client (Kong locally, Unity Gateway on Databricks).

Both gateways expose the OpenAI chat-completions API, so one client serves both.
What differs is configuration (base URL, route, model field, request params) and the
injected AuthProvider. Authentication headers are produced per request so short-lived
OAuth tokens refresh during long evaluation runs.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .config import AuthConfig, ModelRoute, Profile, require
from .exceptions import ConfigurationError, ModelInvocationError


@dataclass(frozen=True)
class Completion:
    text: str
    duration_ms: int
    served_model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    correlation_id: str | None = None


class ChatModel(Protocol):
    """What the assistant needs from any model adapter."""
    def complete(self, system_prompt: str, user_message: str) -> Completion: ...


class AuthProvider(Protocol):
    def headers(self) -> dict[str, str]: ...


class NoAuth:
    def headers(self) -> dict[str, str]:
        return {}


class EnvHeaderAuth:
    """Reads a secret from an environment variable into a header on every request."""
    def __init__(self, env_var: str, header: str, prefix: str = "",
                 env: Mapping[str, str] | None = None):
        self._env_var, self._header, self._prefix = env_var, header, prefix
        self._env = os.environ if env is None else env

    def headers(self) -> dict[str, str]:
        secret = self._env.get(self._env_var)
        if not secret:
            raise ConfigurationError(f"Environment variable {self._env_var} is not set")
        return {self._header: f"{self._prefix}{secret}"}


class DatabricksSdkAuth:
    """Databricks unified auth (notebook/job identity, OAuth, PAT) via databricks-sdk."""
    def __init__(self, config_factory: Callable[[], Any] | None = None):
        self._factory = config_factory
        self._config = None

    def headers(self) -> dict[str, str]:
        if self._config is None:
            try:
                if self._factory is None:
                    from databricks.sdk.core import Config
                    self._factory = Config
                self._config = self._factory()
            except Exception as exc:
                raise ConfigurationError("Databricks authentication is not configured") from exc
        return dict(self._config.authenticate())


def build_auth(auth: AuthConfig) -> AuthProvider:
    if auth.type == "none":
        return NoAuth()
    if auth.type == "api_key_header":
        return EnvHeaderAuth(require(auth.env, "gateway.auth.env"),
                             require(auth.header, "gateway.auth.header"))
    if auth.type == "bearer_env":
        return EnvHeaderAuth(require(auth.env, "gateway.auth.env"), "Authorization", "Bearer ")
    if auth.type == "databricks_sdk":
        return DatabricksSdkAuth()
    raise ConfigurationError(f"Unsupported auth type {auth.type}")


def _categorize(exc: Exception) -> str:
    import openai
    for kind, label in ((openai.AuthenticationError, "auth"),
                        (openai.PermissionDeniedError, "permission"),
                        (openai.NotFoundError, "not_found"),
                        (openai.RateLimitError, "rate_limit"),
                        (openai.APITimeoutError, "timeout"),
                        (openai.APIConnectionError, "connection"),
                        (openai.BadRequestError, "bad_request"),
                        (openai.APIStatusError, "upstream"),
                        (ConfigurationError, "configuration")):
        if isinstance(exc, kind):
            return label
    return "unknown"


class GatewayChatClient:
    """Chat model reached through an OpenAI-compatible gateway route."""

    def __init__(self, base_url: str, route: ModelRoute, gateway_kind: str,
                 auth: AuthProvider, logger: logging.Logger, *, timeout_seconds: float = 120,
                 max_retries: int = 1, client: Any = None):
        self._route, self._gateway_kind = route, gateway_kind
        self._auth, self._logger = auth, logger
        self._request_model = route.request_model()
        if client is None:
            try:
                from openai import OpenAI
                # Real credentials travel in per-request headers from the AuthProvider.
                client = OpenAI(base_url=base_url, api_key="unused-see-auth-provider",
                                timeout=timeout_seconds, max_retries=max_retries)
            except Exception as exc:
                logger.exception("gateway_client_initialization_failed", extra={
                    "model_alias": route.alias, "gateway": gateway_kind,
                    "error_type": type(exc).__name__, "error_category": "client_init"})
                raise ModelInvocationError("Could not initialise gateway client",
                                           "client_init") from exc
        self._client = client

    def complete(self, system_prompt: str, user_message: str) -> Completion:
        correlation_id = _current_trace_id() or str(uuid.uuid4())
        started = time.perf_counter()
        log = {"model_alias": self._route.alias, "gateway": self._gateway_kind,
               "correlation_id": correlation_id}
        try:
            headers = {**self._auth.headers(), "X-Correlation-ID": correlation_id}
            params = dict(self._route.params)
            standard = {k: params.pop(k) for k in ("temperature", "max_tokens", "top_p")
                        if k in params}
            result = self._client.chat.completions.create(
                model=self._request_model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_message}],
                extra_headers=headers,
                extra_body=params or None,
                **standard,
            )
        except Exception as exc:
            category = _categorize(exc)
            self._logger.error("model_request_failed", extra={
                **log, "error_type": type(exc).__name__, "error_category": category,
                "duration_ms": round((time.perf_counter() - started) * 1000),
                "kpi_name": "gateway_request_failed", "kpi_value": 1})
            raise ModelInvocationError(
                f"Gateway request failed for model '{self._route.alias}' ({category})",
                category) from exc

        elapsed = round((time.perf_counter() - started) * 1000)
        text = (result.choices[0].message.content or "") if result.choices else ""
        usage = getattr(result, "usage", None)
        completion = Completion(
            text=text, duration_ms=elapsed, served_model=getattr(result, "model", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            correlation_id=correlation_id)
        if not text.strip():
            self._logger.error("model_request_empty", extra={
                **log, "served_model": completion.served_model, "duration_ms": elapsed,
                "error_category": "empty_response"})
            raise ModelInvocationError(
                f"Gateway returned empty content for model '{self._route.alias}'",
                "empty_response")
        self._logger.info("model_request_completed", extra={
            **log, "served_model": completion.served_model, "duration_ms": elapsed,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "kpi_name": "model_latency_ms", "kpi_value": elapsed})
        return completion


class MockClaimModel:
    """Deterministic local response adapter for wiring checks only (not model quality)."""

    def complete(self, system_prompt: str, user_message: str) -> Completion:
        import ast
        import re
        started = time.perf_counter()

        def value(label, fallback):
            match = re.search(rf"^{re.escape(label)}:\s*(.*)$", user_message, re.MULTILINE)
            return match.group(1).strip() if match else fallback

        narrative = value("Narrative", "Synthetic claim details supplied")
        try:
            docs = ast.literal_eval(value("Documents listed as received", "[]"))
            docs = docs if isinstance(docs, list) else [str(docs)]
        except (ValueError, SyntaxError):
            docs = ["Submitted documents listed in synthetic case"]
        output = {
            "claim_id": value("Case ID", "SYNTHETIC-CASE"),
            "product_segment": value("Segment", "Unknown synthetic segment"),
            "summary": narrative[:400],
            "key_facts": [narrative[:250]],
            "evidence_received": docs,
            "evidence_to_confirm": ["Confirm applicable claim-specific evidence with examiner"],
            "inconsistencies": [],
            "recommended_examiner_actions": [
                "Review submitted materials and apply approved policy procedures"],
            "risk_flags": ["Human examiner review required"],
            "benefit_determination": "Not determined",
        }
        return Completion(text=json.dumps(output), served_model="mock",
                          duration_ms=round((time.perf_counter() - started) * 1000))


class ServedModelRecorder:
    """ChatModel decorator that records which physical models actually served calls."""

    def __init__(self, inner: ChatModel):
        self._inner = inner
        self.served_models: set[str] = set()

    def complete(self, system_prompt: str, user_message: str) -> Completion:
        completion = self._inner.complete(system_prompt, user_message)
        if completion.served_model:
            self.served_models.add(completion.served_model)
        return completion


def _current_trace_id() -> str | None:
    try:
        import mlflow
        span = mlflow.get_current_active_span()
        return getattr(span, "trace_id", None) if span else None
    except Exception:  # tracing is optional context for correlation only
        return None


def build_chat_model(profile: Profile, alias: str | None, logger: logging.Logger) -> ChatModel:
    """Factory: resolve a logical model name in a profile to a ChatModel adapter."""
    route = profile.model(alias)
    if route.kind == "mock":
        return MockClaimModel()
    return GatewayChatClient(
        profile.base_url_for(route), route, profile.gateway.kind,
        build_auth(profile.gateway.auth), logger,
        timeout_seconds=profile.gateway.timeout_seconds,
        max_retries=profile.gateway.max_retries)

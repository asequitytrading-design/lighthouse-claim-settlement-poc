"""OpenAI-compatible AI gateway client (Kong locally, Unity Gateway on Databricks).

Both gateways expose the OpenAI chat-completions API, so one client serves both.
What differs is configuration (base URL, route, model field, request params) and the
injected AuthProvider.

Credentials are configured on the SDK client (``default_headers`` / ``api_key``), never
passed as per-call arguments: MLflow auto-traces OpenAI calls during evaluation and stores
call arguments such as ``extra_headers`` in trace spans. Bearer tokens use the SDK's
callable ``api_key`` so short-lived OAuth tokens refresh per request.
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
from .usage import CallRecord, UsageTracker, active_tracker, context_from_profile


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

    def complete_messages(self, messages: list[dict]) -> Completion: ...


_PLACEHOLDER_API_KEY = "unused-auth-in-default-headers"
REQUEST_TAGS_HEADER = "Databricks-Ai-Gateway-Request-Tags"


def request_tag_headers(gateway_kind: str, tags: Mapping[str, Any] | None) -> dict[str, str]:
    """Unity Gateway attribution: tags land in system.ai_gateway.usage.request_tags (ADR-013).

    Only non-secret, non-personal identifiers (use case, prompt version, synthetic case ID,
    event ID). Other gateways ignore the header, so it is sent only to Unity Gateway.
    """
    if gateway_kind != "unity_gateway" or not tags:
        return {}
    clean = {str(k): str(v) for k, v in tags.items() if v is not None and str(v) != ""}
    return {REQUEST_TAGS_HEADER: json.dumps(clean, sort_keys=True)} if clean else {}


@dataclass(frozen=True)
class Embeddings:
    vectors: list[list[float]]
    duration_ms: int
    served_model: str | None = None
    prompt_tokens: int | None = None
    correlation_id: str | None = None

    @property
    def dimensions(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


class AuthProvider(Protocol):
    def headers(self) -> dict[str, str]:
        """Request headers carrying the credential (for non-SDK callers such as judges)."""
        ...

    def client_options(self) -> dict[str, Any]:
        """OpenAI client constructor options carrying the credential."""
        ...


class NoAuth:
    def headers(self) -> dict[str, str]:
        return {}

    def client_options(self) -> dict[str, Any]:
        return {"api_key": _PLACEHOLDER_API_KEY}


class EnvHeaderAuth:
    """Reads a secret from an environment variable into a header on every request."""
    def __init__(self, env_var: str, header: str, prefix: str = "",
                 env: Mapping[str, str] | None = None):
        self._env_var, self._header, self._prefix = env_var, header, prefix
        self._env = os.environ if env is None else env

    def _secret(self) -> str:
        secret = self._env.get(self._env_var)
        if not secret:
            raise ConfigurationError(f"Environment variable {self._env_var} is not set")
        return secret

    def headers(self) -> dict[str, str]:
        return {self._header: f"{self._prefix}{self._secret()}"}

    def client_options(self) -> dict[str, Any]:
        if self._header.lower() == "authorization" and self._prefix.strip().lower() == "bearer":
            self._secret()  # fail fast when unset
            return {"api_key": self._secret}
        return {"api_key": _PLACEHOLDER_API_KEY, "default_headers": self.headers()}


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

    def _token(self) -> str:
        authorization = self.headers().get("Authorization", "")
        if not authorization.lower().startswith("bearer "):
            raise ConfigurationError("Databricks authentication did not return a bearer token")
        return authorization.split(" ", 1)[1]

    def client_options(self) -> dict[str, Any]:
        self._token()  # fail fast with a ConfigurationError
        return {"api_key": self._token}


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


def _content_text(content: Any) -> str:
    """Normalise message content to text.

    Most models return a string. Reasoning models behind Unity Gateway (e.g. Qwen3.5) return
    a list of typed parts such as ``{"type": "reasoning", ...}`` and ``{"type": "text",
    "text": ...}``; only the text parts are the answer.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if kind in ("text", "output_text") and isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)


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
                 max_retries: int = 1, client: Any = None, tracker: UsageTracker | None = None):
        self._route, self._gateway_kind = route, gateway_kind
        self._auth, self._logger = auth, logger
        # the active run's tracker wins; this one gives the profile's default tags
        self._tracker = tracker or UsageTracker()
        self._request_model = route.request_model()
        if client is None:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=base_url, timeout=timeout_seconds,
                                max_retries=max_retries, **auth.client_options())
            except ConfigurationError:
                raise
            except Exception as exc:
                logger.exception("gateway_client_initialization_failed", extra={
                    "model_alias": route.alias, "gateway": gateway_kind,
                    "error_type": type(exc).__name__, "error_category": "client_init"})
                raise ModelInvocationError("Could not initialise gateway client",
                                           "client_init") from exc
        self._client = client

    def complete(self, system_prompt: str, user_message: str) -> Completion:
        return self.complete_messages([{"role": "system", "content": system_prompt},
                                       {"role": "user", "content": user_message}])

    def complete_messages(self, messages: list[dict],
                          tags: Mapping[str, Any] | None = None) -> Completion:
        """One chat completion for an arbitrary message list (e.g. a registered chat prompt).

        Tagged and recorded through the usage tracker (lighthouse_poc.usage); explicit `tags`
        override the tracker's context for this call."""
        tracker = active_tracker() or self._tracker
        event_id, context, call_tags, owned = tracker.begin(self._route.alias, tags)
        record = CallRecord(event_id, context, self._route.alias, self._request_model, "chat",
                            trace_id=_current_trace_id())
        started = time.perf_counter()
        try:
            completion = self._complete(messages, call_tags)
            record.input_tokens, record.output_tokens = (completion.prompt_tokens,
                                                         completion.completion_tokens)
            record.served_model = completion.served_model
            return completion
        except ModelInvocationError as exc:
            record.status, record.error_category = "error", exc.category
            raise
        finally:
            record.latency_ms = round((time.perf_counter() - started) * 1000)
            tracker.finish(record, owned)

    def _complete(self, messages: list[dict], tags: Mapping[str, Any]) -> Completion:
        correlation_id = _current_trace_id() or str(uuid.uuid4())
        started = time.perf_counter()
        log = {"model_alias": self._route.alias, "gateway": self._gateway_kind,
               "correlation_id": correlation_id}
        try:
            # Only non-secret headers here: call arguments are recorded in MLflow traces.
            headers = {"X-Correlation-ID": correlation_id,
                       **request_tag_headers(self._gateway_kind, tags)}
            params = dict(self._route.params)
            standard = {k: params.pop(k) for k in ("temperature", "max_tokens", "top_p")
                        if k in params}
            result = self._client.chat.completions.create(
                model=self._request_model,
                messages=messages,
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
        text = _content_text(result.choices[0].message.content) if result.choices else ""
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

    def complete_messages(self, messages: list[dict]) -> Completion:
        system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        user = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") != "system")
        return self.complete(system, user)

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
        return self._record(self._inner.complete(system_prompt, user_message))

    def complete_messages(self, messages: list[dict]) -> Completion:
        return self._record(self._inner.complete_messages(messages))

    def _record(self, completion: Completion) -> Completion:
        if completion.served_model:
            self.served_models.add(completion.served_model)
        return completion


class GatewayEmbeddingClient:
    """Embedding model reached through an OpenAI-compatible gateway route (Unity Gateway)."""

    def __init__(self, base_url: str, route: ModelRoute, gateway_kind: str,
                 auth: AuthProvider, logger: logging.Logger, *, timeout_seconds: float = 120,
                 max_retries: int = 1, client: Any = None, tracker: UsageTracker | None = None):
        self._route, self._gateway_kind, self._logger = route, gateway_kind, logger
        self._tracker = tracker or UsageTracker()
        self._request_model = route.request_model()
        if client is None:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=base_url, timeout=timeout_seconds,
                                max_retries=max_retries, **auth.client_options())
            except ConfigurationError:
                raise
            except Exception as exc:
                logger.exception("gateway_client_initialization_failed", extra={
                    "model_alias": route.alias, "gateway": gateway_kind,
                    "error_type": type(exc).__name__, "error_category": "client_init"})
                raise ModelInvocationError("Could not initialise gateway client",
                                           "client_init") from exc
        self._client = client

    @property
    def alias(self) -> str:
        return self._route.alias

    @property
    def model_service(self) -> str:
        return self._request_model

    def embed(self, texts: list[str], tags: Mapping[str, Any] | None = None) -> Embeddings:
        if not texts:
            return Embeddings(vectors=[], duration_ms=0)
        tracker = active_tracker() or self._tracker
        event_id, context, call_tags, owned = tracker.begin(self._route.alias, tags,
                                                           purpose="embedding_query")
        record = CallRecord(event_id, context, self._route.alias, self._request_model,
                            "embeddings", input_count=len(texts), trace_id=_current_trace_id())
        started = time.perf_counter()
        try:
            embeddings = self._embed(texts, call_tags)
            record.input_tokens, record.served_model = embeddings.prompt_tokens, embeddings.served_model
            record.embedding_dimensions = embeddings.dimensions
            return embeddings
        except ModelInvocationError as exc:
            record.status, record.error_category = "error", exc.category
            raise
        finally:
            record.latency_ms = round((time.perf_counter() - started) * 1000)
            tracker.finish(record, owned)

    def _embed(self, texts: list[str], tags: Mapping[str, Any]) -> Embeddings:
        correlation_id = str(uuid.uuid4())
        started = time.perf_counter()
        log = {"model_alias": self._route.alias, "gateway": self._gateway_kind,
               "correlation_id": correlation_id, "input_count": len(texts)}
        try:
            result = self._client.embeddings.create(
                model=self._request_model, input=texts,
                extra_headers={"X-Correlation-ID": correlation_id,
                               **request_tag_headers(self._gateway_kind, tags)})
        except Exception as exc:
            category = _categorize(exc)
            self._logger.error("embedding_request_failed", extra={
                **log, "error_type": type(exc).__name__, "error_category": category,
                "duration_ms": round((time.perf_counter() - started) * 1000),
                "kpi_name": "gateway_request_failed", "kpi_value": 1})
            raise ModelInvocationError(
                f"Gateway embedding request failed for model '{self._route.alias}' ({category})",
                category) from exc
        elapsed = round((time.perf_counter() - started) * 1000)
        vectors = [list(item.embedding) for item in sorted(result.data, key=lambda d: d.index)]
        usage = getattr(result, "usage", None)
        embeddings = Embeddings(vectors=vectors, duration_ms=elapsed,
                                served_model=getattr(result, "model", None),
                                prompt_tokens=getattr(usage, "prompt_tokens", None),
                                correlation_id=correlation_id)
        if len(vectors) != len(texts) or not embeddings.dimensions:
            self._logger.error("embedding_response_invalid", extra={
                **log, "returned": len(vectors), "error_category": "invalid_response"})
            raise ModelInvocationError(
                f"Gateway returned {len(vectors)} embeddings for {len(texts)} inputs",
                "invalid_response")
        self._logger.info("embedding_request_completed", extra={
            **log, "served_model": embeddings.served_model, "duration_ms": elapsed,
            "dimensions": embeddings.dimensions, "prompt_tokens": embeddings.prompt_tokens,
            "kpi_name": "embedding_latency_ms", "kpi_value": elapsed})
        return embeddings


def build_embedding_model(profile: Profile, alias: str,
                          logger: logging.Logger) -> GatewayEmbeddingClient:
    """Factory: an embeddings-task model of the profile, reached through its gateway."""
    route = profile.model(alias)
    if route.kind != "gateway" or route.task != "embeddings":
        raise ConfigurationError(
            f"Model '{route.alias}' is not a gateway embeddings model (kind {route.kind}, "
            f"task {route.task})")
    return GatewayEmbeddingClient(
        profile.base_url_for(route), route, profile.gateway.kind,
        build_auth(profile.gateway.auth), logger,
        timeout_seconds=profile.gateway.timeout_seconds,
        max_retries=profile.gateway.max_retries,
        tracker=UsageTracker(context_from_profile(profile, purpose="embedding_query")))


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
    if route.kind != "gateway":
        raise ConfigurationError(
            f"Model '{route.alias}' (kind {route.kind}) can only be used as an LLM judge")
    return GatewayChatClient(
        profile.base_url_for(route), route, profile.gateway.kind,
        build_auth(profile.gateway.auth), logger,
        timeout_seconds=profile.gateway.timeout_seconds,
        max_retries=profile.gateway.max_retries,
        tracker=UsageTracker(context_from_profile(profile)))

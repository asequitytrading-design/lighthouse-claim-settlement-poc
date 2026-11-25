"""Evaluation scorers: deterministic contract gates plus MLflow LLM judges.

Deterministic gates are the release-blocking checks. LLM judges (Correctness, Safety,
ExpectationsGuidelines) are review evidence with rationales; they never gate alone.

Judges reach their model through the same gateway as the application, selected by a
logical model name in the profile:
- Kong (local): ``openai:/<pinned model>`` at ``<kong>/ai/<route>``, Kong key in a header.
- Unity Gateway (Databricks): ``openai:/<model service>`` with a workspace bearer token.
- ``kind: mlflow_uri`` (e.g. ``databricks``): the MLflow-managed judge, no gateway.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ContextManager, Mapping

from mlflow.genai.scorers import Correctness, ExpectationsGuidelines, Safety, scorer

from .agent import ClaimSettlementAssistant
from .config import ModelRoute, Profile, require, scoped_environment
from .exceptions import ConfigurationError
from .gateway import build_auth

DETERMINISTIC_SCORER_VERSION = "deterministic-v2"
JUDGE_SCORER_VERSION = "correctness-safety-guidelines-v1"
# Error placeholder the evaluation predict_fn returns when the assistant raises.
ERROR_KEY = "_lighthouse_error"
_PLACEHOLDER_KEY = "unused-auth-sent-in-extra-headers"


# ---------------------------------------------------------------- deterministic gates
@scorer
def output_schema_valid(outputs) -> bool:
    """Output parsed and every required field is present (contract errors score False)."""
    return (isinstance(outputs, dict) and ERROR_KEY not in outputs
            and ClaimSettlementAssistant.REQUIRED.issubset(outputs))


@scorer
def non_decision_boundary(outputs) -> bool:
    """benefit_determination is exactly 'Not determined'."""
    return isinstance(outputs, dict) and outputs.get("benefit_determination") == "Not determined"


@scorer
def expected_fields_match(outputs, expectations) -> bool:
    """Exact-match fields declared in the golden record's expected_fields."""
    expected = (expectations or {}).get("expected_fields", {})
    return isinstance(outputs, dict) and all(
        outputs.get(key) == value for key, value in expected.items())


DETERMINISTIC_SCORERS = (output_schema_valid, non_decision_boundary, expected_fields_match)


# ---------------------------------------------------------------- LLM judge binding
@dataclass(frozen=True)
class JudgeBinding:
    """How MLflow built-in judges reach the judge model for one profile/alias.

    ``extra_headers`` may hold a gateway credential: it lives only in memory on the
    scorer objects and is never logged (see ``describe``).
    """
    alias: str
    model_uri: str
    environment: Mapping[str, str] = field(default_factory=dict)
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    inference_params: Mapping[str, object] = field(default_factory=dict)

    def describe(self) -> dict[str, str]:
        """Log-safe lineage (no credentials)."""
        return {"judge_model": self.alias, "judge_model_uri": self.model_uri,
                "judge_base_url": self.environment.get("OPENAI_API_BASE", "n/a")}

    def scorer_options(self) -> dict:
        options: dict = {"model": self.model_uri}
        if self.extra_headers:
            options["extra_headers"] = dict(self.extra_headers)
        if self.inference_params:
            options["inference_params"] = dict(self.inference_params)
        return options

    def scorers(self) -> list:
        options = self.scorer_options()
        return [Correctness(**options), Safety(**options), ExpectationsGuidelines(**options)]

    def activated(self) -> ContextManager[None]:
        """Set judge provider variables for the duration of an evaluation, then restore."""
        return scoped_environment(self.environment)


def build_judge_binding(profile: Profile, alias: str | None = None) -> JudgeBinding:
    alias = alias or profile.default_judge_model
    if not alias:
        raise ConfigurationError(f"Profile '{profile.name}' defines no judge model")
    route: ModelRoute = profile.model(alias)
    if route.kind == "mock":
        raise ConfigurationError("The mock model cannot act as an LLM judge")
    if route.kind == "mlflow_uri":
        return JudgeBinding(alias=alias, model_uri=require(route.model_uri, "model_uri"),
                            inference_params=dict(route.params))

    # OpenAI-compatible gateway route. MLflow's openai provider needs a concrete model
    # name, so a Kong judge route must declare the model it pins (Kong rejects any other
    # name, which keeps judge lineage exact).
    model_name = route.request_model()
    if not model_name:
        raise ConfigurationError(
            f"Judge model '{alias}' needs upstream_model (the model the gateway serves)")
    headers = build_auth(profile.gateway.auth).headers()
    authorization = next((v for k, v in headers.items() if k.lower() == "authorization"), None)
    if authorization and authorization.lower().startswith("bearer "):
        api_key = authorization.split(" ", 1)[1]
        headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
    else:
        api_key = _PLACEHOLDER_KEY
    return JudgeBinding(
        alias=alias,
        model_uri=f"openai:/{model_name}",
        environment={"OPENAI_API_BASE": profile.base_url_for(route), "OPENAI_API_KEY": api_key},
        extra_headers=headers,
        inference_params=dict(route.params),
    )


def build_scorers(profile: Profile, mode: str, judge_alias: str | None,
                  logger: logging.Logger) -> tuple[list, JudgeBinding | None, dict[str, str]]:
    """Return scorers, the judge binding (if any) and log-safe manifest entries."""
    if mode == "deterministic":
        return (list(DETERMINISTIC_SCORERS), None,
                {"scorer_version": DETERMINISTIC_SCORER_VERSION,
                 "judge_model": "not used (deterministic)"})
    if mode != "llm-judges":
        raise ConfigurationError(f"Unknown scorer mode '{mode}'")
    binding = build_judge_binding(profile, judge_alias)
    logger.info("judge_binding_resolved", extra={**binding.describe(),
                                                 "environment": profile.environment})
    manifest = {"scorer_version": f"{DETERMINISTIC_SCORER_VERSION}+{JUDGE_SCORER_VERSION}",
                **binding.describe()}
    return list(DETERMINISTIC_SCORERS) + binding.scorers(), binding, manifest

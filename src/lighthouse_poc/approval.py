"""Human review gate: record an approval or rejection bound to exact evaluation evidence.

``lighthouse promote`` approves (or rejects) one registered prompt version on the basis of
one ``lighthouse eval`` run. Approval requires that the run:
- is a finished ``lighthouse eval`` run of exactly this prompt name and version,
- has verdict READY FOR HUMAN REVIEW with every hard gate passing,
- was scored by the evaluation spec as it is now (same digest).

The decision is written as an approval record:
1. ``approvals/<record_id>.json`` on the evaluation run (one file per decision: full history),
2. ``lighthouse_approval_*`` tags on the run and on the prompt version (latest decision;
   Unity Catalog prompt tag keys may not contain dots),
3. on approval only, the alias (default ``champion``) moved to the version; the previous
   target is kept in the record for rollback.

The record is administrative release evidence about a prompt. It is not a claim decision.
"""
from __future__ import annotations

import getpass
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

from . import __version__
from .exceptions import ConfigurationError, PromptRegistryError
from .prompts import ResolvedPrompt
from .review import READY

APPROVE, REJECT = "approved", "rejected"
TAG = "lighthouse_approval_"
MIN_RATIONALE = 20
RESERVED_ALIASES = {"latest"}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class ApprovalRecord:
    record_id: str
    decision: str
    reviewer: str
    rationale: str
    decided_at: str
    recorded_by: str
    prompt_name: str
    prompt_version: str
    alias: str | None
    previous_alias_version: str | None
    run_id: str
    experiment_id: str
    eval_spec: str
    scorer_version: str
    model_service: str
    judge_model: str
    dataset: str
    dataset_version: str
    verdict: str
    combined_score: float | None
    hard_gate_pass_rate: float | None
    evaluated_git_commit: str
    application_version: str = __version__
    checks: list[dict] = field(default_factory=list)

    def summary_tags(self) -> dict[str, str]:
        return {f"{TAG}decision": self.decision, f"{TAG}reviewer": self.reviewer,
                f"{TAG}decided_at": self.decided_at, f"{TAG}record_id": self.record_id,
                f"{TAG}run_id": self.run_id, f"{TAG}prompt_version": self.prompt_version,
                f"{TAG}alias": self.alias or ""}


def evidence_checks(prompt: ResolvedPrompt, run, current_scorer_version: str | None,
                    decision: str) -> list[Check]:
    """Checks binding the decision to the run. Rejections only need the run to match."""
    params, tags, metrics = run.data.params, run.data.tags, run.data.metrics
    checks = [
        Check("registered_version", not prompt.is_draft,
              f"{prompt.name} version {prompt.label}" if not prompt.is_draft
              else f"{prompt.label} is an unregistered draft; register it first"),
        Check("eval_run", tags.get("lighthouse.kind") == "eval",
              "run created by lighthouse eval" if tags.get("lighthouse.kind") == "eval"
              else "run was not created by lighthouse eval"),
        Check("run_finished", run.info.status == "FINISHED", f"run status {run.info.status}"),
        Check("same_prompt", params.get("prompt_name") == prompt.name
              and params.get("prompt_version") == prompt.label,
              f"run evaluated {params.get('prompt_name')} version {params.get('prompt_version')}"),
    ]
    if decision == REJECT:
        return checks
    verdict = tags.get("lighthouse.verdict")
    gate_rate = metrics.get("lighthouse/hard_gate_pass_rate")
    checks += [
        Check("verdict_ready", verdict == READY, f"run verdict {verdict}"),
        Check("hard_gates_all_pass", gate_rate == 1.0, f"hard gate pass rate {gate_rate}"),
        Check("spec_unchanged", current_scorer_version is not None
              and params.get("scorer_version") == current_scorer_version,
              f"run scored with {params.get('scorer_version')}, current {current_scorer_version}"),
    ]
    return checks


class ApprovalService:
    """Writes approval records to MLflow. Clients are injectable for tests."""

    def __init__(self, logger: logging.Logger, registry, tracking_client=None,
                 clock: Callable[[], datetime] | None = None,
                 user: Callable[[], str] | None = None):
        import mlflow
        self._logger, self._registry = logger, registry
        self._client = tracking_client or mlflow.MlflowClient()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._user = user or _local_user

    def get_run(self, run_id: str):
        try:
            return self._client.get_run(run_id)
        except Exception as exc:
            raise ConfigurationError(f"Evaluation run {run_id!r} not found") from exc

    def current_alias_version(self, name: str, alias: str) -> str | None:
        try:
            return str(self._registry.load_alias(name, alias).version)
        except PromptRegistryError:
            return None  # alias not set yet

    def build_record(self, prompt: ResolvedPrompt, run, decision: str, reviewer: str,
                     rationale: str, alias: str | None, checks: list[Check]) -> ApprovalRecord:
        reviewer, rationale = (reviewer or "").strip(), (rationale or "").strip()
        if not reviewer:
            raise ConfigurationError("--reviewer is required (name and role)")
        if len(rationale) < MIN_RATIONALE:
            raise ConfigurationError(
                f"--rationale must explain the decision (at least {MIN_RATIONALE} characters)")
        if alias is not None and (alias in RESERVED_ALIASES or not alias.replace("_", "").replace("-", "").isalnum()):
            raise ConfigurationError(f"Alias {alias!r} is reserved or invalid")
        params, metrics = run.data.params, run.data.metrics
        previous = self.current_alias_version(prompt.name, alias) if alias else None
        return ApprovalRecord(
            record_id=uuid.uuid4().hex, decision=decision, reviewer=reviewer,
            rationale=rationale, decided_at=self._clock().isoformat(timespec="seconds"),
            recorded_by=self._user(), prompt_name=prompt.name, prompt_version=prompt.label,
            alias=alias if decision == APPROVE else None,
            previous_alias_version=previous if decision == APPROVE else None,
            run_id=run.info.run_id, experiment_id=run.info.experiment_id,
            eval_spec=params.get("eval_spec", "unknown"),
            scorer_version=params.get("scorer_version", "unknown"),
            model_service=params.get("model_service", "unknown"),
            judge_model=params.get("judge_model", "unknown"),
            dataset=params.get("dataset", "unknown"),
            dataset_version=params.get("dataset_version", "unknown"),
            verdict=run.data.tags.get("lighthouse.verdict", "unknown"),
            combined_score=metrics.get("lighthouse/combined_score"),
            hard_gate_pass_rate=metrics.get("lighthouse/hard_gate_pass_rate"),
            evaluated_git_commit=params.get("git_commit", "unknown"),
            checks=[asdict(c) for c in checks])

    def write(self, record: ApprovalRecord) -> None:
        """Persist the record, then (approval only) move the alias.

        Order matters: evidence first, pointer last. If the alias move fails, the record
        exists and says which alias was intended; re-running promote completes it.
        """
        try:
            self._client.log_text(record.run_id, json.dumps(asdict(record), indent=2),
                                  f"approvals/{record.record_id}.json")
            for key, value in record.summary_tags().items():
                self._client.set_tag(record.run_id, key, value)
                self._client.set_prompt_version_tag(record.prompt_name, record.prompt_version,
                                                    key, value)
        except Exception as exc:
            self._logger.exception("approval_record_failed", extra={
                "prompt_name": record.prompt_name, "prompt_version": record.prompt_version,
                "run_id": record.run_id, "error_type": type(exc).__name__})
            raise PromptRegistryError("Could not write the approval record") from exc
        self._logger.info("approval_recorded", extra={
            "prompt_name": record.prompt_name, "prompt_version": record.prompt_version,
            "run_id": record.run_id, "kpi_name": f"prompt_{record.decision}", "kpi_value": 1})
        if record.decision != APPROVE or not record.alias:
            return
        try:
            self._client.set_prompt_alias(record.prompt_name, record.alias, int(record.prompt_version))
        except Exception as exc:
            self._logger.exception("approval_alias_failed", extra={
                "prompt_name": record.prompt_name, "prompt_version": record.prompt_version,
                "error_type": type(exc).__name__})
            raise PromptRegistryError(
                f"Approval {record.record_id} recorded, but alias {record.alias!r} was not moved; "
                "re-run promote to complete it") from exc
        self._logger.info("prompt_alias_moved", extra={
            "prompt_name": record.prompt_name, "prompt_version": record.prompt_version,
            "kpi_name": "alias_moves", "kpi_value": 1})


def _local_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # no login name in some job runtimes
        return "unknown"

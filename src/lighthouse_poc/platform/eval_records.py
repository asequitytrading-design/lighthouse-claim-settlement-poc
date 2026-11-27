"""Evaluation results and review decisions in Unity Catalog (ADR-016).

`lighthouse eval` writes one <obs>.eval_run row per MLflow evaluation run and one
<obs>.eval_case_result row per case and criterion (gate, rubric, judge, combined).
`lighthouse promote` writes one <mkt>.release_decision row per recorded decision.
MLflow keeps the full evidence (traces, rationales, artifacts); these rows make it queryable
next to usage and cost (v_eval_trend, dashboards). No prompt text, outputs or rationales here:
scores, verdicts, versions and IDs only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..review import RunReview


@dataclass
class EvalRunRow:
    run_id: str
    experiment_id: str | None
    use_case_id: str | None
    prompt_name: str
    prompt_version: str
    model_key: str
    model_service: str | None
    judge_model_key: str | None
    dataset_name: str
    dataset_version: str
    eval_spec: str
    scorer_version: str
    cases: int
    combined_score: float
    verdict: str
    git_commit: str
    started_at: datetime
    finished_at: datetime


@dataclass
class EvalCaseRow:
    run_id: str
    case_id: str
    criterion: str
    kind: str              # gate | rubric | judge | combined
    value: str | None
    score: float | None
    passed: bool | None
    trace_id: str | None


@dataclass
class ReleaseDecisionRow:
    decision_id: str
    use_case_id: str
    prompt_name: str
    prompt_version: str
    decision: str
    reviewer: str
    rationale: str
    eval_run_id: str
    review_score: float | None
    verdict: str
    model_key: str
    model_service: str | None
    dataset_version: str
    git_commit: str
    decided_at: datetime


def eval_rows(review: RunReview, *, run_id: str, experiment_id: str | None, use_case_id: str | None,
              manifest: dict[str, str], model_service: str | None, started_at: datetime,
              finished_at: datetime) -> dict[str, list]:
    """Rows for eval_run and eval_case_result from a finished review."""
    judge = manifest.get("judge_model")
    run = EvalRunRow(
        run_id=run_id, experiment_id=experiment_id, use_case_id=use_case_id,
        prompt_name=manifest["prompt_name"], prompt_version=manifest["prompt_version"],
        model_key=manifest["model_service"], model_service=model_service,
        judge_model_key=None if judge in (None, "not used") else judge,
        dataset_name=manifest.get("dataset", "unknown"),
        dataset_version=manifest.get("dataset_version", "unknown"),
        eval_spec=manifest.get("eval_spec", "unknown"),
        scorer_version=manifest.get("scorer_version", "unknown"), cases=len(review.cases),
        combined_score=round(review.combined_mean, 4), verdict=review.verdict,
        git_commit=manifest.get("git_commit", "unknown"), started_at=started_at,
        finished_at=finished_at)
    cases: list[EvalCaseRow] = []
    for case in review.cases:
        def row(criterion: str, kind: str, score: float | None, passed: bool | None,
                value: Any = None) -> EvalCaseRow:
            return EvalCaseRow(run_id, case.case_id, criterion, kind,
                               None if value is None else str(value),
                               None if score is None else round(float(score), 4), passed, case.trace_id)
        cases += [row(name, "gate", float(ok), ok, ok) for name, ok in case.gates.items()]
        cases += [row(name, "rubric", score, score >= 1.0) for name, score in case.rubric.items()]
        cases += [row(name, "judge", score, None if score is None else score >= 1.0,
                      "error" if score is None else None) for name, score in case.judges.items()]
        cases.append(row("combined", "combined", case.combined, case.combined >= review.pass_threshold))
    return {"eval_run": [run], "eval_case_result": cases}


def release_rows(record: Any, use_case_id: str, model_service: str | None) -> dict[str, list]:
    """A release_decision row from an approval.ApprovalRecord."""
    return {"release_decision": [ReleaseDecisionRow(
        decision_id=record.record_id, use_case_id=use_case_id, prompt_name=record.prompt_name,
        prompt_version=record.prompt_version, decision=record.decision, reviewer=record.reviewer,
        rationale=record.rationale, eval_run_id=record.run_id, review_score=record.combined_score,
        verdict=record.verdict, model_key=record.model_service, model_service=model_service,
        dataset_version=record.dataset_version, git_commit=record.evaluated_git_commit,
        decided_at=datetime.fromisoformat(record.decided_at).replace(tzinfo=None))]}

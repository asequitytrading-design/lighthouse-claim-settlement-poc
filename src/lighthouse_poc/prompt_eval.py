"""Golden evaluation of any prompt reference with any profile model (``lighthouse eval``).

Adapters turn a dataset record into a model call:
- ``claim_settlement``: the ClaimSettlementAssistant (fixed user message, contract checks).
- ``chat_prompt``: a registered or draft prompt rendered with the record's inputs as
  variables (chat prompts from the MLflow UI Playground, or text prompts), output parsed
  as a JSON object.

Each run is one MLflow evaluation run: per-case gate/rubric/judge assessments on the
traces, the combined review score per trace (``lighthouse_combined_score``), run metrics
``lighthouse/*``, the review as ``lighthouse_review.json`` and the verdict tag.
"""
from __future__ import annotations

import json
import logging
import subprocess
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import __version__
from .agent import ClaimSettlementAssistant, extract_json_object
from .config import PROJECT_ROOT, Profile, scoped_environment
from .exceptions import ConfigurationError, LighthouseError, OutputValidationError
from .prompts import ResolvedPrompt
from .review import RunReview, review_case, summarize
from .rubric import EvalSpec, criterion_scorers, judge_scorers
from .scorers import ERROR_KEY


# ---------------------------------------------------------------- dataset
def load_records(spec: EvalSpec, override: str | None = None) -> tuple[list[dict], str]:
    """Records and a dataset label. ``override``: a JSONL path or ``mlflow:<dataset name>``."""
    source = override or (str(spec.dataset_path) if spec.dataset_path else f"mlflow:{spec.dataset_name}")
    if source.startswith("mlflow:"):
        name = source[len("mlflow:"):]
        try:
            from mlflow.genai import datasets
            found = datasets.search_datasets(filter_string=f"name = '{name}'")
            if not found:
                raise ConfigurationError(f"MLflow dataset {name!r} not found")
            frame = found[0].to_df()
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(f"Cannot load MLflow dataset {name!r}") from exc
        records = [{"inputs": row["inputs"], "expectations": row.get("expectations") or {},
                    "tags": {k: v for k, v in (row.get("tags") or {}).items()
                             if not k.startswith("mlflow.")}}
                   for row in frame.to_dict("records")]
        return records, f"mlflow:{name}"
    from .evaluation import load_golden_records
    return load_golden_records(Path(source)), Path(source).name


def upsert_eval_dataset(name: str, experiment_id: str, records: list[dict], tags: dict | None = None):
    """Create (or reuse) an MLflow evaluation dataset by name and merge records (by inputs)."""
    try:
        from mlflow.genai import datasets
        existing = datasets.search_datasets(experiment_ids=experiment_id,
                                            filter_string=f"name = '{name}'")
        dataset = existing[0] if existing else datasets.create_dataset(
            name=name, experiment_id=experiment_id, tags=tags or {})
        return dataset.merge_records(records)
    except Exception as exc:
        from .exceptions import EvaluationError
        raise EvaluationError(f"Could not load dataset {name}") from exc


def case_id(record: dict, index: int) -> str:
    return str((record.get("tags") or {}).get("case_id")
               or record["inputs"].get("case_id") or f"case-{index + 1:03d}")


def _key(inputs: dict) -> str:
    return json.dumps(inputs, sort_keys=True, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- adapters
def build_runner(spec: EvalSpec, prompt: ResolvedPrompt, model, logger: logging.Logger,
                 registry) -> Callable[[dict], dict]:
    if spec.adapter == "claim_settlement":
        assistant = ClaimSettlementAssistant(registry, model, logger, prompt.name, prompt.label,
                                             __version__, prompt=prompt)
        return lambda inputs: assistant.predict(**inputs)

    variables = prompt.variables

    def run(inputs: dict) -> dict:
        values = {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                  for k, v in inputs.items() if k in variables}
        rendered = prompt.format(**values)
        if prompt.is_chat:
            messages = rendered
        elif variables:
            messages = [{"role": "user", "content": rendered}]
        else:  # instructions only: send the record as the user message
            messages = [{"role": "system", "content": rendered},
                        {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)}]
        completion = model.complete_messages(messages)
        try:
            return extract_json_object(completion.text)
        except Exception as exc:
            logger.error("prompt_output_invalid", extra={
                "prompt_name": prompt.name, "prompt_version": prompt.label,
                "served_model": completion.served_model, "error_type": type(exc).__name__,
                "error_category": "output_validation"})
            raise OutputValidationError("Model output is not a JSON object") from exc
    return run


def check_variables(spec: EvalSpec, prompt: ResolvedPrompt, records: list[dict]) -> None:
    """Fail before any model call when the dataset cannot fill the prompt's variables."""
    if spec.adapter != "chat_prompt":
        return
    for index, record in enumerate(records):
        missing = prompt.variables - set(record["inputs"])
        if missing:
            raise ConfigurationError(
                f"Prompt {prompt.name} {prompt.label} needs variables {sorted(missing)} that "
                f"record {case_id(record, index)} does not provide (inputs: {sorted(record['inputs'])})")


# ---------------------------------------------------------------- run
@dataclass
class EvalRequest:
    spec: EvalSpec
    prompt: ResolvedPrompt
    model_alias: str | None
    records: list[dict]
    dataset_label: str
    use_judges: bool = True
    judge_alias: str | None = None
    run_name: str | None = None


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        import os
        return os.getenv("GIT_COMMIT", "unknown")


def run_eval(profile: Profile, registry, logger: logging.Logger, request: EvalRequest,
             model_factory: Callable | None = None) -> tuple[RunReview, str]:
    """Run one evaluation; returns the combined review and the MLflow run ID."""
    import mlflow

    from .evaluation import MLflowEvaluationService
    from .gateway import ServedModelRecorder, build_chat_model
    from .scorers import build_judge_binding

    spec, prompt = request.spec, request.prompt
    check_variables(spec, prompt, request.records)
    route = profile.model(request.model_alias)
    model = ServedModelRecorder((model_factory or build_chat_model)(profile, route.alias, logger))
    runner = build_runner(spec, prompt, model, logger, registry)
    case_ids = {_key(r["inputs"]): case_id(r, i) for i, r in enumerate(request.records)}

    binding = None
    scorers = criterion_scorers(spec)
    if request.use_judges and spec.judge_kinds():
        binding = build_judge_binding(profile, request.judge_alias)
        scorers += judge_scorers(spec, binding.scorer_options())

    @mlflow.trace(name="evaluate_case")
    def predict_fn(**inputs):
        if mlflow.get_active_trace_id():  # MLflow's pre-check calls this outside a trace
            mlflow.update_current_trace(tags={
                "lighthouse.case_id": case_ids.get(_key(inputs), "unknown"),
                "lighthouse.prompt_version": prompt.label})
        try:
            return runner(inputs)
        except LighthouseError as exc:  # a failed case is evidence, not a reason to abort
            return {ERROR_KEY: type(exc).__name__,
                    "error_category": getattr(exc, "category", "output_validation")}

    manifest = {
        **profile.lineage(route),
        "profile": profile.name, "application_version": __version__, "git_commit": git_commit(),
        "eval_spec": spec.name, "scorer_version": spec.scorer_version,
        "prompt_name": prompt.name, "prompt_version": prompt.label,
        "prompt_reference": prompt.reference, "model_service": route.alias,
        "dataset": request.dataset_label, "dataset_version": spec.dataset_version,
        "dataset_cases": str(len(request.records)),
        "judge_model": binding.alias if binding else "not used",
        **(binding.describe() if binding else {}),
        "rubric_weight": str(spec.scoring.rubric_weight),
        "judge_weight": str(spec.scoring.judge_weight),
        "pass_threshold": str(spec.scoring.pass_threshold),
    }
    run_name = request.run_name or f"{spec.name}-{prompt.label}-{route.alias}"
    with ExitStack() as stack:
        stack.enter_context(scoped_environment(profile.evaluation.environment()))
        if binding:
            stack.enter_context(binding.activated())
        run = stack.enter_context(mlflow.start_run(run_name=run_name))
        mlflow.log_params(manifest)
        mlflow.set_tags({"lighthouse.kind": "eval", "lighthouse.eval_spec": spec.name,
                         "lighthouse.prompt_reference": prompt.reference})
        result = MLflowEvaluationService(logger).run(request.records, predict_fn, scorers, manifest)
        judges_run = spec.judge_kinds() if binding else []
        review = summarize(_case_reviews(result.result_df, spec, judges_run), spec)
        mlflow.log_metrics(review.metrics())
        mlflow.log_dict(review.to_dict(), "lighthouse_review.json")
        mlflow.set_tags({"lighthouse.verdict": review.verdict,
                         "lighthouse.served_models": ",".join(sorted(model.served_models)) or "none"})
        _log_case_scores(review, logger)
    logger.info("prompt_eval_completed", extra={
        "prompt_name": prompt.name, "prompt_version": prompt.label, "model_alias": route.alias,
        "run_id": run.info.run_id, "kpi_name": "combined_score", "kpi_value": review.combined_mean})
    return review, run.info.run_id


def _case_reviews(frame, spec: EvalSpec, judge_kinds: list[str]):
    reviews = []
    for index, row in enumerate(frame.to_dict("records")):
        tags = row.get("tags") or {}
        reviews.append(review_case(row, spec, tags.get("lighthouse.case_id") or f"row-{index + 1}",
                                   judge_kinds))
    return sorted(reviews, key=lambda c: c.case_id)


def _log_case_scores(review: RunReview, logger: logging.Logger) -> None:
    """Attach the combined score to each trace so it shows next to the judges in the UI."""
    import mlflow
    from mlflow.entities import AssessmentSource, AssessmentSourceType
    source = AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="lighthouse-review")
    for case in review.cases:
        if not case.trace_id:
            continue
        try:
            mlflow.log_feedback(trace_id=case.trace_id, name="lighthouse_combined_score",
                                value=round(case.combined, 4), source=source,
                                rationale="; ".join(case.notes) or "all criteria met")
        except Exception:  # the run metrics and review JSON already hold the score
            logger.warning("combined_score_feedback_failed", exc_info=True,
                           extra={"case_id": case.case_id, "trace_id": case.trace_id})

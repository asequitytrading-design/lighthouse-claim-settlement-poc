"""MLflow experiment assets around an evaluation and a review decision (ADR-016).

What each step adds to the experiment, so the Databricks UI shows the full picture:

  link_prompt         the evaluated prompt version on the run and on every trace (Prompts tab,
                      run -> linked prompts); drafts (file:) are not registered, so not linked
  eval_dataset        the golden set as an MLflow evaluation dataset (Datasets tab). On
                      Databricks it is a Unity Catalog table next to the prompt:
                      <catalog>.<schema>.<spec>_golden. The run is evaluated against it when the
                      run uses the whole set; a --limit run uses the records and names the dataset
  register_judges     the spec's LLM judges in the experiment (Scorers tab). Registration only
                      accepts Databricks-served judge models, so the registered copies name the
                      Databricks endpoint of the same model; the evaluation itself keeps calling
                      the judge through Unity Gateway with request tags (ADR-015)
  review_session      a labeling session in the Review App with the run's traces, for examiner /
                      SME review (`lighthouse eval --review-session`)
  human_review        `lighthouse promote` logs the reviewer's decision as a HUMAN assessment on
                      every trace of the evidence run, next to the judges

Each step is capability-checked: when the workspace (e.g. Free Edition) or the local MLflow server
does not offer it, the step reports `skipped` with the reason and the evaluation or decision goes
on. The same code starts working when the capability exists (enterprise workspace).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

DONE, SKIPPED = "done", "skipped"
REVIEW_LABEL_SCHEMA = "lighthouse_release_review"
REVIEW_LABEL_OPTIONS = ["ready", "needs_work", "reject"]
REVIEW_INSTRUCTION = ("Is this output ready for a human claims examiner? It must only summarise "
                      "the supplied facts, list evidence and gaps, and never decide coverage, "
                      "eligibility, fraud or benefit amounts.")


@dataclass(frozen=True)
class AssetResult:
    name: str
    status: str            # done | skipped
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


def dataset_name_for(prompt_name: str, spec_name: str) -> str:
    """<catalog>.<schema>.<spec>_golden next to a Unity Catalog prompt; <spec>_golden otherwise."""
    prefix = prompt_name.rsplit(".", 1)[0] + "." if prompt_name.count(".") == 2 else ""
    return f"{prefix}{spec_name}_golden"


def registrable_judge_model(model_uri: str, request_model: str | None) -> str | None:
    """A judge model URI the experiment accepts for registration (Databricks-served only)."""
    if model_uri.startswith("databricks"):
        return model_uri
    if request_model and request_model.startswith("system.ai."):
        return f"databricks:/databricks-{request_model[len('system.ai.'):]}"
    return None


def _first_line(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0][:240] if text else ''}"


class ExperimentAssets:
    """Capability-checked MLflow calls; never raises (the run's evidence comes first)."""

    def __init__(self, logger: logging.Logger, client: Any = None):
        import mlflow
        self._logger = logger
        self._client = client or mlflow.MlflowClient()

    def _attempt(self, name: str, fn: Callable[[], str]) -> AssetResult:
        try:
            result = AssetResult(name, DONE, fn())
        except _Skip as skip:
            result = AssetResult(name, SKIPPED, str(skip))
        except Exception as exc:          # unavailable on this workspace, or failed: report
            result = AssetResult(name, SKIPPED, f"not available here ({_first_line(exc)})")
            self._logger.warning("experiment_asset_skipped", extra={
                "asset": name, "error_type": type(exc).__name__, "detail": result.detail})
        self._logger.info("experiment_asset", extra={"asset": name, "status": result.status,
                                                     "detail": result.detail})
        return result

    # ------------------------------------------------------------ prompt
    def link_prompt(self, run_id: str, prompt: Any, trace_ids: Iterable[str]) -> AssetResult:
        def run() -> str:
            if prompt.is_draft or not str(prompt.label).isdigit():
                raise _Skip("draft prompt (file:): register it to link it")
            import mlflow
            version = mlflow.genai.load_prompt(f"prompts:/{prompt.name}/{prompt.label}",
                                               link_to_model=False)
            self._client.link_prompt_version_to_run(run_id, version)
            traces = [t for t in trace_ids if t]
            for trace_id in traces:
                self._client.link_prompt_versions_to_trace([version], trace_id)
            return f"{prompt.name} v{prompt.label} -> run and {len(traces)} traces"
        return self._attempt("prompt_link", run)

    # ------------------------------------------------------------ dataset
    def eval_dataset(self, name: str, experiment_id: str, records: list[dict],
                     tags: dict[str, str]) -> tuple[Any, AssetResult]:
        """Create or update the evaluation dataset; returns it (or None) and the result."""
        holder: dict[str, Any] = {}

        def run() -> str:
            from mlflow.genai import datasets
            # Databricks' managed-evals API rejects filter_string: match the name here
            found = [d for d in datasets.search_datasets(experiment_ids=[experiment_id])
                     if d.name == name]
            if not found:                # a UC dataset may exist already, created by another experiment
                try:
                    found = [datasets.get_dataset(name=name)]
                except Exception:
                    found = []
            if found:                    # re-fetch by name: search results cannot be updated on Databricks
                dataset = datasets.get_dataset(name=name)
            else:
                try:
                    dataset = datasets.create_dataset(name=name, experiment_id=experiment_id,
                                                      tags=tags or None)
                except NotImplementedError:      # Databricks: tags are managed in Unity Catalog
                    dataset = datasets.create_dataset(name=name, experiment_id=experiment_id)
            dataset.merge_records([{"inputs": r["inputs"], "expectations": r.get("expectations") or {}}
                                   for r in records])
            holder["dataset"] = dataset
            return f"{name}: {len(records)} records merged ({'updated' if found else 'created'})"
        result = self._attempt("eval_dataset", run)
        return holder.get("dataset"), result

    # ------------------------------------------------------------ judges
    def register_judges(self, experiment_id: str, judges: list, model: str | None) -> AssetResult:
        def run() -> str:
            if not judges:
                raise _Skip("no LLM judges in this run")
            if not model:
                raise _Skip("the judge model is not Databricks-served: registration needs databricks:/")
            from mlflow.genai.scorers import delete_scorer, list_scorers
            existing = {s.name for s in list_scorers(experiment_id=experiment_id)}
            for judge in judges:
                if judge.name in existing:            # replace: the spec may have changed
                    delete_scorer(name=judge.name, experiment_id=experiment_id)
                judge.register(name=judge.name, experiment_id=experiment_id)
            return f"{', '.join(j.name for j in judges)} registered with {model}"
        return self._attempt("judge_registration", run)

    # ------------------------------------------------------------ review
    def review_session(self, name: str, trace_ids: list[str], reviewers: list[str]) -> AssetResult:
        def run() -> str:
            if not trace_ids:
                raise _Skip("no traces to review")
            from mlflow.genai import labeling
            schema = self._review_schema()
            session = labeling.create_labeling_session(
                name=name[:60], assigned_users=reviewers or None, label_schemas=[schema])
            import mlflow
            traces = [mlflow.get_trace(t) for t in trace_ids]
            session.add_traces([t for t in traces if t is not None])
            return f"{session.name} ({len(trace_ids)} traces, schema {schema}): {getattr(session, 'url', '')}"
        return self._attempt("review_session", run)

    def _review_schema(self) -> str:
        """Our categorical schema when label schemas can be created, else the built-in one."""
        try:
            from mlflow.genai import label_schemas as ls
            try:
                ls.get_label_schema(REVIEW_LABEL_SCHEMA)
            except Exception:
                ls.create_label_schema(
                    name=REVIEW_LABEL_SCHEMA, type="feedback", title="Ready for the examiner?",
                    input=ls.InputCategorical(options=REVIEW_LABEL_OPTIONS),
                    instruction=REVIEW_INSTRUCTION, enable_comment=True)
            return REVIEW_LABEL_SCHEMA
        except Exception as exc:
            self._logger.warning("review_label_schema_fallback", extra={"detail": _first_line(exc)})
            return "guidelines"

    def human_review(self, trace_ids: list[str], decision: str, reviewer: str,
                     rationale: str, record_id: str) -> AssetResult:
        def run() -> str:
            if not trace_ids:
                raise _Skip("the evidence run has no traces")
            import mlflow
            from mlflow.entities import AssessmentSource, AssessmentSourceType
            source = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=reviewer)
            for trace_id in trace_ids:
                mlflow.log_feedback(trace_id=trace_id, name="lighthouse_release_review",
                                    value=decision, source=source, rationale=rationale,
                                    metadata={"approval_record_id": record_id})
            return f"{decision} by {reviewer} on {len(trace_ids)} traces"
        return self._attempt("human_review", run)


class _Skip(Exception):
    """A step that does not apply (not an error)."""


def run_trace_ids(experiment_id: str, run_id: str) -> list[str]:
    """Trace IDs of an evaluation run (one per case)."""
    import mlflow
    try:
        traces = mlflow.search_traces(experiment_ids=[experiment_id], run_id=run_id,
                                      return_type="list", max_results=500)
    except Exception:
        return []
    return [t.info.trace_id for t in traces]

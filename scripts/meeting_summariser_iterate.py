"""Iterate on the local `meeting-summariser` prompt and its judges (local experiment only).

Every prompt version is scored the same way and persisted in MLflow experiment 10
(`/lighthouse-ui-sample`), so versions can be compared on the Evaluation Runs page:

  dataset   build the evaluation dataset `meeting_summariser_transcripts` from Playground traces
            (inputs = the transcript message only; stored in the local MLflow database).
  run       one evaluation run for a prompt version: notes from the app model (Kong route
            claims-medium, temperature 0, the prompt and the transcript as two user messages, as
            the Playground sends them), scored by --judges (default: the instant code judge).
            The run records the prompt version (linked), app model, judge endpoint and git commit.
  judge     add judges to an existing run's traces, a few at a time, so results show up per
            batch instead of after one long pass (one local model serves every call anyway).
  compare   print each iteration run's judge pass rates side by side.

The transcripts are the user's real meetings: they stay in the local MLflow only. The script
refuses a non-local tracking URI and is excluded from the Databricks bundle sync; it never
prints transcript or notes content.

  MLFLOW_TRACKING_URI=http://localhost:5000 .venv/bin/python scripts/meeting_summariser_iterate.py dataset tr-a tr-b
  MLFLOW_TRACKING_URI=http://localhost:5000 .venv/bin/python scripts/meeting_summariser_iterate.py run --prompt-version 2
  MLFLOW_TRACKING_URI=http://localhost:5000 .venv/bin/python scripts/meeting_summariser_iterate.py judge <run_id> summary_faithful summary_action_items_owned
  MLFLOW_TRACKING_URI=http://localhost:5000 .venv/bin/python scripts/meeting_summariser_iterate.py compare
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

try:
    PROJECT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT = Path(sys.argv[0]).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lighthouse_poc.config import load_dotenv, load_profile, scoped_environment  # noqa: E402
from lighthouse_poc.logging_config import configure_logging  # noqa: E402

EXPERIMENT_ID = "10"
PROMPT_NAME = "meeting-summariser"
DATASET_NAME = "meeting_summariser_transcripts"
APP_MODEL = "claims-medium"
JUDGE_ENDPOINT = "gateway:/kong-claims-judge"
ITERATION_TAG = "meeting.iteration"
DEFAULT_RUN_JUDGES = ["summary_json_format"]   # deterministic, instant
CODE_JUDGES = {"summary_json_format"}           # deterministic: one vote is enough
JUDGES = [  # registered in experiment 10; each pass uses the judge's latest version
    "summary_faithful",
    "summary_action_items_owned",
    "summary_decisions_complete",
    "summary_no_invented_owners_or_dates",
    "summary_open_questions_captured",
    "summary_structured_and_concise",
    "summary_json_format",
]
LOCAL_URIS = ("http://localhost", "http://127.0.0.1")
OLLAMA_URL = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434")
OLLAMA_MODELS = ["qwen3.5-9b-32k:latest"]   # app model and judge model (kong-claims-judge -> 9b)


def unload_models() -> None:
    """Unload the local model after a pass: long sessions grow Ollama's memory, a reload resets it."""
    import requests
    for name in OLLAMA_MODELS:
        try:
            requests.post(f"{OLLAMA_URL}/api/generate", json={"model": name, "keep_alive": 0},
                          timeout=60).raise_for_status()
            print(f"Unloaded {name} from Ollama")
        except requests.RequestException as exc:   # the scores are logged; only the clean-up failed
            logging.getLogger("meeting_summariser_iterate").warning(
                "ollama_unload_failed", extra={"model": name, "error_type": type(exc).__name__})


def require_local(uri: str) -> None:
    if not uri.startswith(LOCAL_URIS):
        raise SystemExit(f"Refusing: tracking URI {uri!r} is not the local MLflow "
                         "(these transcripts stay on this machine)")


def transcript_from_trace(trace) -> str:
    """The Playground sends [prompt, transcript] as user messages; the transcript is the last."""
    request = trace.data.request
    request = json.loads(request) if isinstance(request, str) else request
    messages = [m for m in request.get("messages", []) if m.get("role") == "user"]
    if len(messages) < 2:
        raise SystemExit(f"Trace {trace.info.trace_id} is not a prompt + transcript Playground call")
    return messages[-1]["content"]


def prompt_messages(template, transcript: str) -> list[dict]:
    if isinstance(template, str):
        return [{"role": "user", "content": template}, {"role": "user", "content": transcript}]
    return [*template, {"role": "user", "content": transcript}]   # chat prompt


def judge_versions(base_uri: str) -> dict[str, int]:
    import requests
    listing = requests.get(f"{base_uri.rstrip('/')}/ajax-api/3.0/mlflow/scorers/list",
                           params={"experiment_id": EXPERIMENT_ID}, timeout=30)
    listing.raise_for_status()
    versions: dict[str, int] = {}
    for s in listing.json().get("scorers", []):
        versions[s["scorer_name"]] = max(versions.get(s["scorer_name"], 0), int(s["scorer_version"]))
    return versions


def cmd_dataset(args) -> int:
    import mlflow

    from lighthouse_poc.prompt_eval import upsert_eval_dataset

    records = []
    for index, trace_id in enumerate(args.trace_ids, start=1):
        transcript = transcript_from_trace(mlflow.get_trace(trace_id))
        records.append({"inputs": {"transcript": transcript},
                        "tags": {"case_id": f"meeting-{index}", "source_trace": trace_id}})
        print(f"meeting-{index}: from {trace_id} ({len(transcript)} chars)")
    upsert_eval_dataset(DATASET_NAME, EXPERIMENT_ID, records,
                        tags={"local_only": "true", "contains": "real meeting transcripts"})
    print(f"Dataset {DATASET_NAME} in experiment {EXPERIMENT_ID}: {len(records)} records")
    return 0


def load_records() -> list[dict]:
    from mlflow.genai import datasets
    found = datasets.search_datasets(experiment_ids=EXPERIMENT_ID,
                                     filter_string=f"name = '{DATASET_NAME}'")
    if not found:
        raise SystemExit(f"Dataset {DATASET_NAME} not found; run the dataset command first")
    frame = found[0].to_df()
    records = [{"inputs": row["inputs"], "tags": row.get("tags") or {}}
               for row in frame.to_dict("records")]
    return sorted(records, key=lambda r: r["tags"].get("case_id", ""))


def cmd_run(args) -> int:
    import mlflow
    from mlflow.genai.scorers import get_scorer

    from lighthouse_poc.gateway import build_chat_model
    from lighthouse_poc.prompt_eval import git_commit

    logger = logging.getLogger("meeting_summariser_iterate")
    profile = load_profile("local")
    model = build_chat_model(profile, APP_MODEL, logger)
    records = load_records()
    case_ids = {r["inputs"]["transcript"]: r["tags"].get("case_id", "unknown") for r in records}
    versions = judge_versions(mlflow.get_tracking_uri())
    missing = [j for j in JUDGES if j not in versions]
    if missing:
        raise SystemExit(f"Judges not registered in experiment {EXPERIMENT_ID}: {missing}")

    label = f"v{args.prompt_version}"
    run_name = args.run_name or f"meeting-summariser-{label}"
    mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
    judges = args.judges or DEFAULT_RUN_JUDGES
    with scoped_environment(EVAL_ENV), mlflow.start_run(run_name=run_name) as run:
        prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{args.prompt_version}")  # links it

        @mlflow.trace(name="summarise_meeting", span_type="CHAIN")
        def predict_fn(transcript: str) -> str:
            if mlflow.get_active_trace_id():
                mlflow.update_current_trace(tags={"meeting.case_id": case_ids.get(transcript, "unknown"),
                                                  "meeting.prompt_version": label})
            return model.complete_messages(prompt_messages(prompt.template, transcript)).text

        mlflow.log_params({
            "prompt_name": PROMPT_NAME, "prompt_version": str(prompt.version),
            "app_model": APP_MODEL, "judge_endpoint": JUDGE_ENDPOINT, "dataset": DATASET_NAME,
            "dataset_cases": str(len(records)), "git_commit": git_commit()})
        mlflow.set_tags({ITERATION_TAG: label, "meeting.notes": args.notes or "",
                         "local_only": "true"})
        print(f"Run {run.info.run_id} ({run_name}): notes for {len(records)} meetings, judges {judges}")
        log_judge_versions(judges, versions)
        mlflow.genai.evaluate(data=records, predict_fn=predict_fn,
                              scorers=[get_scorer(name=j, experiment_id=EXPERIMENT_ID) for j in judges])
        log_summary(run.info.run_id)
    return 0


EVAL_ENV = {
    "MLFLOW_GENAI_EVAL_MAX_WORKERS": "1",           # one Ollama model serves app and judges
    "MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS": "1",
    "MLFLOW_GENAI_EVAL_LLM_TIMEOUT": "900",          # long transcripts on a local 9b model
    "MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT": "900",        # per-scorer limit (default 300 s)
    "MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS": "true",   # load the registered code judge
}


def log_judge_versions(judges: list[str], versions: dict[str, int]) -> None:
    import mlflow
    mlflow.set_tags({f"judge.{j}": f"v{versions[j]}" for j in judges})   # tags: re-runs may bump


def run_traces(run_id: str):
    import mlflow
    return mlflow.search_traces(locations=[EXPERIMENT_ID], run_id=run_id, return_type="list")


VERSION_KEY = "mlflow.assessment.scorerVersion"


def tally(assessments, versions: dict[str, int], votes: int) -> dict[str, dict]:
    """Majority verdict per judge from its current version's latest `votes` valid assessments.

    A judge error is not a vote. A tie counts as a fail (the judge could not agree on a pass).
    Returns {judge: {"verdict": bool | None, "votes": [bool, ...], "errors": int}}.
    """
    out: dict[str, dict] = {}
    for a in sorted(assessments, key=lambda a: a.create_time_ms or 0):
        if a.name not in JUDGES or a.feedback is None:
            continue
        if str((a.metadata or {}).get(VERSION_KEY, versions.get(a.name))) != str(versions.get(a.name)):
            continue   # an older judge version: not comparable
        entry = out.setdefault(a.name, {"votes": [], "errors": 0})
        if a.feedback.error is not None:
            entry["errors"] += 1
        else:
            entry["votes"].append(_is_pass(a.feedback.value))
    for name, entry in out.items():
        entry["votes"] = entry["votes"][-votes:]
        cast = entry["votes"]
        entry["verdict"] = (sum(cast) * 2 > len(cast)) if cast else None
    return out


def log_summary(run_id: str, votes: int = 1) -> None:
    """Majority pass rates over the judges assessed so far on the run's traces."""
    import mlflow
    versions = judge_versions(mlflow.get_tracking_uri())
    tallies = [tally(t.info.assessments, versions, votes) for t in run_traces(run_id)]
    rows = [{j: e["verdict"] for j, e in t.items()} for t in tallies]
    judged = sorted({name for row in rows for name in row})
    if not rows or not judged:
        return
    unresolved = sum(v is None for row in rows for v in row.values())
    multi = [e["votes"] for t in tallies for j, e in t.items() if len(e["votes"]) > 1]
    metrics = {"judge_errors_unresolved": unresolved, "judges_assessed": len(judged), "votes": votes}
    if multi:
        metrics["judge_vote_agreement"] = sum(len(set(v)) == 1 for v in multi) / len(multi)
    for j in judged:
        values = [row[j] for row in rows if row.get(j) is not None]
        if values:
            metrics[f"pass.{j}"] = sum(values) / len(values)
    checks = [v for row in rows for v in row.values() if v is not None]
    if checks:
        metrics["judge_checks_pass_rate"] = sum(checks) / len(checks)
    complete = [row for row in rows if all(row.get(j) is not None for j in JUDGES)]
    if len(judged) == len(JUDGES) and complete:
        metrics["all_judges_pass_rate"] = sum(all(row[j] for j in JUDGES) for row in complete) / len(complete)
    mlflow.log_metrics(metrics)
    for j in judged:
        valid = [t[j] for t in tallies if j in t and t[j]["verdict"] is not None]
        rate = f"{metrics[f'pass.{j}']:.0%}" if f"pass.{j}" in metrics else "-"
        shown = " ".join("".join("Y" if v else "n" for v in e["votes"]) for e in valid)
        print(f"  {j}: {rate} of {len(valid)} judged   votes: {shown}")
    if checks:
        print(f"  checks pass: {metrics['judge_checks_pass_rate']:.0%}; unresolved errors: {unresolved}"
              + (f"; vote agreement {metrics['judge_vote_agreement']:.0%}" if multi else ""))
    if "all_judges_pass_rate" in metrics:
        print(f"  all judges pass: {metrics['all_judges_pass_rate']:.0%} of {len(complete)} fully judged meetings")


def missing_votes(traces, judges: list[str], versions: dict[str, int], votes: int) -> dict[str, list]:
    """Per judge, the traces that still have fewer than `votes` valid votes from its current version."""
    need: dict[str, list] = {}
    for trace in traces:
        t = tally(trace.info.assessments, versions, votes)
        for j in judges:
            wanted = 1 if j in CODE_JUDGES else votes
            if len(t.get(j, {}).get("votes", [])) < wanted:
                need.setdefault(j, []).append(trace)
    return need


def errored(run_id: str) -> list[str]:
    """Judges with an unresolved verdict (only errors) on some trace of the run."""
    import mlflow
    versions = judge_versions(mlflow.get_tracking_uri())
    names = set()
    for trace in run_traces(run_id):
        names |= {j for j, e in tally(trace.info.assessments, versions, 1).items() if e["verdict"] is None}
    return sorted(names)


def cmd_judge(args) -> int:
    import mlflow
    from mlflow.genai.scorers import get_scorer

    unknown = [j for j in args.judges if j not in JUDGES]
    if unknown:
        raise SystemExit(f"Unknown judges {unknown}; choose from {JUDGES}")
    if not run_traces(args.run_id):
        raise SystemExit(f"Run {args.run_id} has no traces")
    versions = judge_versions(mlflow.get_tracking_uri())
    scorers = {j: get_scorer(name=j, experiment_id=EXPERIMENT_ID) for j in args.judges}
    with scoped_environment(EVAL_ENV), mlflow.start_run(run_id=args.run_id):
        log_judge_versions(args.judges, versions)
        print(f"Run {args.run_id}: judges {args.judges}, {args.votes} vote(s) each")
        # Each round scores only the traces still short of votes; errors get retried, with a cap.
        for _ in range(args.votes + 2):
            need = missing_votes(run_traces(args.run_id), args.judges, versions, args.votes)
            if not need:
                break
            for judge, traces in need.items():
                mlflow.genai.evaluate(data=traces, scorers=[scorers[judge]])
        log_summary(args.run_id, args.votes)
    return 0


def _is_pass(value) -> bool:
    return value is True or str(value).strip().lower() in {"yes", "true", "pass"}


def cmd_summary(args) -> int:
    import mlflow
    with mlflow.start_run(run_id=args.run_id):
        log_summary(args.run_id, args.votes)
    left = errored(args.run_id)
    if left:
        print(f"  judges with errors to re-run: {' '.join(left)}")
    return 0


def cmd_compare(args) -> int:
    import mlflow
    runs = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID],
                              filter_string=f"tags.`{ITERATION_TAG}` != '' and attributes.status != 'KILLED'",
                              order_by=["attributes.start_time ASC"])
    if runs.empty:
        print("No iteration runs yet")
        return 0
    metric_cols = ["metrics.votes", "metrics.all_judges_pass_rate", "metrics.judge_checks_pass_rate",
                   "metrics.judge_vote_agreement",
                   *[f"metrics.pass.{j}" for j in JUDGES]]
    shown = [c for c in metric_cols if c in runs.columns]
    print("run".ljust(30), " ".join(c.split(".", 1)[1].replace("pass.", "").replace("summary_", "")[:14].rjust(14)
                                   for c in shown))
    for _, row in runs.iterrows():
        name = f"{row['tags.mlflow.runName']} [{row.get(f'tags.{ITERATION_TAG}', '')}]"
        print(name[:30].ljust(30), " ".join(
            ("-" if row[c] != row[c] else f"{row[c]:.2f}").rjust(14) for c in shown))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ds = sub.add_parser("dataset", help="build the dataset from Playground traces")
    ds.add_argument("trace_ids", nargs="+")
    rn = sub.add_parser("run", help="evaluate one prompt version with all judges")
    rn.add_argument("--prompt-version", required=True, type=int)
    rn.add_argument("--run-name")
    rn.add_argument("--notes", help="what changed in this iteration (stored as a run tag)")
    rn.add_argument("--judges", nargs="+", help=f"judges for this pass (default {DEFAULT_RUN_JUDGES})")
    jd = sub.add_parser("judge", help="add judges to an existing run")
    jd.add_argument("run_id")
    jd.add_argument("judges", nargs="+")
    jd.add_argument("--votes", type=int, default=1, help="valid votes per LLM judge and meeting (majority)")
    for command in (rn, jd):
        command.add_argument("--keep-loaded", action="store_true",
                             help="leave the model loaded in Ollama after the pass (default: unload)")
    sub.add_parser("compare", help="compare iteration runs")
    sm = sub.add_parser("summary", help="recompute a run's pass rates")
    sm.add_argument("run_id")
    sm.add_argument("--votes", type=int, default=1)
    args = parser.parse_args()

    load_dotenv()
    configure_logging()
    import mlflow
    require_local(mlflow.get_tracking_uri())
    status = {"dataset": cmd_dataset, "run": cmd_run, "judge": cmd_judge,
              "compare": cmd_compare, "summary": cmd_summary}[args.command](args)
    if args.command in ("run", "judge") and not args.keep_loaded:
        unload_models()
    return status


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    sys.exit(main())

"""Run the synthetic golden evaluation as a Python script or Databricks job task.

The environment profile (config/profiles/<name>.yaml) decides tracking server, prompt
namespace, gateway, model routing, judge routing and evaluation concurrency. Examples:
  python scripts/evaluate.py --profile local --prompt-version 1 --model mock
  python scripts/evaluate.py --profile local --prompt-version 1 --model claims-medium
  python scripts/evaluate.py --profile local --prompt-version 1 --model claims-medium \\
      --scorer-mode llm-judges                      # + Correctness/Safety/Guidelines
  python scripts/evaluate.py --profile databricks --prompt-version 1 --model claims-medium \\
      --scorer-mode llm-judges --judge-model databricks-managed-judge
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

try:
    PROJECT = Path(__file__).resolve().parents[1]
except NameError:  # some job runtimes execute scripts without __file__
    PROJECT = Path(sys.argv[0]).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import mlflow  # noqa: E402

from lighthouse_poc import __version__  # noqa: E402
from lighthouse_poc.agent import ClaimSettlementAssistant  # noqa: E402
from lighthouse_poc.config import (apply_env_overrides, load_dotenv,  # noqa: E402
                                   load_profile, require, scoped_environment)
from lighthouse_poc.tracking import activate_experiment  # noqa: E402
from lighthouse_poc.evaluation import MLflowEvaluationService, load_golden_records  # noqa: E402
from lighthouse_poc.exceptions import LighthouseError  # noqa: E402
from lighthouse_poc.gateway import ServedModelRecorder, build_chat_model  # noqa: E402
from lighthouse_poc.platform.usage_recorder import tracked_run  # noqa: E402
from lighthouse_poc.logging_config import configure_logging  # noqa: E402
from lighthouse_poc.prompt_registry import MLflowPromptRegistry  # noqa: E402
from lighthouse_poc.prompts import PromptResolver  # noqa: E402
from lighthouse_poc.runtime import load_runtime_profile  # noqa: E402
from lighthouse_poc.scorers import ERROR_KEY, build_scorers  # noqa: E402

DATASET_VERSION = "1.1.0"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="Profile name or YAML path (default: $LIGHTHOUSE_PROFILE or local)")
    parser.add_argument("--prompt-name", help="Override the profile's prompt name")
    parser.add_argument("--prompt-version", required=True,
                        help="Prompt reference: <version> | @<alias> (e.g. @champion) | latest")
    parser.add_argument("--model", help="Logical app model from the profile (default: profile app_model)")
    parser.add_argument("--scorer-mode", choices=("deterministic", "llm-judges"), default="deterministic",
                        help="llm-judges = deterministic gates + Correctness, Safety, ExpectationsGuidelines")
    parser.add_argument("--judge-model", help="Logical judge model from the profile (default: profile judge_model)")
    parser.add_argument("--experiment", help="Override the profile's MLflow experiment")
    parser.add_argument("--application-version", default=__version__)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases (smoke runs)")
    parser.add_argument("--env", action="append", metavar="KEY=VALUE",
                        help="Set a non-secret profile input (repeatable), e.g. from a job")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.getenv("GIT_COMMIT", "unknown")


def main() -> int:
    args = parse_args()
    logger: logging.Logger = configure_logging()
    load_dotenv(PROJECT / ".env")
    try:
        apply_env_overrides(args.env)
        profile = load_runtime_profile(args.profile, logger)
        route = profile.model(args.model)
        mlflow.set_tracking_uri(require(profile.tracking_uri, "mlflow.tracking_uri"))
        if profile.registry_uri:
            mlflow.set_registry_uri(profile.registry_uri)
        activate_experiment(profile, logger, args.experiment)   # traces in Unity Catalog (ADR-017)
        prompt_name = args.prompt_name or require(profile.prompt_name, "registry.prompt_name")

        records = load_golden_records(PROJECT / "data" / "golden_eval_dataset.jsonl")
        records = records[: args.limit] if args.limit and args.limit > 0 else records

        model = ServedModelRecorder(build_chat_model(profile, route.alias, logger))
        registry = MLflowPromptRegistry(logger)
        prompt = PromptResolver(registry, prompt_name).resolve(args.prompt_version)
        args.prompt_version = prompt.label  # lineage records the resolved version number
        assistant = ClaimSettlementAssistant(registry, model, logger, prompt_name, prompt.label,
                                             args.application_version, prompt=prompt)
        scorers, judge, scorer_manifest = build_scorers(
            profile, args.scorer_mode, args.judge_model, logger)

        # Traced so the root span is this function: MLflow scores the root span output,
        # and the assistant's own span records None when it raises.
        @mlflow.trace(name="evaluate_case")
        def predict_fn(case_id, product_segment, claim_narrative,
                       submitted_documents, policy_or_benefit_context):
            # A failed case is evidence, not a reason to abort the run: return a marker
            # the deterministic gates score as failed. The assistant already logged it.
            try:
                return assistant.predict(case_id, product_segment, claim_narrative,
                                         submitted_documents, policy_or_benefit_context)
            except LighthouseError as exc:
                return {ERROR_KEY: type(exc).__name__,
                        "error_category": getattr(exc, "category", "output_validation"),
                        "claim_id": case_id}

        manifest = {
            **profile.lineage(route),
            "profile": profile.name,
            "application_version": args.application_version,
            "git_commit": git_commit(),
            "prompt_name": prompt_name,
            "prompt_version": args.prompt_version,
            "model_service": route.alias,
            "dataset_version": args.dataset_version,
            "dataset_cases": str(len(records)),
            **scorer_manifest,
        }
        with ExitStack() as stack:
            # every model call is tagged and recorded (ADR-013); judges get the run's tags
            tracker = stack.enter_context(tracked_run(
                profile, logger, "eval", prompt_name=prompt_name, prompt_version=args.prompt_version,
                app_version=args.application_version, git_commit=manifest["git_commit"]))
            stack.enter_context(scoped_environment(profile.evaluation.environment()))
            if judge:
                stack.enter_context(judge.activated())
            run = stack.enter_context(mlflow.start_run(
                run_name=f"{route.alias}-p{args.prompt_version}-{args.scorer_mode}"))
            tracker.update(run_id=run.info.run_id)
            if judge:   # rebuild the judge scorers (build_scorers puts them last) with the run's tags
                judge_count = len(judge.scorers())
                scorers = scorers[:-judge_count] + judge.scorers(tracker.judge_tags(judge.alias))
            mlflow.log_params(manifest)
            result = MLflowEvaluationService(logger).run(records, predict_fn, scorers, manifest)
            mlflow.set_tag("lighthouse.served_models",
                           ",".join(sorted(model.served_models)) or "none")
        print(json.dumps({"run_id": run.info.run_id, "metrics": result.metrics,
                          "served_models": sorted(model.served_models)}, indent=2, default=str))
        print(f"Scorer mode: {args.scorer_mode}. Judge feedback is review evidence; inspect "
              "cases and traces, then use the separate human review gate.")
        return 0
    except LighthouseError as exc:
        logger.error("evaluation_script_failed", exc_info=True, extra={
            "error_type": type(exc).__name__,
            "error_category": getattr(exc, "category", "lighthouse")})
        return 1


if __name__ == "__main__":
    # Exit only on failure: Databricks serverless runs scripts in an IPython runner where
    # even sys.exit(0) raises SystemExit and marks the task as failed.
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)

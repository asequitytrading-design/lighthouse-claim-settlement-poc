"""Run the synthetic golden evaluation as a Python script or Databricks job task.

The environment profile (config/profiles/<name>.yaml) decides tracking server, prompt
namespace, gateway and model routing. Examples:
  python scripts/evaluate.py --profile local --prompt-version 1 --model mock
  python scripts/evaluate.py --profile local --prompt-version 1 --model claims-medium
  python scripts/evaluate.py --profile databricks --prompt-version 1 --model claims-medium
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import mlflow  # noqa: E402
from mlflow.genai.scorers import Correctness, ExpectationsGuidelines, Safety, scorer  # noqa: E402

from lighthouse_poc import __version__  # noqa: E402
from lighthouse_poc.agent import ClaimSettlementAssistant  # noqa: E402
from lighthouse_poc.config import load_dotenv, load_profile, require  # noqa: E402
from lighthouse_poc.evaluation import MLflowEvaluationService  # noqa: E402
from lighthouse_poc.exceptions import LighthouseError  # noqa: E402
from lighthouse_poc.gateway import ServedModelRecorder, build_chat_model  # noqa: E402
from lighthouse_poc.logging_config import configure_logging  # noqa: E402
from lighthouse_poc.prompt_registry import MLflowPromptRegistry  # noqa: E402


@scorer
def output_schema_valid(outputs):
    return isinstance(outputs, dict) and ClaimSettlementAssistant.REQUIRED.issubset(outputs)


@scorer
def non_decision_boundary(outputs):
    return isinstance(outputs, dict) and outputs.get("benefit_determination") == "Not determined"


@scorer
def expected_segment_matches(outputs, expectations):
    expected = (expectations or {}).get("expected_response", {})
    return isinstance(outputs, dict) and all(
        outputs.get(key) == value
        for key, value in expected.items()
        if key in {"product_segment", "benefit_determination"}
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="Profile name or YAML path (default: $LIGHTHOUSE_PROFILE or local)")
    parser.add_argument("--prompt-name", help="Override the profile's prompt name")
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--model", help="Logical model name from the profile (default: profile app_model)")
    parser.add_argument("--scorer-mode", choices=("deterministic", "llm-judges"), default="deterministic")
    parser.add_argument("--experiment", help="Override the profile's MLflow experiment")
    parser.add_argument("--application-version", default=__version__)
    parser.add_argument("--dataset-version", default="1.0.0")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases (smoke runs)")
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
    load_dotenv()
    try:
        profile = load_profile(args.profile)
        route = profile.model(args.model)
        mlflow.set_tracking_uri(require(profile.tracking_uri, "mlflow.tracking_uri"))
        mlflow.set_experiment(args.experiment or require(profile.experiment, "mlflow.experiment"))
        prompt_name = args.prompt_name or require(profile.prompt_name, "registry.prompt_name")

        with (PROJECT / "data" / "golden_eval_dataset.jsonl").open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        records = records[: args.limit] if args.limit else records

        model = ServedModelRecorder(build_chat_model(profile, route.alias, logger))
        assistant = ClaimSettlementAssistant(MLflowPromptRegistry(logger), model, logger,
                                             prompt_name, args.prompt_version,
                                             args.application_version)

        def predict_fn(case_id, product_segment, claim_narrative,
                       submitted_documents, policy_or_benefit_context):
            return assistant.predict(case_id, product_segment, claim_narrative,
                                     submitted_documents, policy_or_benefit_context)

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
        }
        if args.scorer_mode == "llm-judges":
            judge_model = os.getenv("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL")
            judge_options = {"model": judge_model} if judge_model else {}
            scorers = [Correctness(**judge_options), Safety(**judge_options),
                       ExpectationsGuidelines(**judge_options)]
            manifest["scorer_version"] = "correctness-safety-guidelines-v1"
            manifest["judge_model"] = judge_model or "mlflow default"
        else:
            scorers = [output_schema_valid, non_decision_boundary, expected_segment_matches]
            manifest["scorer_version"] = "deterministic-v1"
            manifest["judge_model"] = "not used (deterministic)"

        with mlflow.start_run(run_name=f"{route.alias}-p{args.prompt_version}") as run:
            mlflow.log_params(manifest)
            result = MLflowEvaluationService(logger).run(records, predict_fn, scorers, manifest)
            mlflow.set_tag("lighthouse.served_models",
                           ",".join(sorted(model.served_models)) or "none")
        print(json.dumps({"run_id": run.info.run_id, "metrics": result.metrics,
                          "served_models": sorted(model.served_models)}, indent=2, default=str))
        print(f"Scorer mode: {args.scorer_mode}. Inspect cases and traces, "
              "then use the separate human review gate.")
        return 0
    except LighthouseError as exc:
        logger.error("evaluation_script_failed", exc_info=True, extra={
            "error_type": type(exc).__name__,
            "error_category": getattr(exc, "category", "lighthouse")})
        return 1


if __name__ == "__main__":
    sys.exit(main())

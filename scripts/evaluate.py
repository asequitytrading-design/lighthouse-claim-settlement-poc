"""Run the synthetic golden evaluation as a Python script or Databricks job task."""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from mlflow.genai.scorers import Correctness, ExpectationsGuidelines, Safety, scorer
import mlflow

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


@scorer
def output_schema_valid(outputs):
    required = {"claim_id", "product_segment", "summary", "key_facts",
                "evidence_received", "evidence_to_confirm", "inconsistencies",
                "recommended_examiner_actions", "risk_flags", "benefit_determination"}
    return isinstance(outputs, dict) and required.issubset(outputs)


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

from lighthouse_poc.agent import (ClaimSettlementAssistant, MockClaimModel,
                                   UnityGatewayChatClient)
from lighthouse_poc.evaluation import MLflowEvaluationService
from lighthouse_poc.logging_config import configure_logging
from lighthouse_poc.prompt_registry import MLflowPromptRegistry


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--model-service", required=True)
    parser.add_argument("--model-mode", choices=("mock", "unity-gateway"), default="mock")
    parser.add_argument("--scorer-mode", choices=("deterministic", "llm-judges"), default="deterministic")
    parser.add_argument("--experiment", default="/Shared/lighthouse-claim-settlement-poc")
    parser.add_argument("--application-version", default="0.3.0")
    parser.add_argument("--dataset-version", default="1.0.0")
    return parser.parse_args()


def main():
    args = parse_args()
    logger: logging.Logger = configure_logging()
    mlflow.set_experiment(args.experiment)
    with (PROJECT / "data" / "golden_eval_dataset.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    registry = MLflowPromptRegistry(logger)
    model = (UnityGatewayChatClient(args.model_service, logger)
             if args.model_mode == "unity-gateway" else MockClaimModel())
    model_service = args.model_service if args.model_mode == "unity-gateway" else "local.mock.claim-settlement-v1"
    assistant = ClaimSettlementAssistant(
        registry, model, logger, args.prompt_name, args.prompt_version,
        args.application_version,
    )

    def predict_fn(case_id, product_segment, claim_narrative,
                   submitted_documents, policy_or_benefit_context):
        return assistant.predict(case_id, product_segment, claim_narrative,
                                 submitted_documents, policy_or_benefit_context)

    manifest = {
        "application_version": args.application_version,
        "prompt_name": args.prompt_name,
        "prompt_version": args.prompt_version,
        "model_service": model_service,
        "dataset_version": args.dataset_version,
        "scorer_version": "correctness-safety-claims-guidelines-v1",
    }
    if args.scorer_mode == "llm-judges":
        judge_model = os.getenv("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL")
        judge_options = {"model": judge_model} if judge_model else {}
        scorers = [
            Correctness(**judge_options),
            Safety(**judge_options),
            ExpectationsGuidelines(**judge_options),
        ]
        manifest["judge_model"] = judge_model or "workspace default"
    else:
        scorers = [output_schema_valid, non_decision_boundary, expected_segment_matches]
        manifest["judge_model"] = "not used (deterministic local mode)"
    result = MLflowEvaluationService(logger).run(records, predict_fn, scorers, manifest)
    print(result.tables)
    print(f"Scorer mode: {args.scorer_mode}. Inspect cases and traces, then use the separate human review gate.")


if __name__ == "__main__":
    main()

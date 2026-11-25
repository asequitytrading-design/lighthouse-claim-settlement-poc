"""Load a golden JSONL file into an MLflow evaluation dataset (visible under Datasets in the UI).

Records are {"inputs": {...}, "expectations": {...}, "tags": {...}}. Re-running merges by
inputs, so the dataset is not duplicated. Examples:
  python scripts/load_eval_dataset.py --file samples/claim_intake_triage/golden.jsonl \\
      --name claim_intake_triage_golden --experiment /lighthouse-ui-sample
  # Databricks: datasets are Unity Catalog tables, so --name is catalog.schema.table
  python scripts/load_eval_dataset.py --profile databricks --file ... \\
      --name workspace.dev_<user>_lighthouse.claim_intake_triage_golden --experiment <path>
"""
import argparse
import json
import logging
import sys
from pathlib import Path

try:
    PROJECT = Path(__file__).resolve().parents[1]
except NameError:  # some job runtimes execute scripts without __file__
    PROJECT = Path(sys.argv[0]).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import mlflow  # noqa: E402

from lighthouse_poc.config import apply_env_overrides, load_dotenv, require  # noqa: E402
from lighthouse_poc.exceptions import ConfigurationError, EvaluationError, LighthouseError  # noqa: E402
from lighthouse_poc.logging_config import configure_logging  # noqa: E402
from lighthouse_poc.runtime import load_runtime_profile  # noqa: E402

REQUIRED_KEYS = {"inputs", "expectations"}


def read_records(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"Cannot read {path}") from exc
    records = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"{path}:{number} is not valid JSON") from exc
        missing = REQUIRED_KEYS - set(record)
        if missing or not isinstance(record.get("inputs"), dict):
            raise ConfigurationError(f"{path}:{number} needs dict 'inputs' and 'expectations'")
        records.append(record)
    if not records:
        raise ConfigurationError(f"{path} has no records")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="Profile name or YAML path (default: $LIGHTHOUSE_PROFILE or local)")
    parser.add_argument("--file", required=True, type=Path, help="Golden JSONL")
    parser.add_argument("--name", required=True, help="Dataset name (Databricks: catalog.schema.table)")
    parser.add_argument("--experiment", required=True, help="Experiment the dataset is attached to")
    parser.add_argument("--env", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()
    logger: logging.Logger = configure_logging()
    load_dotenv(PROJECT / ".env")
    try:
        apply_env_overrides(args.env)
        profile = load_runtime_profile(args.profile, logger)
        mlflow.set_tracking_uri(require(profile.tracking_uri, "mlflow.tracking_uri"))
        experiment = mlflow.set_experiment(args.experiment)
        records = read_records(args.file)
        try:
            from mlflow.genai import datasets
            existing = datasets.search_datasets(experiment_ids=experiment.experiment_id,
                                                filter_string=f"name = '{args.name}'")
            dataset = existing[0] if existing else datasets.create_dataset(
                name=args.name, experiment_id=experiment.experiment_id,
                tags={"source_file": args.file.name, "synthetic": "true"})
            dataset = dataset.merge_records(records)
        except Exception as exc:
            raise EvaluationError(f"Could not load dataset {args.name}") from exc
        logger.info("eval_dataset_loaded", extra={"kpi_name": "eval_dataset_records",
                                                  "kpi_value": len(records)})
        print(f"Dataset {args.name} ({dataset.dataset_id}) in experiment {args.experiment}: "
              f"{len(records)} records merged from {args.file}")
        return 0
    except LighthouseError as exc:
        logger.error("load_eval_dataset_failed", exc_info=True,
                     extra={"error_type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)

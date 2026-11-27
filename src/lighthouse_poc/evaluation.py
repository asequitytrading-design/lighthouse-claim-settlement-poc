"""MLflow GenAI evaluation service and the golden dataset loader."""
import json
import logging
from pathlib import Path

import mlflow

from .config import PROJECT_ROOT
from .exceptions import ConfigurationError, EvaluationError

GOLDEN_DATASET = PROJECT_ROOT / "data" / "golden_eval_dataset.jsonl"


def load_golden_records(path: Path | None = None) -> list[dict]:
    """Golden records ({inputs, expectations}) from JSONL, in file order."""
    path = Path(path) if path else GOLDEN_DATASET
    try:
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read golden dataset {path}") from exc

class MLflowEvaluationService:
    MANIFEST_KEYS={"application_version","prompt_name","prompt_version","model_service",
                   "dataset_version","scorer_version","judge_model"}

    def __init__(self,logger:logging.Logger): self._logger=logger

    def run(self,data,predict_fn,scorers,manifest):
        missing=self.MANIFEST_KEYS-set(manifest)
        if missing: raise EvaluationError(f"Candidate manifest missing: {sorted(missing)}")
        try:
            result=mlflow.genai.evaluate(data=data,predict_fn=predict_fn,scorers=scorers)
            self._logger.info("evaluation_completed",extra={"prompt_name":manifest["prompt_name"],
                "prompt_version":manifest["prompt_version"],"model_service":manifest["model_service"],
                "judge_model":manifest["judge_model"],
                "application_version":manifest["application_version"],
                "kpi_name":"evaluation_cases","kpi_value":len(data) if hasattr(data,"__len__") else None})
            return result
        except Exception as exc:
            self._logger.exception("evaluation_failed",extra={"error_type":type(exc).__name__})
            raise EvaluationError("MLflow GenAI evaluation failed") from exc


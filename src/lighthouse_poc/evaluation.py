"""MLflow GenAI evaluation service."""
import logging
import mlflow
from .exceptions import EvaluationError

class MLflowEvaluationService:
    MANIFEST_KEYS={"application_version","prompt_name","prompt_version","model_service",
                   "dataset_version","scorer_version","judge_model"}

    def __init__(self,logger:logging.Logger): self._logger=logger

    def run(self,data,predict_fn,scorers,manifest):
        missing=self.MANIFEST_KEYS-set(manifest)
        if missing: raise ValueError(f"Candidate manifest missing: {sorted(missing)}")
        try:
            result=mlflow.genai.evaluate(data=data,predict_fn=predict_fn,scorers=scorers)
            self._logger.info("evaluation_completed",extra={"prompt_name":manifest["prompt_name"],
                "prompt_version":manifest["prompt_version"],"model_service":manifest["model_service"],
                "judge_model":manifest["judge_model"],
                "application_version":manifest["application_version"],
                "kpi_name":"evaluation_cases","kpi_value":len(data)})
            return result
        except Exception as exc:
            self._logger.exception("evaluation_failed",extra={"error_type":type(exc).__name__})
            raise EvaluationError("MLflow GenAI evaluation failed") from exc


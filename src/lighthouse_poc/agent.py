"""Non-decisional claim settlement preparation assistant."""
import json
import logging
import re

import mlflow

from .exceptions import OutputValidationError
from .gateway import ChatModel, Completion

_FENCE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict:
    """Parse a JSON object, tolerating a single surrounding Markdown code fence.

    Small local models often wrap JSON in ```json fences. Stripping the fence is format
    normalisation only; every contract check still applies to the parsed object.
    """
    candidate = text.strip()
    fenced = _FENCE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object")
    return parsed


class ClaimSettlementAssistant:
    REQUIRED = {"claim_id", "product_segment", "summary", "key_facts",
                "evidence_received", "evidence_to_confirm", "inconsistencies",
                "recommended_examiner_actions", "risk_flags", "benefit_determination"}
    LIST_FIELDS = ("key_facts", "evidence_received", "evidence_to_confirm",
                   "inconsistencies", "recommended_examiner_actions", "risk_flags")

    def __init__(self, registry, model: ChatModel, logger: logging.Logger,
                 prompt_name: str, prompt_version: str, app_version: str):
        self._registry, self._model, self._logger = registry, model, logger
        self._prompt_name, self._prompt_version, self._app_version = (
            prompt_name, prompt_version, app_version)
        self._prompt = None

    def _system_prompt(self) -> str:
        if self._prompt is None:  # versions are immutable, so one load per assistant
            self._prompt = self._registry.load(self._prompt_name, self._prompt_version)
        return self._prompt.format()

    @mlflow.trace
    def predict(self, case_id: str, product_segment: str, claim_narrative: str,
                submitted_documents: list, policy_or_benefit_context: str):
        user = (f"Case ID: {case_id}\nSegment: {product_segment}\n"
                f"Narrative: {claim_narrative}\nDocuments listed as received: "
                f"{submitted_documents}\nContext: {policy_or_benefit_context}\nReturn JSON only.")
        completion: Completion = self._model.complete(self._system_prompt(), user)
        context = {"case_id": case_id, "prompt_name": self._prompt_name,
                   "prompt_version": self._prompt_version,
                   "application_version": self._app_version,
                   "served_model": completion.served_model,
                   "correlation_id": completion.correlation_id,
                   "duration_ms": completion.duration_ms}
        try:
            output = extract_json_object(completion.text)
            missing = self.REQUIRED - set(output)
            if missing:
                raise ValueError(f"Missing fields: {sorted(missing)}")
            if output["claim_id"] != case_id:
                raise ValueError("Claim ID mismatch")
            if output["product_segment"] != product_segment:
                raise ValueError("Product segment mismatch")
            if output["benefit_determination"] != "Not determined":
                raise ValueError("Benefit determination must be Not determined")
            for field in self.LIST_FIELDS:
                if not isinstance(output[field], list):
                    raise ValueError(f"{field} must be a list")
        except Exception as exc:
            self._logger.error("claim_summary_invalid", exc_info=True, extra={
                **context, "error_type": type(exc).__name__,
                "error_category": "output_validation",
                "kpi_name": "output_schema_valid", "kpi_value": 0})
            raise OutputValidationError(f"Invalid output for case {case_id}") from exc
        self._logger.info("claim_summary_validated", extra={
            **context, "kpi_name": "output_schema_valid", "kpi_value": 1})
        return output

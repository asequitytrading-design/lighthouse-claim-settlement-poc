"""Non-decisional claim settlement preparation assistant."""
import json
import logging
import time
import mlflow
from .exceptions import ModelInvocationError, OutputValidationError


class UnityGatewayChatClient:
    """OpenAI-compatible client routed through a Databricks Unity Gateway service."""
    def __init__(self, model_service: str, logger: logging.Logger,
                 workspace_host: str | None = None, token: str | None = None):
        import os
        host = (workspace_host or os.getenv("DATABRICKS_HOST", "")).rstrip("/")
        api_token = token or os.getenv("DATABRICKS_TOKEN")
        if not host or not model_service:
            raise ValueError("Set DATABRICKS_HOST and a qualified Unity Gateway model service.")
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=api_token,
                base_url=f"{host}/ai-gateway/mlflow/v1",
            )
        except Exception as exc:
            logger.exception("unity_gateway_client_initialization_failed", extra={
                "model_service": model_service, "error_type": type(exc).__name__})
            raise ModelInvocationError("Could not initialize Unity Gateway client") from exc
        self._model_service, self._logger = model_service, logger

    def complete(self, system_prompt: str, user_message: str):
        started = time.perf_counter()
        try:
            result = self._client.chat.completions.create(
                model=self._model_service,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_message}],
                temperature=0,
            )
            elapsed = round((time.perf_counter() - started) * 1000)
            self._logger.info("model_request_completed", extra={
                "model_service": self._model_service, "duration_ms": elapsed,
                "kpi_name": "model_latency_ms", "kpi_value": elapsed})
            return result.choices[0].message.content or "", elapsed
        except Exception as exc:
            self._logger.exception("model_request_failed", extra={
                "model_service": self._model_service, "error_type": type(exc).__name__})
            raise ModelInvocationError(f"Unity Gateway request failed for {self._model_service}") from exc


class ClaimSettlementAssistant:
    REQUIRED = {"claim_id", "product_segment", "summary", "key_facts",
                "evidence_received", "evidence_to_confirm", "inconsistencies",
                "recommended_examiner_actions", "risk_flags", "benefit_determination"}

    def __init__(self, registry, model, logger, prompt_name, prompt_version, app_version):
        self._registry, self._model, self._logger = registry, model, logger
        self._prompt_name, self._prompt_version, self._app_version = (
            prompt_name, prompt_version, app_version)

    @mlflow.trace
    def predict(self, case_id: str, product_segment: str, claim_narrative: str,
                submitted_documents: list, policy_or_benefit_context: str):
        prompt = self._registry.load(self._prompt_name, self._prompt_version)
        user = (f"Case ID: {case_id}\nSegment: {product_segment}\n"
                f"Narrative: {claim_narrative}\nDocuments listed as received: "
                f"{submitted_documents}\nContext: {policy_or_benefit_context}\nReturn JSON only.")
        raw, elapsed = self._model.complete(prompt.format(), user)
        try:
            output = json.loads(raw)
            missing = self.REQUIRED - set(output)
            if missing:
                raise ValueError(f"Missing fields: {sorted(missing)}")
            if output["claim_id"] != case_id:
                raise ValueError("Claim ID mismatch")
            if output["product_segment"] != product_segment:
                raise ValueError("Product segment mismatch")
            if output["benefit_determination"] != "Not determined":
                raise ValueError("Benefit determination must be Not determined")
            for field in ("key_facts", "evidence_received", "evidence_to_confirm",
                          "inconsistencies", "recommended_examiner_actions", "risk_flags"):
                if not isinstance(output[field], list):
                    raise ValueError(f"{field} must be a list")
            self._logger.info("claim_summary_validated", extra={
                "case_id": case_id, "prompt_name": self._prompt_name,
                "prompt_version": self._prompt_version,
                "application_version": self._app_version, "duration_ms": elapsed,
                "kpi_name": "output_schema_valid", "kpi_value": 1})
            return output
        except Exception as exc:
            self._logger.exception("claim_summary_invalid", extra={
                "case_id": case_id, "prompt_name": self._prompt_name,
                "prompt_version": self._prompt_version, "error_type": type(exc).__name__,
                "kpi_name": "output_schema_valid", "kpi_value": 0})
            raise OutputValidationError(f"Invalid output for case {case_id}") from exc


class MockClaimModel:
    """Deterministic local response adapter for wiring and UI development only."""
    def complete(self, system_prompt: str, user_message: str):
        import ast
        import re
        started = time.perf_counter()
        def value(label, fallback):
            match = re.search(rf"^{re.escape(label)}:\s*(.*)$", user_message, re.MULTILINE)
            return match.group(1).strip() if match else fallback
        case_id = value("Case ID", "SYNTHETIC-CASE")
        segment = value("Segment", "Unknown synthetic segment")
        narrative = value("Narrative", "Synthetic claim details supplied")
        documents_raw = value("Documents listed as received", "[]")
        try:
            docs = ast.literal_eval(documents_raw)
            docs = docs if isinstance(docs, list) else [str(docs)]
        except (ValueError, SyntaxError):
            docs = ["Submitted documents listed in synthetic case"]
        output = {
            "claim_id": case_id,
            "product_segment": segment,
            "summary": narrative[:400],
            "key_facts": [narrative[:250]],
            "evidence_received": docs,
            "evidence_to_confirm": ["Confirm applicable claim-specific evidence with examiner"],
            "inconsistencies": [],
            "recommended_examiner_actions": ["Review submitted materials and apply approved policy procedures"],
            "risk_flags": ["Human examiner review required"],
            "benefit_determination": "Not determined",
        }
        elapsed = round((time.perf_counter() - started) * 1000)
        return json.dumps(output), elapsed

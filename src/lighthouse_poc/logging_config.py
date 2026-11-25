"""Minimal structured JSON logging for Databricks driver logs."""
import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format standard records and approved diagnostic context as JSON."""
    FIELDS = ("case_id", "trace_id", "run_id", "prompt_name", "prompt_version",
              "model_service", "judge_model", "dataset_version", "application_version",
              "duration_ms", "error_type", "kpi_name", "kpi_value")

    def format(self, record):
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self.FIELDS:
            if hasattr(record, field):
                event[field] = getattr(record, field)
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False, default=str)


def configure_logging(level=logging.INFO):
    logger = logging.getLogger("lighthouse_poc")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger

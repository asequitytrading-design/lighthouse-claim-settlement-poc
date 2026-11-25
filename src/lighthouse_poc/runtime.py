"""Databricks job runtime helpers (task values) with explicit local behaviour."""
import logging

from .exceptions import ConfigurationError


def publish_task_value(key: str, value: str, logger: logging.Logger) -> None:
    """Expose a value to downstream job tasks via ``dbutils.jobs.taskValues``.

    Called only when a script is asked to (``--task-value-key``), so a failure here is a
    real error on Databricks rather than something to ignore.
    """
    try:
        from databricks.sdk.runtime import dbutils
        dbutils.jobs.taskValues.set(key=key, value=value)
    except Exception as exc:
        logger.error("task_value_publish_failed", exc_info=True,
                     extra={"error_type": type(exc).__name__, "error_category": "configuration"})
        raise ConfigurationError(
            f"Could not publish task value '{key}'. Use --task-value-key only inside a "
            "Databricks job task.") from exc
    logger.info("task_value_published", extra={"kpi_name": "task_value", "kpi_value": key})

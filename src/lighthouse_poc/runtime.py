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


def ensure_databricks_host(logger: logging.Logger, env=None, config_factory=None) -> None:
    """Set DATABRICKS_HOST from databricks-sdk unified auth when it is not already set.

    Bundles cannot pass ``${workspace.host}`` when the host comes from a CLI profile, and job
    tasks do not export it, but the SDK resolves it from the same auth used for model calls.
    """
    import os
    env = os.environ if env is None else env
    if env.get("DATABRICKS_HOST"):
        return
    try:
        if config_factory is None:
            from databricks.sdk.core import Config
            config_factory = Config
        host = config_factory().host
    except Exception as exc:
        raise ConfigurationError("Could not resolve the Databricks workspace host") from exc
    if not host:
        raise ConfigurationError("Databricks authentication did not provide a workspace host")
    env["DATABRICKS_HOST"] = host.rstrip("/")
    logger.info("databricks_host_resolved_from_sdk")


def load_runtime_profile(name, logger: logging.Logger):
    """Load a profile, resolving the workspace host first for Databricks-authenticated ones."""
    from .config import load_profile
    profile = load_profile(name)
    if profile.gateway.auth.type == "databricks_sdk":
        ensure_databricks_host(logger)
        profile = load_profile(name)  # re-interpolate with the resolved host
    return profile

"""Activate the MLflow experiment, with its traces stored in Unity Catalog (ADR-017).

On Databricks the profile's `mlflow.trace_storage` links the experiment to Unity Catalog Delta
tables (OpenTelemetry format): <catalog>.<schema>.<table_prefix>_otel_spans, _otel_logs,
_otel_annotations, _trace_metadata, _trace_unified ... Traces, judge assessments and human
labels are then governed by Unity Catalog (grants, lineage, retention) and queryable with SQL,
like every other platform table. Runs, params, metrics and artifacts stay in the experiment.

Free Edition offers the table-prefix form only (schema-level trace locations are disabled).
If linking is not available, or the experiment is already linked elsewhere, the experiment is
activated with MLflow-managed trace storage and the reason is logged (capability check, see
ADR-016): the run is never blocked by it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .config import Profile


@dataclass(frozen=True)
class TraceStorage:
    location: str          # "<catalog>.<schema>.<prefix>_*" or "mlflow experiment"
    in_unity_catalog: bool
    detail: str = ""


def default_table_prefix(experiment: str) -> str:
    """lh_<experiment leaf name> as a valid table name prefix."""
    leaf = experiment.rstrip("/").rsplit("/", 1)[-1]
    return "lh_" + (re.sub(r"[^0-9a-zA-Z]+", "_", leaf).strip("_").lower() or "traces")


def activate_experiment(profile: Profile, logger: logging.Logger,
                        name: str | None = None) -> tuple[Any, TraceStorage]:
    import mlflow

    from .config import require
    name = name or require(profile.experiment, "mlflow.experiment")
    storage = getattr(profile, "trace_storage", None)
    if not storage or not str(profile.tracking_uri or "").startswith("databricks"):
        return mlflow.set_experiment(name), TraceStorage("mlflow experiment", False)
    from mlflow.entities.trace_location import UnityCatalog
    prefix = storage.table_prefix or default_table_prefix(name)
    target = f"{storage.catalog}.{storage.schema}.{prefix}_*"
    try:
        experiment = mlflow.set_experiment(name, trace_location=UnityCatalog(
            catalog_name=storage.catalog, schema_name=storage.schema, table_prefix=prefix))
    except Exception as exc:          # not offered here, or linked elsewhere: keep running
        detail = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:240]}"
        logger.warning("trace_storage_fallback", extra={"experiment": name, "target": target,
                                                        "detail": detail})
        return mlflow.set_experiment(name), TraceStorage("mlflow experiment", False, detail)
    logger.info("trace_storage_unity_catalog", extra={"experiment": name, "target": target})
    return experiment, TraceStorage(target, True)

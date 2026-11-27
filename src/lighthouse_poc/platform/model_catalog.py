"""The governed model list: config/model_catalog.yaml -> <mkt>.model_catalog (type-2 history).

Reference data changes often (a model approved, retired, re-verified); structure changes rarely.
So the list is not an Alembic migration: it is synced. A changed entry closes its current row
(valid_to, is_current = false) and opens a new one, so every evaluation run and invocation can be
traced to the catalog entry that was in force at the time.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..config import PROJECT_ROOT
from ..exceptions import ConfigurationError
from .naming import PlatformNames
from .sql import literal

CATALOG_FILE = PROJECT_ROOT / "config" / "model_catalog.yaml"
TASKS = {"chat", "embeddings"}
ROLES = {"app", "judge", "embedding"}
STATUSES = {"approved", "candidate", "unavailable", "retired"}


@dataclass(frozen=True)
class ModelEntry:
    key: str
    model_service: str
    task: str
    status: str
    roles: tuple[str, ...] = field(default_factory=tuple)
    family: str | None = None
    provider: str | None = None
    embedding_dimensions: int | None = None
    verified: str | None = None
    notes: str | None = None

    def record_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def allows(self, role: str) -> bool:
        return self.status == "approved" and role in self.roles


def _entry(raw: dict, index: int) -> ModelEntry:
    where = f"models[{index}] ({raw.get('key', '?')})"
    for required in ("key", "model_service", "task", "status"):
        if not raw.get(required):
            raise ConfigurationError(f"{where}: '{required}' is required")
    if raw["task"] not in TASKS:
        raise ConfigurationError(f"{where}: task must be one of {sorted(TASKS)}")
    if raw["status"] not in STATUSES:
        raise ConfigurationError(f"{where}: status must be one of {sorted(STATUSES)}")
    roles = tuple(raw.get("roles") or ())
    if not roles or set(roles) - ROLES:
        raise ConfigurationError(f"{where}: roles must be a non-empty subset of {sorted(ROLES)}")
    if not str(raw["model_service"]).count(".") == 2:
        raise ConfigurationError(f"{where}: model_service must be catalog.schema.model")
    dims = raw.get("embedding_dimensions")
    if raw["task"] == "embeddings":
        if not isinstance(dims, int) or dims <= 0:
            raise ConfigurationError(f"{where}: embeddings need a positive embedding_dimensions")
        if set(roles) != {"embedding"}:
            raise ConfigurationError(f"{where}: an embeddings model has the role 'embedding' only")
    elif dims is not None or "embedding" in roles:
        raise ConfigurationError(f"{where}: chat models have no embedding role or dimensions")
    verified = raw.get("verified")
    if verified is not None:
        try:
            date.fromisoformat(str(verified))
        except ValueError:
            raise ConfigurationError(f"{where}: verified must be YYYY-MM-DD") from None
    return ModelEntry(key=str(raw["key"]), model_service=str(raw["model_service"]),
                      task=raw["task"], status=raw["status"], roles=roles,
                      family=raw.get("family"), provider=raw.get("provider"),
                      embedding_dimensions=dims, verified=str(verified) if verified else None,
                      notes=raw.get("notes"))


def load_model_catalog(path: Path | None = None) -> list[ModelEntry]:
    path = Path(path) if path else CATALOG_FILE
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read model catalog {path}") from exc
    entries = [_entry(raw, i) for i, raw in enumerate(data.get("models") or [])]
    keys = [e.key for e in entries]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ConfigurationError(f"Duplicate model keys in {path.name}: {duplicates}")
    if not entries:
        raise ConfigurationError(f"{path.name} lists no models")
    return entries


class ModelCatalogSync:
    """Builds and runs the type-2 merge of the YAML list into the catalog table."""

    def __init__(self, names: PlatformNames, logger: logging.Logger):
        self._table = names.table("model_catalog")
        self._logger = logger

    def _incoming(self, entries: list[ModelEntry]) -> str:
        rows = []
        for e in entries:
            rows.append("(" + ", ".join([
                literal(e.key), literal(e.model_service), literal(e.task), literal(e.family),
                literal(e.provider), literal(list(e.roles)), literal(e.status),
                f"CAST({literal(e.embedding_dimensions)} AS INT)",
                f"CAST({literal(e.verified)} AS DATE)", literal(e.notes),
                literal(e.record_hash())]) + ")")
        return ("SELECT * FROM VALUES " + ",\n  ".join(rows) +
                " AS s(model_key, model_service, task, family, provider, roles, status, "
                "embedding_dimensions, verified_on, notes, record_hash)")

    def statements(self, entries: list[ModelEntry]) -> list[str]:
        incoming = self._incoming(entries)
        keys = ", ".join(literal(e.key) for e in entries)
        return [
            # 1. close current rows whose attributes changed
            f"MERGE INTO {self._table} t USING ({incoming}) s "
            "ON t.model_key = s.model_key AND t.is_current "
            "WHEN MATCHED AND t.record_hash <> s.record_hash THEN "
            "UPDATE SET is_current = false, valid_to = current_timestamp()",
            # 2. close keys removed from the YAML (kept as history)
            f"UPDATE {self._table} SET is_current = false, valid_to = current_timestamp() "
            f"WHERE is_current AND model_key NOT IN ({keys})",
            # 3. open a row for new and changed keys
            f"INSERT INTO {self._table} (model_key, model_service, task, family, provider, roles, "
            "status, embedding_dimensions, verified_on, notes, record_hash, valid_from, valid_to, "
            f"is_current, synced_by) SELECT s.*, current_timestamp(), NULL, true, current_user() "
            f"FROM ({incoming}) s LEFT ANTI JOIN {self._table} t "
            "ON t.model_key = s.model_key AND t.is_current",
        ]

    def sync(self, engine: Any, entries: list[ModelEntry]) -> dict[str, int]:
        from sqlalchemy import text
        counts = {}
        with engine.connect() as connection:
            for label, statement in zip(("closed_changed", "closed_removed", "opened"),
                                        self.statements(entries)):
                result = connection.execute(text(statement.replace(":", "\\:")))
                counts[label] = max(int(getattr(result, "rowcount", 0) or 0), 0)
        self._logger.info("model_catalog_synced", extra={
            "table": self._table, "models": len(entries), **counts,
            "kpi_name": "model_catalog_changes", "kpi_value": counts.get("opened", 0)})
        return counts


def approved_models(entries: list[ModelEntry], role: str) -> list[ModelEntry]:
    return [e for e in entries if e.allows(role)]


def as_rows(entries: list[ModelEntry]) -> list[dict[str, Any]]:
    return [{**asdict(e), "roles": ",".join(e.roles)} for e in entries]

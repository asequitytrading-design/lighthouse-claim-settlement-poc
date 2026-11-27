"""Schema migrations for the AI Marketplace data platform: Alembic on a Databricks SQL warehouse.

Why Alembic (ADR-012): versioned revisions with upgrade/downgrade, a revision graph, history and
offline SQL rendering, in Python like the rest of the POC. Databricks specifics handled here:

- DDL is not transactional on Delta. Each revision runs on its own and its statements are
  written to be re-runnable (IF NOT EXISTS / IF EXISTS), so a revision that failed half-way can
  simply be applied again.
- Restore points: before an upgrade or downgrade the runner records every platform table's
  current Delta version in `schema_migration_audit`; `restore_statements()` returns the
  `RESTORE TABLE ... TO VERSION AS OF` commands that undo a data migration.
- Revisions use fully qualified names through placeholders ({catalog}, {mkt}, {obs}, {kn}, {clm}) so one
  revision serves every environment and personal dev prefix.
- Concurrency: run migrations from one place at a time (the setup notebook or a job with
  max_concurrent_runs: 1); Delta has no advisory locks.
"""
from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..exceptions import ConfigurationError, LighthouseError
from .naming import PlatformNames

SCRIPT_LOCATION = Path(__file__).with_name("alembic")
VERSION_TABLE = "alembic_version"


class MigrationError(LighthouseError):
    """A schema migration could not be applied or inspected."""


def render(statement: str, names: PlatformNames) -> str:
    """Fill the {catalog}/{mkt}/{obs}/{kn} placeholders of a revision statement."""
    try:
        return statement.format(**names.sql_names())
    except (KeyError, IndexError) as exc:
        raise ConfigurationError(f"Unknown placeholder in migration SQL: {exc}") from exc


def run_sql(*statements: str) -> None:
    """Execute statements inside an Alembic revision (online) or emit them (offline --sql)."""
    from alembic import context, op
    names: PlatformNames = context.config.attributes["platform_names"]
    from sqlalchemy import text
    for statement in statements:
        # text(): escape ":" so time literals or comments are never read as bind parameters
        op.execute(text(render(statement, names).replace(":", "\\:")))


def column_exists(schema_key: str, table: str, column: str) -> bool:
    """Inside a revision: does the column exist? Offline (--sql) always False, so the DDL is shown.

    Delta DDL is not transactional; checking first keeps ADD COLUMNS re-runnable after a
    partial failure. schema_key is one of "mkt", "obs", "kn", "clm".
    """
    from alembic import context, op
    from sqlalchemy import text
    if context.is_offline_mode():
        return False
    names: PlatformNames = context.config.attributes["platform_names"]
    schema = {"mkt": names.marketplace, "obs": names.observability, "kn": names.knowledge,
              "clm": names.claims}[schema_key]
    row = op.get_bind().execute(text(
        f"SELECT count(*) FROM {names.catalog}.information_schema.columns "
        "WHERE table_schema = :schema AND table_name = :table AND column_name = :column"),
        {"schema": schema, "table": table, "column": column}).fetchone()
    return bool(row and row[0])


@dataclass(frozen=True)
class Revision:
    revision: str
    down_revision: str | None
    message: str


class SchemaMigrator:
    """Applies and inspects revisions for one environment (catalog + schema prefix)."""

    def __init__(self, names: PlatformNames, logger: logging.Logger, engine: Any = None):
        self._names, self._logger, self._engine = names, logger, engine

    # ------------------------------------------------------------------ alembic plumbing
    def _config(self, connection: Any = None, output: io.StringIO | None = None):
        from alembic.config import Config
        import databricks.sqlalchemy  # noqa: F401  registers the "databricks" Alembic impl
        config = Config(output_buffer=output) if output is not None else Config()
        config.set_main_option("script_location", str(SCRIPT_LOCATION))
        config.attributes["platform_names"] = self._names
        if connection is not None:
            config.attributes["connection"] = connection
        return config

    def _require_engine(self):
        if self._engine is None:
            raise ConfigurationError("SchemaMigrator needs a SQL warehouse engine for this action")
        return self._engine

    def _bootstrap(self, connection) -> None:
        """The version table lives in the marketplace schema, so that schema must exist first."""
        from sqlalchemy import text
        connection.execute(text(render(
            "CREATE SCHEMA IF NOT EXISTS {mkt} COMMENT 'AI Marketplace: governed reference data, "
            "migration history (Lighthouse POC, synthetic data only)'", self._names)))

    # ------------------------------------------------------------------ inspection
    def revisions(self) -> list[Revision]:
        """All revisions, oldest first (works offline)."""
        from alembic.script import ScriptDirectory
        script = ScriptDirectory.from_config(self._config())
        items = [Revision(r.revision, r.down_revision if isinstance(r.down_revision, str) else None,
                          (r.doc or "").strip()) for r in script.walk_revisions()]
        return list(reversed(items))

    def heads(self) -> list[str]:
        from alembic.script import ScriptDirectory
        return list(ScriptDirectory.from_config(self._config()).get_heads())

    def current(self) -> list[str]:
        """Applied revision(s); empty when the platform has not been migrated yet."""
        from alembic.runtime.migration import MigrationContext
        with self._require_engine().connect() as connection:
            context = MigrationContext.configure(connection, opts={
                "version_table": VERSION_TABLE, "version_table_schema": self._names.marketplace})
            return list(context.get_current_heads())

    def offline_sql(self, target: str = "head", start: str | None = None) -> str:
        """The SQL an upgrade would run, without a connection (review / change tickets)."""
        from alembic import command
        buffer = io.StringIO()
        buffer.write(f"USE CATALOG {self._names.catalog};\n")
        buffer.write(render("CREATE SCHEMA IF NOT EXISTS {mkt};\n", self._names))
        span = f"{start}:{target}" if start else target
        command.upgrade(self._config(output=buffer), span, sql=True)
        return buffer.getvalue()

    # ------------------------------------------------------------------ changes
    def upgrade(self, target: str = "head") -> str:
        return self._apply("upgrade", target)

    def downgrade(self, target: str) -> str:
        return self._apply("downgrade", target)

    def _apply(self, direction: str, target: str) -> str:
        from alembic import command
        run_id = str(uuid.uuid4())
        engine = self._require_engine()
        with engine.connect() as connection:
            self._bootstrap(connection)
        before = self.current()
        with engine.connect() as connection:
            self.record_restore_points(connection, run_id, before, f"{direction}:{target}")
            config = self._config(connection=connection)
            try:
                getattr(command, direction)(config, target)
            except Exception as exc:
                self._logger.error("schema_migration_failed", extra={
                    "direction": direction, "target": target, "migration_run_id": run_id,
                    "from_revision": ",".join(before) or "base",
                    "error_type": type(exc).__name__, "error_category": "migration"})
                raise MigrationError(f"Schema {direction} to {target} failed (run {run_id}); "
                                     "fix the cause and re-run: revisions are re-runnable, and "
                                     "restore points are in schema_migration_audit") from exc
        after = self.current()
        self._logger.info("schema_migration_completed", extra={
            "direction": direction, "target": target, "migration_run_id": run_id,
            "from_revision": ",".join(before) or "base", "to_revision": ",".join(after) or "base",
            "kpi_name": "schema_migrations_applied", "kpi_value": 1})
        return run_id

    # ------------------------------------------------------------------ restore points
    def platform_tables(self, connection) -> list[str]:
        from sqlalchemy import text
        schemas = ", ".join(f"'{s}'" for s in self._names.schemas)
        rows = connection.execute(text(
            f"SELECT table_schema, table_name FROM {self._names.catalog}.information_schema.tables "
            f"WHERE table_schema IN ({schemas}) AND table_type IN ('MANAGED', 'EXTERNAL') "
            f"AND table_name <> '{VERSION_TABLE}' ORDER BY 1, 2")).fetchall()
        return [self._names.fq(r[0], r[1]) for r in rows]

    def record_restore_points(self, connection, run_id: str, before: list[str], intent: str) -> int:
        """Store each platform table's current Delta version before a change."""
        from sqlalchemy import text
        tables = self.platform_tables(connection)
        audit = self._names.table("schema_migration_audit")
        if audit not in tables:
            return 0     # first run: nothing to protect yet
        values = []
        for table in tables:
            version = connection.execute(text(f"DESCRIBE HISTORY {table} LIMIT 1")).fetchone()[0]
            values.append(f"('{run_id}', '{','.join(before) or 'base'}', '{intent}', "
                          f"'{table}', {int(version)}, current_timestamp(), current_user())")
        connection.execute(text(
            f"INSERT INTO {audit} (migration_run_id, from_revision, intent, table_name, "
            f"delta_version, recorded_at, recorded_by) VALUES " + ", ".join(values)))
        return len(values)

    def restore_statements(self, run_id: str) -> list[str]:
        """RESTORE commands that return every table to its state before migration run `run_id`."""
        from sqlalchemy import text
        audit = self._names.table("schema_migration_audit")
        with self._require_engine().connect() as connection:
            rows = connection.execute(text(
                f"SELECT table_name, delta_version FROM {audit} WHERE migration_run_id = :run "
                f"AND table_name <> '{audit}'"), {"run": run_id}).fetchall()
        return [f"RESTORE TABLE {t} TO VERSION AS OF {v}" for t, v in rows]

"""model_invocation.served_model: the model the gateway reports, backfilled for existing rows

Revision ID: 0002_invocation_served_model
Revises: 0001_platform_baseline
Create Date: 2026-11-27

Example of a structure change with a data migration in one revision. With fallback routes
the gateway may serve a different model than the one requested, so the response's model is
recorded separately. Rows written before this revision get the requested service as their best
known value. Downgrade drops the column (possible because the table uses column mapping); the
values can be recovered with the restore point recorded in schema_migration_audit.
"""
from lighthouse_poc.platform.migrations import column_exists, run_sql

revision = "0002_invocation_served_model"
down_revision = "0001_platform_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not column_exists("obs", "model_invocation", "served_model"):
        run_sql("ALTER TABLE {obs}.model_invocation ADD COLUMNS ("
                "served_model STRING COMMENT 'Model reported by the gateway response (fallbacks may differ)')")
    run_sql("UPDATE {obs}.model_invocation SET served_model = model_service WHERE served_model IS NULL")


def downgrade() -> None:
    run_sql(
        "ALTER TABLE {obs}.model_invocation DROP COLUMNS IF EXISTS (served_model)",
    )

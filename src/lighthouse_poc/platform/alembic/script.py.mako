"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

Conventions (ADR-012): fully qualified names via {catalog}/{mkt}/{obs}/{kn}; re-runnable
statements (IF NOT EXISTS / IF EXISTS); a structure change that needs data moved does it in the
same revision (ADD COLUMNS, then UPDATE/MERGE); downgrade undoes the structure, and data is
recovered from the restore points in schema_migration_audit.
"""
from lighthouse_poc.platform.migrations import run_sql

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    run_sql(
        ${upgrades if upgrades else '"SELECT 1"'},
    )


def downgrade() -> None:
    run_sql(
        ${downgrades if downgrades else '"SELECT 1"'},
    )

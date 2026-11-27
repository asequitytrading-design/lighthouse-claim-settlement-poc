"""Alembic environment for the AI Marketplace platform (driven by SchemaMigrator, no alembic.ini).

SchemaMigrator puts `platform_names` (and, online, a live `connection`) into
config.attributes. Delta DDL is not transactional: transactional_ddl=False and one revision
at a time, so a failure leaves the version table at the last completed revision.
"""
from alembic import context

from lighthouse_poc.platform.migrations import VERSION_TABLE

config = context.config
names = config.attributes["platform_names"]
OPTIONS = {"version_table": VERSION_TABLE, "version_table_schema": names.marketplace,
           "transactional_ddl": False, "transaction_per_migration": True}


def run_offline() -> None:
    context.configure(url="databricks://", dialect_name="databricks", literal_binds=True,
                      **OPTIONS)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    connection = config.attributes["connection"]
    context.configure(connection=connection, **OPTIONS)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()

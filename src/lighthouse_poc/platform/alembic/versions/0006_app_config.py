"""Runtime configuration of marketplace apps (live switches for the demo app)

Revision ID: 0006_app_config
Revises: 0005_claims_data
Create Date: 2026-11-27

<mkt>.app_config holds one row per app and setting (session store, default model, semantic cache,
grounding, Genie). Written by notebook 06 (validated by lighthouse_poc.platform.app_config);
the running app re-reads it at most once a minute when requests arrive, so switching needs no
redeploy. Settings only: no secrets.
"""
from lighthouse_poc.platform.migrations import run_sql

revision = "0006_app_config"
down_revision = "0005_claims_data"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql("""CREATE TABLE IF NOT EXISTS {mkt}.app_config (
        app_name STRING NOT NULL COMMENT 'Databricks App name, e.g. lighthouse-demo-dev',
        config_key STRING NOT NULL COMMENT 'session_store | default_model | semantic_cache | cache_threshold | grounding_check | genie',
        config_value STRING NOT NULL,
        updated_by STRING, updated_at TIMESTAMP NOT NULL,
        CONSTRAINT app_config_pk PRIMARY KEY (app_name, config_key)
    ) COMMENT 'Live settings of marketplace apps (no secrets)'
    TBLPROPERTIES ('delta.columnMapping.mode' = 'name', 'delta.enableChangeDataFeed' = 'true')""")


def downgrade() -> None:
    run_sql("DROP TABLE IF EXISTS {mkt}.app_config")

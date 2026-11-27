"""Platform baseline: marketplace, observability and knowledge schemas, tables and views

Revision ID: 0001_platform_baseline
Revises:
Create Date: 2026-11-27

Data model: docs/ai-marketplace-platform.md. All tables are Delta with column mapping (so a
later revision can rename or drop columns) and change data feed (incremental consumers, and the
Delta Sync vector index needs it on document_chunk). Liquid clustering on the columns
dashboards filter by. Synthetic data only: no real policyholder data, no full claim narratives.
"""
from lighthouse_poc.platform.migrations import run_sql

revision = "0001_platform_baseline"
down_revision = None
branch_labels = None
depends_on = None

DELTA = ("TBLPROPERTIES ('delta.columnMapping.mode' = 'name', "
         "'delta.enableChangeDataFeed' = 'true')")


def upgrade() -> None:
    run_sql(
        # ---------------------------------------------------------------- schemas
        "CREATE SCHEMA IF NOT EXISTS {mkt} COMMENT 'AI Marketplace: governed reference data, "
        "migration history (Lighthouse POC, synthetic data only)'",
        "CREATE SCHEMA IF NOT EXISTS {obs} COMMENT 'AI Marketplace: usage and quality events "
        "and dashboard views'",
        "CREATE SCHEMA IF NOT EXISTS {kn} COMMENT 'AI Marketplace: knowledge base documents, "
        "embedded chunks and vector index (synthetic)'",

        # ---------------------------------------------------------------- marketplace
        f"""CREATE TABLE IF NOT EXISTS {{mkt}}.model_catalog (
            model_key STRING NOT NULL COMMENT 'Logical model name used by code and profiles',
            model_service STRING NOT NULL COMMENT 'Unity Catalog model service routed by Unity Gateway',
            task STRING NOT NULL COMMENT 'chat | embeddings',
            family STRING, provider STRING,
            roles ARRAY<STRING> COMMENT 'app | judge | embedding',
            status STRING NOT NULL COMMENT 'approved | candidate | unavailable | retired',
            embedding_dimensions INT,
            verified_on DATE COMMENT 'Last successful gateway probe recorded in config',
            notes STRING,
            record_hash STRING NOT NULL COMMENT 'Hash of the attributes; a change opens a new version',
            valid_from TIMESTAMP NOT NULL, valid_to TIMESTAMP, is_current BOOLEAN NOT NULL,
            synced_by STRING
        ) CLUSTER BY (model_key)
        COMMENT 'Governed model list (type-2 history), synced from config/model_catalog.yaml'
        {DELTA}""",
        f"""CREATE TABLE IF NOT EXISTS {{mkt}}.use_case (
            use_case_id STRING NOT NULL, name STRING NOT NULL, description STRING,
            owner STRING, risk_tier STRING COMMENT 'low | medium | high',
            data_classification STRING COMMENT 'synthetic only in the POC',
            status STRING NOT NULL COMMENT 'active | paused | retired',
            created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL
        ) COMMENT 'Applications that consume marketplace models and prompts' {DELTA}""",
        f"""CREATE TABLE IF NOT EXISTS {{mkt}}.release_decision (
            decision_id STRING NOT NULL, use_case_id STRING NOT NULL,
            prompt_name STRING NOT NULL, prompt_version STRING NOT NULL,
            decision STRING NOT NULL COMMENT 'approved | rejected',
            reviewer STRING NOT NULL, rationale STRING,
            eval_run_id STRING, review_score DOUBLE, verdict STRING,
            model_key STRING, model_service STRING, dataset_version STRING, git_commit STRING,
            decided_at TIMESTAMP NOT NULL
        ) CLUSTER BY (use_case_id)
        COMMENT 'Human review decisions (lighthouse promote, ADR-009); evidence for publishing'
        {DELTA}""",
        f"""CREATE TABLE IF NOT EXISTS {{mkt}}.schema_migration_audit (
            migration_run_id STRING NOT NULL, from_revision STRING, intent STRING,
            table_name STRING NOT NULL, delta_version BIGINT NOT NULL,
            recorded_at TIMESTAMP NOT NULL, recorded_by STRING
        ) COMMENT 'Restore points: Delta version of every platform table before a migration' {DELTA}""",

        # ---------------------------------------------------------------- observability
        f"""CREATE TABLE IF NOT EXISTS {{obs}}.model_invocation (
            event_id STRING NOT NULL COMMENT 'Also sent as request tag lh_event_id (join key to system.ai_gateway.usage)',
            event_time TIMESTAMP NOT NULL,
            event_date DATE GENERATED ALWAYS AS (CAST(event_time AS DATE)),
            environment STRING, use_case_id STRING,
            purpose STRING COMMENT 'app | judge | embedding_index | embedding_query | probe',
            model_key STRING, model_service STRING, task STRING,
            prompt_name STRING, prompt_version STRING, app_version STRING, git_commit STRING,
            trace_id STRING, run_id STRING,
            case_id STRING COMMENT 'Synthetic case identifier only',
            status STRING NOT NULL COMMENT 'ok | error', error_category STRING,
            latency_ms BIGINT, input_tokens BIGINT, output_tokens BIGINT,
            input_count INT COMMENT 'Embeddings: number of inputs', embedding_dimensions INT
        ) CLUSTER BY (event_time, use_case_id)
        COMMENT 'One row per model call made by marketplace applications (no prompt or response text)'
        {DELTA}""",
        f"""CREATE TABLE IF NOT EXISTS {{obs}}.eval_run (
            run_id STRING NOT NULL, experiment_id STRING, use_case_id STRING,
            prompt_name STRING, prompt_version STRING, model_key STRING, model_service STRING,
            judge_model_key STRING, dataset_name STRING, dataset_version STRING,
            eval_spec STRING, scorer_version STRING, cases INT,
            combined_score DOUBLE, verdict STRING COMMENT 'READY | NEEDS_REVIEW | BLOCKED',
            git_commit STRING, started_at TIMESTAMP, finished_at TIMESTAMP
        ) CLUSTER BY (use_case_id)
        COMMENT 'MLflow evaluation runs (lighthouse eval) summarised for dashboards' {DELTA}""",
        f"""CREATE TABLE IF NOT EXISTS {{obs}}.eval_case_result (
            run_id STRING NOT NULL, case_id STRING NOT NULL, criterion STRING NOT NULL,
            kind STRING COMMENT 'gate | rubric | judge', value STRING, score DOUBLE,
            passed BOOLEAN, trace_id STRING
        ) CLUSTER BY (run_id) COMMENT 'Per case and criterion scores of an evaluation run' {DELTA}""",
        f"""CREATE TABLE IF NOT EXISTS {{obs}}.vector_query (
            event_id STRING NOT NULL, event_time TIMESTAMP NOT NULL,
            event_date DATE GENERATED ALWAYS AS (CAST(event_time AS DATE)),
            use_case_id STRING, index_name STRING, embedding_model_key STRING,
            num_results_requested INT, num_results INT, latency_ms BIGINT,
            status STRING NOT NULL, error_category STRING
        ) CLUSTER BY (event_time) COMMENT 'Vector Search queries (no query text)' {DELTA}""",

        # ---------------------------------------------------------------- knowledge
        f"""CREATE TABLE IF NOT EXISTS {{kn}}.source_document (
            doc_id STRING NOT NULL, title STRING NOT NULL, doc_type STRING,
            product_line STRING COMMENT 'std | ltd | life | accident | critical_illness | dental | all',
            version STRING, source_uri STRING, content_hash STRING NOT NULL,
            classification STRING NOT NULL COMMENT 'synthetic in the POC',
            status STRING NOT NULL, ingested_at TIMESTAMP NOT NULL
        ) COMMENT 'Knowledge-base documents (examiner guidance; synthetic)' {DELTA}""",
        f"""CREATE TABLE IF NOT EXISTS {{kn}}.document_chunk (
            chunk_id STRING NOT NULL, doc_id STRING NOT NULL, chunk_index INT NOT NULL,
            text STRING NOT NULL, char_count INT,
            embedding ARRAY<FLOAT> COMMENT 'Self-managed embedding computed through Unity Gateway',
            embedding_model_key STRING, embedding_model_service STRING, embedding_dimensions INT,
            content_hash STRING NOT NULL, embedded_at TIMESTAMP,
            CONSTRAINT document_chunk_pk PRIMARY KEY (chunk_id)
        ) COMMENT 'Embedded chunks: source of the Delta Sync vector index' {DELTA}""",

        # ---------------------------------------------------------------- views
        """CREATE OR REPLACE VIEW {mkt}.v_model_catalog_current
        COMMENT 'Current version of each model in the catalog' AS
        SELECT model_key, model_service, task, family, provider, roles, status,
               embedding_dimensions, verified_on, notes, valid_from
        FROM {mkt}.model_catalog WHERE is_current""",
        """CREATE OR REPLACE VIEW {obs}.v_gateway_usage
        COMMENT 'system.ai_gateway.usage with Lighthouse request tags as columns' AS
        SELECT request_id AS gateway_request_id, event_time, CAST(event_time AS DATE) AS event_date,
               request_tags['lh_event_id'] AS event_id,
               request_tags['lh_use_case'] AS use_case_id,
               request_tags['lh_purpose'] AS purpose,
               request_tags['lh_environment'] AS environment,
               coalesce(service_name, destination_name) AS model_service, destination_model,
               api_type, requester, status_code, latency_ms, time_to_first_byte_ms,
               input_tokens, output_tokens, total_tokens
        FROM system.ai_gateway.usage""",
        """CREATE OR REPLACE VIEW {obs}.v_model_usage_daily
        COMMENT 'Calls, errors, latency and tokens per day, use case and model (gateway values preferred)' AS
        SELECT coalesce(i.event_date, g.event_date) AS event_date,
               coalesce(i.use_case_id, g.use_case_id, 'unattributed') AS use_case_id,
               coalesce(i.model_key, 'unknown') AS model_key,
               coalesce(i.model_service, g.model_service) AS model_service,
               coalesce(i.purpose, g.purpose) AS purpose,
               count(*) AS calls,
               sum(CASE WHEN coalesce(g.status_code, CASE WHEN i.status = 'ok' THEN 200 ELSE 500 END) >= 400
                        THEN 1 ELSE 0 END) AS errors,
               percentile_approx(coalesce(g.latency_ms, i.latency_ms), 0.5) AS latency_p50_ms,
               percentile_approx(coalesce(g.latency_ms, i.latency_ms), 0.95) AS latency_p95_ms,
               sum(coalesce(g.input_tokens, i.input_tokens)) AS input_tokens,
               sum(coalesce(g.output_tokens, i.output_tokens)) AS output_tokens,
               sum(CASE WHEN g.gateway_request_id IS NOT NULL THEN 1 ELSE 0 END) AS gateway_confirmed
        FROM {obs}.model_invocation i
        FULL OUTER JOIN {obs}.v_gateway_usage g ON g.event_id = i.event_id
        GROUP BY 1, 2, 3, 4, 5""",
        """CREATE OR REPLACE VIEW {obs}.v_eval_trend
        COMMENT 'Evaluation score and verdict over time per use case and prompt version' AS
        SELECT use_case_id, prompt_name, prompt_version, model_key, judge_model_key,
               dataset_version, combined_score, verdict, finished_at
        FROM {obs}.eval_run""",
    )


def downgrade() -> None:
    run_sql(
        "DROP VIEW IF EXISTS {obs}.v_eval_trend",
        "DROP VIEW IF EXISTS {obs}.v_model_usage_daily",
        "DROP VIEW IF EXISTS {obs}.v_gateway_usage",
        "DROP VIEW IF EXISTS {mkt}.v_model_catalog_current",
        "DROP TABLE IF EXISTS {kn}.document_chunk",
        "DROP TABLE IF EXISTS {kn}.source_document",
        "DROP TABLE IF EXISTS {obs}.vector_query",
        "DROP TABLE IF EXISTS {obs}.eval_case_result",
        "DROP TABLE IF EXISTS {obs}.eval_run",
        "DROP TABLE IF EXISTS {obs}.model_invocation",
        "DROP TABLE IF EXISTS {mkt}.release_decision",
        "DROP TABLE IF EXISTS {mkt}.use_case",
        "DROP TABLE IF EXISTS {mkt}.model_catalog",
        # schema_migration_audit and the version table stay: they record the downgrade itself.
        "DROP SCHEMA IF EXISTS {kn}",
        "DROP SCHEMA IF EXISTS {obs}",
    )

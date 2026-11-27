"""Local copy of gateway usage with all request tags; activity on model_invocation

Revision ID: 0004_gateway_usage_copy
Revises: 0003_model_cost
Create Date: 2026-11-27

Why: every query on system.ai_gateway.usage costs ~2 s of planning however small the data
(measured 2026-11-27: 2.6 s on the system table vs 0.4 s on the same rows in our own Delta
table), and every usage/cost view and dashboard chart paid it. <obs>.gateway_usage keeps an
incremental copy (GatewayUsageSync: MERGE of the last 48 h by request ID, run by the
lighthouse_platform_usage_sync job, notebook 04 and `lighthouse platform usage sync`), and the
views read the copy. The copy also turns every Lighthouse request tag into a column, including
the new lh_activity, lh_run_id and lh_model_key, so calls no application recorded (LLM judges
inside MLflow evaluation) are still attributed.

model_invocation.activity says what kind of work made the call (app, try, eval, demo,
explorer, verify, notebook); existing rows are backfilled from their use case.
Downgrade restores the views on the system table (0001/0003 definitions) and drops the copy.
"""
import importlib.util
from pathlib import Path

from lighthouse_poc.platform.migrations import column_exists, run_sql

revision = "0004_gateway_usage_copy"
down_revision = "0003_model_cost"
branch_labels = None
depends_on = None

DELTA = ("TBLPROPERTIES ('delta.columnMapping.mode' = 'name', "
         "'delta.enableChangeDataFeed' = 'true')")

GATEWAY_COPY_MERGE = """MERGE INTO {obs}.gateway_usage t
USING (
  SELECT request_id AS gateway_request_id, event_time,
         request_tags['lh_event_id'] AS event_id, request_tags['lh_use_case'] AS use_case_id,
         request_tags['lh_purpose'] AS purpose, request_tags['lh_environment'] AS environment,
         request_tags['lh_activity'] AS activity, request_tags['lh_prompt_version'] AS prompt_version,
         request_tags['lh_run_id'] AS run_id, request_tags['lh_model_key'] AS model_key,
         coalesce(service_name, destination_name) AS model_service, destination_model, api_type,
         requester, CAST(status_code AS INT) AS status_code, CAST(latency_ms AS BIGINT) AS latency_ms,
         CAST(time_to_first_byte_ms AS BIGINT) AS time_to_first_byte_ms,
         CAST(input_tokens AS BIGINT) AS input_tokens, CAST(output_tokens AS BIGINT) AS output_tokens,
         CAST(total_tokens AS BIGINT) AS total_tokens
  FROM system.ai_gateway.usage
) s ON t.gateway_request_id = s.gateway_request_id
WHEN NOT MATCHED THEN INSERT (gateway_request_id, event_time, event_id, use_case_id, purpose,
  environment, activity, prompt_version, run_id, model_key, model_service, destination_model,
  api_type, requester, status_code, latency_ms, time_to_first_byte_ms, input_tokens,
  output_tokens, total_tokens, synced_at)
VALUES (s.gateway_request_id, s.event_time, s.event_id, s.use_case_id, s.purpose, s.environment,
  s.activity, s.prompt_version, s.run_id, s.model_key, s.model_service, s.destination_model,
  s.api_type, s.requester, s.status_code, s.latency_ms, s.time_to_first_byte_ms, s.input_tokens,
  s.output_tokens, s.total_tokens, current_timestamp())"""

ACTIVITY_BACKFILL = """UPDATE {obs}.model_invocation SET activity = CASE use_case_id
  WHEN 'platform_setup_verification' THEN 'verify'
  WHEN 'claim_settlement_demo' THEN 'demo'
  WHEN 'knowledge_search_explorer' THEN 'explorer'
  ELSE 'app' END
WHERE activity IS NULL"""


def upgrade() -> None:
    if not column_exists("obs", "model_invocation", "activity"):
        run_sql("ALTER TABLE {obs}.model_invocation ADD COLUMNS ("
                "activity STRING COMMENT 'app | try | eval | demo | explorer | verify | notebook')")
    run_sql(
        ACTIVITY_BACKFILL,
        f"""CREATE TABLE IF NOT EXISTS {{obs}}.gateway_usage (
            gateway_request_id STRING NOT NULL COMMENT 'system.ai_gateway.usage.request_id',
            event_time TIMESTAMP NOT NULL,
            event_date DATE GENERATED ALWAYS AS (CAST(event_time AS DATE)),
            event_id STRING COMMENT 'request tag lh_event_id (join key to model_invocation)',
            use_case_id STRING, purpose STRING, environment STRING, activity STRING,
            prompt_version STRING, run_id STRING COMMENT 'request tag lh_run_id (MLflow run)',
            model_key STRING COMMENT 'request tag lh_model_key (logical model name)',
            model_service STRING, destination_model STRING, api_type STRING, requester STRING,
            status_code INT, latency_ms BIGINT, time_to_first_byte_ms BIGINT,
            input_tokens BIGINT, output_tokens BIGINT, total_tokens BIGINT,
            synced_at TIMESTAMP NOT NULL
        ) CLUSTER BY (event_time)
        COMMENT 'Incremental copy of system.ai_gateway.usage with Lighthouse request tags as columns'
        {DELTA}""",
        GATEWAY_COPY_MERGE,                              # data migration: initial backfill
        """CREATE OR REPLACE VIEW {obs}.v_gateway_usage
        COMMENT 'Gateway requests (local copy of system.ai_gateway.usage) with Lighthouse tags' AS
        SELECT gateway_request_id, event_time, event_date, event_id, use_case_id, purpose,
               environment, activity, prompt_version, run_id, model_key, model_service,
               destination_model, api_type, requester, status_code, latency_ms,
               time_to_first_byte_ms, input_tokens, output_tokens, total_tokens
        FROM {obs}.gateway_usage""",
        """CREATE OR REPLACE VIEW {obs}.v_model_call_cost
        COMMENT 'One row per model call: our event + gateway record, tokens, latency, status and cost' AS
        WITH service_key AS (
            SELECT model_service, min(model_key) AS model_key, min(task) AS task
            FROM {mkt}.v_model_catalog_current GROUP BY model_service
        ), calls AS (
            SELECT coalesce(i.event_time, g.event_time) AS event_time,
                   coalesce(i.event_id, g.event_id) AS event_id,
                   g.gateway_request_id,
                   coalesce(i.environment, g.environment) AS environment,
                   coalesce(i.use_case_id, g.use_case_id, 'unattributed') AS use_case_id,
                   coalesce(i.purpose, g.purpose, 'unknown') AS purpose,
                   coalesce(i.activity, g.activity, 'unknown') AS activity,
                   coalesce(i.model_key, g.model_key) AS recorded_model_key,
                   coalesce(i.model_service, g.model_service) AS model_service,
                   g.destination_model, g.api_type, i.task AS recorded_task,
                   i.prompt_name, coalesce(i.prompt_version, g.prompt_version) AS prompt_version,
                   coalesce(i.run_id, g.run_id) AS run_id, i.trace_id, i.case_id,
                   coalesce(g.status_code, CASE WHEN i.status = 'ok' THEN 200 ELSE 500 END) AS status_code,
                   i.error_category,
                   coalesce(g.latency_ms, i.latency_ms) AS latency_ms,
                   coalesce(g.input_tokens, i.input_tokens, 0) AS input_tokens,
                   coalesce(g.output_tokens, i.output_tokens, 0) AS output_tokens,
                   g.gateway_request_id IS NOT NULL AS gateway_confirmed,
                   i.event_id IS NOT NULL AS app_recorded
            FROM {obs}.model_invocation i
            FULL OUTER JOIN {obs}.v_gateway_usage g ON g.event_id = i.event_id
        )
        SELECT c.event_time, CAST(c.event_time AS DATE) AS event_date,
               date_trunc('HOUR', c.event_time) AS event_hour,
               c.event_id, c.gateway_request_id, c.environment, c.use_case_id, c.purpose, c.activity,
               coalesce(c.recorded_model_key, k.model_key, 'unknown') AS model_key,
               c.model_service, c.destination_model, c.api_type,
               coalesce(c.recorded_task, k.task,
                        CASE WHEN c.api_type LIKE '%embedding%' THEN 'embeddings' ELSE 'chat' END) AS task,
               c.prompt_name, c.prompt_version, c.run_id, c.trace_id, c.case_id,
               CASE WHEN c.status_code >= 400 THEN 'error' ELSE 'ok' END AS status,
               CASE WHEN c.status_code >= 400 THEN 1 ELSE 0 END AS is_error,
               c.status_code, c.error_category, c.latency_ms,
               c.input_tokens, c.output_tokens, c.input_tokens + c.output_tokens AS total_tokens,
               p.currency, p.price_source,
               p.input_usd_per_1m_tokens, p.output_usd_per_1m_tokens,
               c.input_tokens * p.input_usd_per_1m_tokens / 1e6 AS input_cost_usd,
               c.output_tokens * p.output_usd_per_1m_tokens / 1e6 AS output_cost_usd,
               (c.input_tokens * p.input_usd_per_1m_tokens
                + c.output_tokens * p.output_usd_per_1m_tokens) / 1e6 AS total_cost_usd,
               p.model_key IS NOT NULL AS priced,
               c.gateway_confirmed, c.app_recorded
        FROM calls c
        LEFT JOIN service_key k ON k.model_service = c.model_service
        LEFT JOIN {mkt}.v_model_price p
          ON p.model_key = coalesce(c.recorded_model_key, k.model_key)
         AND CAST(c.event_time AS DATE) >= p.effective_from
         AND (p.effective_to IS NULL OR CAST(c.event_time AS DATE) < p.effective_to)""",
        """CREATE OR REPLACE VIEW {obs}.v_model_cost_daily
        COMMENT 'Calls, errors, latency, tokens and cost per day, use case, activity, model and purpose' AS
        SELECT event_date, use_case_id, activity, model_key, model_service, purpose,
               count(*) AS calls, sum(is_error) AS errors,
               percentile_approx(latency_ms, 0.5) AS latency_p50_ms,
               percentile_approx(latency_ms, 0.95) AS latency_p95_ms,
               sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
               sum(input_cost_usd) AS input_cost_usd, sum(output_cost_usd) AS output_cost_usd,
               sum(total_cost_usd) AS total_cost_usd,
               sum(CASE WHEN priced THEN 0 ELSE 1 END) AS unpriced_calls
        FROM {obs}.v_model_call_cost
        GROUP BY event_date, use_case_id, activity, model_key, model_service, purpose""",
    )


def _previous_revision(module_file: str):
    """Load an earlier revision file (revisions are not a package) to reuse its view SQL."""
    spec = importlib.util.spec_from_file_location(module_file, Path(__file__).with_name(f"{module_file}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def downgrade() -> None:
    # 1. the 0001 view on the system table (columns the 0003 views use)
    run_sql("""CREATE OR REPLACE VIEW {obs}.v_gateway_usage
        COMMENT 'system.ai_gateway.usage with Lighthouse request tags as columns' AS
        SELECT request_id AS gateway_request_id, event_time, CAST(event_time AS DATE) AS event_date,
               request_tags['lh_event_id'] AS event_id,
               request_tags['lh_use_case'] AS use_case_id,
               request_tags['lh_purpose'] AS purpose,
               request_tags['lh_environment'] AS environment,
               coalesce(service_name, destination_name) AS model_service, destination_model,
               api_type, requester, status_code, latency_ms, time_to_first_byte_ms,
               input_tokens, output_tokens, total_tokens
        FROM system.ai_gateway.usage""")
    # 2. the 0003 cost views (its upgrade is re-runnable: CREATE IF NOT EXISTS / OR REPLACE)
    _previous_revision("0003_model_cost").upgrade()
    # 3. drop what 0004 added
    run_sql("DROP TABLE IF EXISTS {obs}.gateway_usage",
            "ALTER TABLE {obs}.model_invocation DROP COLUMNS IF EXISTS (activity)")

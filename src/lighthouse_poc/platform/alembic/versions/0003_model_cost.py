"""Rate card and cost views: model_price, v_model_price, v_model_call_cost, v_model_cost_daily

Revision ID: 0003_model_cost
Revises: 0002_invocation_served_model
Create Date: 2026-11-27

Cost = tokens x the price in force on the call's date. Prices are reference data mirrored from
config/model_pricing.yaml (RateCardSync), so this revision only creates the structure.

v_model_call_cost is the call-grain source of the observability dashboard: our events
(model_invocation) joined with the gateway's record (system.ai_gateway.usage via
v_gateway_usage) on the request tag lh_event_id. Gateway values win for tokens, latency and
status. Calls the gateway saw but no application recorded (other workloads in the workspace,
or events of another schema prefix) are kept as use case 'unattributed'; their model key is
looked up from the model service in the catalog. Downgrade drops these objects; the rate card
is restored by the next sync (the YAML is the source).
"""
from lighthouse_poc.platform.migrations import run_sql

revision = "0003_model_cost"
down_revision = "0002_invocation_served_model"
branch_labels = None
depends_on = None

DELTA = ("TBLPROPERTIES ('delta.columnMapping.mode' = 'name', "
         "'delta.enableChangeDataFeed' = 'true')")


def upgrade() -> None:
    run_sql(
        f"""CREATE TABLE IF NOT EXISTS {{mkt}}.model_price (
            model_key STRING NOT NULL COMMENT 'Logical model key (model_catalog.model_key)',
            effective_from DATE NOT NULL COMMENT 'First day the rate applies',
            currency STRING NOT NULL,
            input_usd_per_1m_tokens DOUBLE NOT NULL COMMENT 'Price per 1M input (prompt) tokens',
            output_usd_per_1m_tokens DOUBLE NOT NULL COMMENT 'Price per 1M output (completion) tokens',
            price_source STRING NOT NULL COMMENT 'illustrative | list | contract',
            notes STRING, synced_at TIMESTAMP NOT NULL, synced_by STRING
        ) CLUSTER BY (model_key)
        COMMENT 'Rate card (effective-dated), mirrored from config/model_pricing.yaml'
        {DELTA}""",
        """CREATE OR REPLACE VIEW {mkt}.v_model_price
        COMMENT 'Rate card with validity periods: effective_to is the next rate of the same model' AS
        SELECT model_key, effective_from,
               lead(effective_from) OVER (PARTITION BY model_key ORDER BY effective_from) AS effective_to,
               currency, input_usd_per_1m_tokens, output_usd_per_1m_tokens, price_source, notes
        FROM {mkt}.model_price""",
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
                   i.model_key AS recorded_model_key,
                   coalesce(i.model_service, g.model_service) AS model_service,
                   g.destination_model, g.api_type, i.task AS recorded_task,
                   i.prompt_name, i.prompt_version, i.run_id, i.trace_id, i.case_id,
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
               c.event_id, c.gateway_request_id, c.environment, c.use_case_id, c.purpose,
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
        COMMENT 'Calls, errors, latency, tokens and cost per day, use case, model and purpose' AS
        SELECT event_date, use_case_id, model_key, model_service, purpose,
               count(*) AS calls, sum(is_error) AS errors,
               percentile_approx(latency_ms, 0.5) AS latency_p50_ms,
               percentile_approx(latency_ms, 0.95) AS latency_p95_ms,
               sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
               sum(input_cost_usd) AS input_cost_usd, sum(output_cost_usd) AS output_cost_usd,
               sum(total_cost_usd) AS total_cost_usd,
               sum(CASE WHEN priced THEN 0 ELSE 1 END) AS unpriced_calls
        FROM {obs}.v_model_call_cost
        GROUP BY event_date, use_case_id, model_key, model_service, purpose""",
    )


def downgrade() -> None:
    run_sql(
        "DROP VIEW IF EXISTS {obs}.v_model_cost_daily",
        "DROP VIEW IF EXISTS {obs}.v_model_call_cost",
        "DROP VIEW IF EXISTS {mkt}.v_model_price",
        "DROP TABLE IF EXISTS {mkt}.model_price",
    )

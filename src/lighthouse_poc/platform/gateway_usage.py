"""Incremental copy of system.ai_gateway.usage into <obs>.gateway_usage (revision 0004).

Reading the system table costs ~2 s of planning per query regardless of size, so the views and
the dashboard read a local Delta copy instead. The sync MERGEs the rows of the last `lookback`
hours before the newest copied row, keyed by request ID: idempotent, cheap, and it picks up
rows that land late in the system table. Runs from the lighthouse_platform_usage_sync job,
notebook 04 and `lighthouse platform usage sync`.
"""
from __future__ import annotations

import logging
from typing import Any

from .naming import PlatformNames

COLUMNS = ("gateway_request_id, event_time, event_id, use_case_id, purpose, environment, activity, "
           "prompt_version, run_id, model_key, model_service, destination_model, api_type, "
           "requester, status_code, latency_ms, time_to_first_byte_ms, input_tokens, "
           "output_tokens, total_tokens")


class GatewayUsageSync:
    def __init__(self, names: PlatformNames, logger: logging.Logger, lookback_hours: int = 48):
        if lookback_hours < 1:
            raise ValueError("lookback_hours must be >= 1")
        self._table = f"{names.catalog}.{names.observability}.gateway_usage"
        self._logger, self._lookback = logger, int(lookback_hours)

    def statement(self) -> str:
        tag = lambda name: f"request_tags['{name}']"   # noqa: E731
        return f"""MERGE INTO {self._table} t
USING (
  SELECT request_id AS gateway_request_id, event_time,
         {tag('lh_event_id')} AS event_id, {tag('lh_use_case')} AS use_case_id,
         {tag('lh_purpose')} AS purpose, {tag('lh_environment')} AS environment,
         {tag('lh_activity')} AS activity, {tag('lh_prompt_version')} AS prompt_version,
         {tag('lh_run_id')} AS run_id, {tag('lh_model_key')} AS model_key,
         coalesce(service_name, destination_name) AS model_service, destination_model, api_type,
         requester, CAST(status_code AS INT) AS status_code, CAST(latency_ms AS BIGINT) AS latency_ms,
         CAST(time_to_first_byte_ms AS BIGINT) AS time_to_first_byte_ms,
         CAST(input_tokens AS BIGINT) AS input_tokens, CAST(output_tokens AS BIGINT) AS output_tokens,
         CAST(total_tokens AS BIGINT) AS total_tokens
  FROM system.ai_gateway.usage
  WHERE event_time >= coalesce((SELECT max(event_time) FROM {self._table}),
                               TIMESTAMP '1970-01-01 00:00:00') - INTERVAL {self._lookback} HOURS
) s ON t.gateway_request_id = s.gateway_request_id
WHEN NOT MATCHED THEN INSERT ({COLUMNS}, synced_at)
VALUES ({', '.join('s.' + c.strip() for c in COLUMNS.split(','))}, current_timestamp())"""

    def sync(self, engine: Any) -> int:
        from sqlalchemy import text
        with engine.connect() as connection:
            result = connection.execute(text(self.statement().replace(":", "\\:")))
        inserted = max(int(getattr(result, "rowcount", 0) or 0), 0)
        self._logger.info("gateway_usage_synced", extra={
            "table": self._table, "rows_inserted": inserted, "lookback_hours": self._lookback,
            "kpi_name": "gateway_usage_rows_synced", "kpi_value": inserted})
        return inserted

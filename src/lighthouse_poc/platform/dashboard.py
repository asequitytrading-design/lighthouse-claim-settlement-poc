"""AI/BI (Lakeview) observability dashboard of the AI Marketplace, built as code (notebook 04).

Why one call-grain dataset: every chart on a page aggregates the same `calls` dataset
(v_model_call_cost), so the page filters apply to all of them and clicking a bar or line point
cross-filters every other chart on the page. That is the drill-down: overview -> click a model ->
everything narrows to that model; the "Model drill-down" page then shows its individual calls.

Percentiles cannot be re-aggregated after filtering, so latency p50/p95/p99 come from a separate
dataset aggregated in SQL (last 30 days, page filters on model/use case apply to it too).

The dashboard JSON (`serialized_dashboard`) is generated here so it is versioned and reviewed like
code. Editing in the UI is fine for exploring, but the next deploy overwrites the draft: change
this module (or copy the dashboard) to keep a change.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .naming import PlatformNames

GRID_WIDTH = 6      # Lakeview canvas columns


# ---------------------------------------------------------------- datasets
@dataclass(frozen=True)
class Dataset:
    name: str
    display_name: str
    sql: str
    purpose: str

    def as_json(self) -> dict:
        return {"name": self.name, "displayName": self.display_name,
                "queryLines": [line + "\n" for line in self.sql.strip().splitlines()]}


def observability_datasets(names: PlatformNames, days: int = 90) -> list[Dataset]:
    obs = f"{names.catalog}.{names.observability}"
    mkt = f"{names.catalog}.{names.marketplace}"
    return [
        Dataset("calls", "Model calls (call grain)", f"""
SELECT event_time, event_date, event_hour, environment, use_case_id, activity, purpose,
       model_key, model_service, task, prompt_version, run_id, case_id,
       status, is_error, status_code, error_category, latency_ms,
       input_tokens, output_tokens, total_tokens,
       input_cost_usd, output_cost_usd, total_cost_usd, price_source, priced,
       gateway_confirmed, app_recorded
FROM {obs}.v_model_call_cost
WHERE event_time >= current_timestamp() - INTERVAL {int(days)} DAYS""",
                "One row per model call (our event joined with system.ai_gateway.usage). Every "
                "chart aggregates it, so filters and click-to-filter work across the page."),
        Dataset("latency", "Latency percentiles by model", f"""
SELECT model_key, use_case_id, activity, purpose,
       count(*) AS calls,
       percentile_approx(latency_ms, 0.5) AS latency_p50_ms,
       percentile_approx(latency_ms, 0.95) AS latency_p95_ms,
       percentile_approx(latency_ms, 0.99) AS latency_p99_ms,
       max(latency_ms) AS latency_max_ms
FROM {obs}.v_model_call_cost
WHERE event_time >= current_timestamp() - INTERVAL 30 DAYS AND latency_ms IS NOT NULL
GROUP BY model_key, use_case_id, activity, purpose""",
                "Percentiles must be computed in SQL (they cannot be summed after filtering)."),
        Dataset("rate_card", "Rate card in force", f"""
SELECT model_key, effective_from, effective_to, currency,
       input_usd_per_1m_tokens, output_usd_per_1m_tokens, price_source
FROM {mkt}.v_model_price""",
                "The prices behind every cost figure (illustrative until contract rates are set)."),
    ]


# ---------------------------------------------------------------- widget helpers
def dim(column: str) -> dict:
    return {"name": column, "expression": f"`{column}`"}


def agg(function: str, column: str) -> dict:
    if column == "*":
        return {"name": "count(*)", "expression": "COUNT(`*`)"}
    return {"name": f"{function.lower()}({column})", "expression": f"{function.upper()}(`{column}`)"}


@dataclass
class Widget:
    name: str
    title: str
    kind: str                       # bar | line | counter | table | filter-* | text
    position: tuple[int, int, int, int]  # x, y, width, height
    dataset: str | None = None
    fields: list[dict] = field(default_factory=list)
    encodings: dict = field(default_factory=dict)
    disaggregated: bool = False
    filter_bindings: list[tuple[str, str]] = field(default_factory=list)  # (dataset, column)
    lines: list[str] = field(default_factory=list)
    extra_spec: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        x, y, width, height = self.position
        position = {"x": x, "y": y, "width": width, "height": height}
        if self.kind == "text":
            return {"widget": {"name": self.name, "multilineTextboxSpec": {"lines": self.lines}},
                    "position": position}
        frame = {"showTitle": True, "title": self.title}
        if self.kind.startswith("filter-"):
            queries, encoded = [], []
            for dataset, column in self.filter_bindings:
                query_name = f"filter_{self.name}_{dataset}"
                queries.append({"name": query_name, "query": {
                    "datasetName": dataset, "disaggregated": False,
                    "fields": [dim(column), {"name": f"{column}_associativity",
                                             "expression": "COUNT_IF(`associative_filter_predicate_group`)"}]}})
                encoded.append({"fieldName": column, "displayName": self.title, "queryName": query_name})
            spec = {"version": 2, "widgetType": self.kind, "encodings": {"fields": encoded}, "frame": frame}
            return {"widget": {"name": self.name, "queries": queries, "spec": spec}, "position": position}
        version = {"counter": 2, "table": 1}.get(self.kind, 3)
        spec = {"version": version, "widgetType": self.kind, "encodings": self.encodings,
                "frame": frame, **self.extra_spec}
        query = {"datasetName": self.dataset, "fields": self.fields, "disaggregated": self.disaggregated}
        return {"widget": {"name": self.name, "queries": [{"name": "main_query", "query": query}],
                           "spec": spec}, "position": position}


def counter(name: str, title: str, x: int, y: int, measure: dict, dataset: str = "calls") -> Widget:
    return Widget(name, title, "counter", (x, y, 1, 2), dataset, [measure],
                  {"value": {"fieldName": measure["name"], "displayName": title}})


def chart(name: str, title: str, kind: str, position: tuple[int, int, int, int], x_column: str,
          measure: dict, measure_title: str, color: str | None = None, temporal: bool = False,
          dataset: str = "calls") -> Widget:
    fields = [dim(x_column), measure] + ([dim(color)] if color else [])
    encodings = {
        "x": {"fieldName": x_column, "displayName": x_column.replace("_", " "),
              "scale": {"type": "temporal" if temporal else "categorical"}},
        "y": {"fieldName": measure["name"], "displayName": measure_title,
              "scale": {"type": "quantitative"}},
    }
    if color:
        encodings["color"] = {"fieldName": color, "displayName": color.replace("_", " "),
                              "scale": {"type": "categorical"}}
    return Widget(name, title, kind, position, dataset, fields, encodings)


_TABLE_TYPES = {"datetime": ("datetime", "datetime"), "int": ("integer", "number"),
                "float": ("float", "number"), "bool": ("boolean", "boolean"), "str": ("string", "string")}


def table(name: str, title: str, position: tuple[int, int, int, int], dataset: str,
          columns: list[tuple[str, str]]) -> Widget:
    """Detail table (rows as they are, no aggregation). columns: (column, type)."""
    encoded = []
    for order, (column, kind) in enumerate(columns):
        data_type, display = _TABLE_TYPES[kind]
        spec = {"fieldName": column, "displayName": column, "title": column, "type": data_type,
                "displayAs": display, "visible": True, "order": order, "allowSearch": kind == "str",
                "alignContent": "left" if display in ("string", "boolean") else "right",
                "booleanValues": ["false", "true"], "allowHTML": False, "highlightLinks": False,
                "useMonospaceFont": False, "preserveWhitespace": False,
                "imageUrlTemplate": "{{ @ }}", "imageTitleTemplate": "{{ @ }}", "imageWidth": "",
                "imageHeight": "", "linkUrlTemplate": "{{ @ }}", "linkTextTemplate": "{{ @ }}",
                "linkTitleTemplate": "{{ @ }}", "linkOpenInNewTab": True}
        if data_type == "float":
            spec["numberFormat"] = "0.000000"
        elif data_type == "integer":
            spec["numberFormat"] = "0"
        elif data_type == "datetime":
            spec["dateTimeFormat"] = "YYYY-MM-DD HH:mm:ss"
        encoded.append(spec)
    return Widget(name, title, "table", position, dataset, [dim(c) for c, _ in columns],
                  {"columns": encoded}, disaggregated=True,
                  extra_spec={"invisibleColumns": [], "allowHTMLByDefault": False,
                              "itemsPerPage": 25, "paginationSize": "default",
                              "condensed": True, "withRowNumber": False})


def page_filters(prefix: str, y: int, model_kind: str = "filter-multi-select",
                 with_latency: bool = True) -> list[Widget]:
    """Date range, model, use case, activity, purpose and status filters for one page. A filter
    applies to every dataset it is bound to: `calls` always, `latency` where percentiles show."""
    def both(column: str) -> list[tuple[str, str]]:
        return [("calls", column)] + ([("latency", column)] if with_latency else [])
    return [
        Widget(f"{prefix}_f_date", "Date range", "filter-date-range-picker", (0, y, 2, 2),
               filter_bindings=[("calls", "event_time")]),
        Widget(f"{prefix}_f_model", "Model", model_kind, (2, y, 1, 2), filter_bindings=both("model_key")),
        Widget(f"{prefix}_f_use_case", "Use case", "filter-multi-select", (3, y, 1, 2),
               filter_bindings=both("use_case_id")),
        Widget(f"{prefix}_f_activity", "Activity", "filter-multi-select", (4, y, 1, 2),
               filter_bindings=both("activity")),
        Widget(f"{prefix}_f_purpose", "Purpose", "filter-multi-select", (5, y, 1, 1),
               filter_bindings=both("purpose")),
        Widget(f"{prefix}_f_status", "Status", "filter-multi-select", (5, y + 1, 1, 1),
               filter_bindings=[("calls", "status")]),
    ]


def kpi_row(prefix: str, y: int) -> list[Widget]:
    return [
        counter(f"{prefix}_k_calls", "Calls", 0, y, agg("count", "*")),
        counter(f"{prefix}_k_errors", "Errors", 1, y, agg("sum", "is_error")),
        counter(f"{prefix}_k_in", "Input tokens", 2, y, agg("sum", "input_tokens")),
        counter(f"{prefix}_k_out", "Output tokens", 3, y, agg("sum", "output_tokens")),
        counter(f"{prefix}_k_cost", "Cost (USD)", 4, y, agg("sum", "total_cost_usd")),
        counter(f"{prefix}_k_latency", "Avg latency (ms)", 5, y, agg("avg", "latency_ms")),
    ]


# ---------------------------------------------------------------- pages
@dataclass
class Page:
    name: str
    display_name: str
    widgets: list[Widget]

    def as_json(self) -> dict:
        return {"name": self.name, "displayName": self.display_name, "pageType": "PAGE_TYPE_CANVAS",
                "layout": [w.as_json() for w in self.widgets]}


INTRO = [
    "## AI Marketplace observability\n",
    "Every model call routed through **Unity Gateway**: usage, latency, tokens and cost per model. "
    "Source: `v_model_call_cost` (our events joined with `system.ai_gateway.usage`).\n",
    "\n",
    "**Activity** separates application runs (`app`, `try`, `eval`) from demos (`demo`, "
    "`explorer`, `verify`, `notebook`). "
    "**Drill down:** use the filters, or click a bar or point to filter every chart on the page "
    "(click again to clear). The **Model drill-down** page lists individual calls. "
    "Cost uses the rate card below: *illustrative* rates until contract prices are set. "
    "Gateway rows arrive with a delay.\n",
]

DRILL_INTRO = [
    "## Model drill-down\n",
    "Pick one model (or click a chart), then narrow by date, use case and purpose. "
    "The table at the bottom lists each call: status code, latency, tokens and cost.\n",
]


def overview_page() -> Page:
    return Page("overview", "Overview", [
        Widget("ov_intro", "", "text", (0, 0, 6, 3), lines=INTRO),
        *page_filters("ov", 3),
        *kpi_row("ov", 5),
        chart("ov_calls_daily", "Calls per day by model", "line", (0, 7, 3, 6),
              "event_date", agg("count", "*"), "Calls", color="model_key", temporal=True),
        chart("ov_calls_status", "Calls by model and status", "bar", (3, 7, 3, 6),
              "model_key", agg("count", "*"), "Calls", color="status"),
        chart("ov_in_tokens", "Input tokens by model", "bar", (0, 13, 2, 6),
              "model_key", agg("sum", "input_tokens"), "Input tokens"),
        chart("ov_out_tokens", "Output tokens by model", "bar", (2, 13, 2, 6),
              "model_key", agg("sum", "output_tokens"), "Output tokens"),
        chart("ov_latency", "Average latency by model (ms)", "bar", (4, 13, 2, 6),
              "model_key", agg("avg", "latency_ms"), "Avg latency (ms)"),
        chart("ov_in_cost", "Input token cost by model (USD)", "bar", (0, 19, 2, 6),
              "model_key", agg("sum", "input_cost_usd"), "Input cost (USD)"),
        chart("ov_out_cost", "Output token cost by model (USD)", "bar", (2, 19, 2, 6),
              "model_key", agg("sum", "output_cost_usd"), "Output cost (USD)"),
        chart("ov_cost_daily", "Cost per day by model (USD)", "line", (4, 19, 2, 6),
              "event_date", agg("sum", "total_cost_usd"), "Cost (USD)", color="model_key",
              temporal=True),
        table("ov_latency_pct", "Latency percentiles, last 30 days (ms)", (0, 25, 3, 6), "latency",
              [("model_key", "str"), ("use_case_id", "str"), ("activity", "str"),
               ("purpose", "str"), ("calls", "int"),
               ("latency_p50_ms", "int"), ("latency_p95_ms", "int"), ("latency_p99_ms", "int"),
               ("latency_max_ms", "int")]),
        table("ov_rate_card", "Rate card (USD per 1M tokens)", (3, 25, 3, 6), "rate_card",
              [("model_key", "str"), ("effective_from", "datetime"), ("effective_to", "datetime"),
               ("input_usd_per_1m_tokens", "float"), ("output_usd_per_1m_tokens", "float"),
               ("price_source", "str")]),
    ])


def drilldown_page() -> Page:
    return Page("model_drilldown", "Model drill-down", [
        Widget("dd_intro", "", "text", (0, 0, 6, 2), lines=DRILL_INTRO),
        *page_filters("dd", 2, model_kind="filter-single-select", with_latency=False),
        *kpi_row("dd", 4),
        chart("dd_cost_purpose", "Cost by purpose (USD)", "bar", (0, 6, 2, 6),
              "purpose", agg("sum", "total_cost_usd"), "Cost (USD)", color="model_key"),
        chart("dd_calls_use_case", "Calls by use case and activity", "bar", (2, 6, 2, 6),
              "use_case_id", agg("count", "*"), "Calls", color="activity"),
        chart("dd_errors_code", "Calls by status code", "bar", (4, 6, 2, 6),
              "status_code", agg("count", "*"), "Calls", color="model_key"),
        chart("dd_tokens_hour", "Input tokens per hour", "line", (0, 12, 3, 6),
              "event_hour", agg("sum", "input_tokens"), "Input tokens", color="model_key",
              temporal=True),
        chart("dd_latency_hour", "Max latency per hour (ms)", "line", (3, 12, 3, 6),
              "event_hour", agg("max", "latency_ms"), "Max latency (ms)", color="model_key",
              temporal=True),
        table("dd_calls", "Calls (newest first in the source; sort by clicking a header)",
              (0, 18, 6, 9), "calls",
              [("event_time", "datetime"), ("model_key", "str"), ("use_case_id", "str"),
               ("activity", "str"), ("purpose", "str"), ("status", "str"), ("status_code", "int"),
               ("error_category", "str"), ("latency_ms", "int"), ("input_tokens", "int"),
               ("output_tokens", "int"), ("total_cost_usd", "float"), ("prompt_version", "str"),
               ("case_id", "str"), ("run_id", "str"), ("gateway_confirmed", "bool"),
               ("app_recorded", "bool")]),
    ])


class ObservabilityDashboard:
    """Datasets + pages -> the serialized Lakeview dashboard."""

    def __init__(self, names: PlatformNames, days: int = 90):
        self.names = names
        self.datasets = observability_datasets(names, days)
        self.pages = [overview_page(), drilldown_page()]

    def as_dict(self) -> dict:
        return {"datasets": [d.as_json() for d in self.datasets],
                "pages": [p.as_json() for p in self.pages]}

    def serialized(self) -> str:
        return json.dumps(self.as_dict())

    def widget_rows(self) -> list[dict[str, str]]:
        """A readable inventory: page, widget, type, dataset, what it shows."""
        rows = []
        for page in self.pages:
            for w in page.widgets:
                shows = ", ".join(f["name"] for f in w.fields) or \
                    ", ".join(f"{d}.{c}" for d, c in w.filter_bindings) or "markdown"
                rows.append({"page": page.display_name, "widget": w.title or w.name,
                             "type": w.kind, "dataset": w.dataset or "-", "fields": shows,
                             "x,y,w,h": ",".join(map(str, w.position))})
        return rows


# ---------------------------------------------------------------- deployment
@dataclass(frozen=True)
class DeployedDashboard:
    dashboard_id: str
    path: str
    draft_url: str
    published_url: str | None
    created: bool


class DashboardPublisher:
    """Creates or updates the dashboard in a workspace folder (idempotent), then publishes it."""

    def __init__(self, workspace_client: Any, logger: logging.Logger):
        self._w, self._logger = workspace_client, logger

    def _existing_id(self, path: str) -> str | None:
        from databricks.sdk.errors import NotFound
        try:
            return self._w.workspace.get_status(path).resource_id
        except NotFound:
            return None

    def deploy(self, display_name: str, parent_path: str, serialized: str, warehouse_id: str,
               publish: bool = True) -> DeployedDashboard:
        from databricks.sdk.service.dashboards import Dashboard
        if "/" in display_name:
            raise ValueError("Dashboard display name cannot contain '/'")
        parent = parent_path.rstrip("/")
        path = f"{parent}/{display_name}.lvdash.json"
        self._w.workspace.mkdirs(parent)
        existing = self._existing_id(path)
        body = Dashboard(display_name=display_name, serialized_dashboard=serialized,
                         warehouse_id=warehouse_id, parent_path=None if existing else parent)
        if existing:
            dashboard = self._w.lakeview.update(existing, dashboard=body)
        else:
            dashboard = self._w.lakeview.create(dashboard=body)
        host = self._w.config.host.rstrip("/")
        published = None
        if publish:
            self._w.lakeview.publish(dashboard.dashboard_id, embed_credentials=True,
                                     warehouse_id=warehouse_id)
            published = f"{host}/dashboardsv3/{dashboard.dashboard_id}/published"
        self._logger.info("dashboard_deployed", extra={
            "dashboard_id": dashboard.dashboard_id, "dashboard_created": existing is None,
            "published": publish, "kpi_name": "dashboard_deployments", "kpi_value": 1})
        return DeployedDashboard(dashboard.dashboard_id, path, f"{host}/dashboardsv3/{dashboard.dashboard_id}",
                                 published, existing is None)

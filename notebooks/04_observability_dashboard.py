# Databricks notebook source
# MAGIC %md
# MAGIC # AI Marketplace observability dashboard
# MAGIC
# MAGIC This notebook builds an **AI/BI dashboard** (Lakeview) of every model call routed through Unity
# MAGIC Gateway: **usage, latency, input/output tokens and input/output cost per model**, with filters and
# MAGIC drill-down. Run it top to bottom; each step says what it does, why, and what you should see.
# MAGIC
# MAGIC | Step | What happens | Writes? |
# MAGIC |---|---|---|
# MAGIC | 0 | Install packages, choose catalog / schema prefix / dashboard folder | no |
# MAGIC | 1 | Bring the platform schemas to the newest revision (adds the rate card and cost views) | schemas |
# MAGIC | 2 | Load the rate card: price per 1M input and output tokens per model | `model_price` |
# MAGIC | 3 | *Optional*: send a little synthetic traffic so the charts have data | `model_invocation` |
# MAGIC | 4 | Look at the source view `v_model_call_cost` and how cost is calculated | no |
# MAGIC | 5 | Look at the three dashboard datasets (the SQL behind every chart) | no |
# MAGIC | 6 | Look at the layout: pages, filters, counters, charts, tables | no |
# MAGIC | 7 | Create (or update) the dashboard and publish it | dashboard |
# MAGIC | 8 | Use it: filters, click-to-drill-down, the drill-down page; change it; troubleshoot | no |
# MAGIC
# MAGIC **Before you start**
# MAGIC - Run `02_ai_marketplace_platform_setup` once with the same schema prefix (it creates the schemas
# MAGIC   and the first usage events).
# MAGIC - Attach **Serverless** compute (top right, *Connect*). A SQL warehouse is also needed: the
# MAGIC   dashboard runs its queries on one (Free Edition has *Serverless Starter Warehouse*).
# MAGIC
# MAGIC **Where the logic lives:** `src/lighthouse_poc/platform/dashboard.py` (datasets and layout as code),
# MAGIC `pricing.py` (rate card), `sample_traffic.py`, and the migration
# MAGIC `alembic/versions/0003_model_cost.py`. This notebook calls them and shows what they produce.
# MAGIC Synthetic data only.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0. Install packages and choose where things go
# MAGIC The first cell installs the libraries the platform code needs, then Python restarts (that is
# MAGIC expected: wait for it to finish before running the next cell).

# COMMAND ----------

# MAGIC %pip install --quiet "databricks-sdk>=0.40" "alembic>=1.13,<2" "databricks-sqlalchemy>=2.0.8,<3" "openai>=1.0" "pyyaml>=6"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Widgets appear at the top of the notebook. The defaults are fine for a personal dev run:
# MAGIC
# MAGIC | Widget | Meaning |
# MAGIC |---|---|
# MAGIC | `catalog` | Unity Catalog catalog (`workspace` on Free Edition) |
# MAGIC | `schema_prefix` | `dev_<you>_` for your own schemas, empty for the shared ones |
# MAGIC | `warehouse_id` | SQL warehouse for migrations and the dashboard; empty = first available |
# MAGIC | `dashboard_folder` | workspace folder for the dashboard file |
# MAGIC | `sample_traffic_rounds` | 0 = no demo calls; each round = ~18 real gateway calls with synthetic text |
# MAGIC | `publish` | `true` also publishes (viewers see a stable version; the draft stays editable) |

# COMMAND ----------

import os
import sys
from pathlib import Path

PROJECT = Path.cwd().parent  # notebooks run with cwd = the notebook folder
sys.path.insert(0, str(PROJECT / "src"))

from databricks.sdk import WorkspaceClient

from lighthouse_poc.config import load_profile
from lighthouse_poc.logging_config import configure_logging
from lighthouse_poc.platform.naming import PlatformNames, dev_prefix

logger = configure_logging()
w = WorkspaceClient()
me = w.current_user.me().user_name

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema_prefix", dev_prefix(me))
dbutils.widgets.text("warehouse_id", "")
dbutils.widgets.text("dashboard_folder", f"/Workspace/Users/{me}/lighthouse")
dbutils.widgets.dropdown("sample_traffic_rounds", "0", ["0", "1", "2", "3", "5"])
dbutils.widgets.dropdown("publish", "true", ["true", "false"])

os.environ["LIGHTHOUSE_CATALOG"] = dbutils.widgets.get("catalog").strip()
os.environ["LIGHTHOUSE_PLATFORM_PREFIX"] = dbutils.widgets.get("schema_prefix").strip()
os.environ["DATABRICKS_WAREHOUSE_ID"] = dbutils.widgets.get("warehouse_id").strip()
os.environ.setdefault("DATABRICKS_HOST", w.config.host)

profile = load_profile("databricks")
names = PlatformNames(profile.platform.catalog, profile.platform.schema_prefix)
MKT = f"{names.catalog}.{names.marketplace}"
OBS = f"{names.catalog}.{names.observability}"
DASHBOARD_NAME = f"AI Marketplace observability ({names.prefix or 'shared'})"
print("Schemas:  ", MKT, OBS, sep="\n  ")
print("Dashboard:", f"{dbutils.widgets.get('dashboard_folder')}/{DASHBOARD_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. Bring the schemas to the newest revision
# MAGIC Cost needs two things the first revisions did not have: a **rate card** table (`model_price`)
# MAGIC and **cost views**. Revision `0003_model_cost` adds them. Schema changes only happen through
# MAGIC Alembic revisions (ADR-012), so this step shows the pending SQL first, then applies it.
# MAGIC Nothing pending? It prints *Up to date*; running it again is safe.
# MAGIC
# MAGIC What 0003 creates:
# MAGIC - `ai_marketplace.model_price`: price per 1M input / output tokens, valid from a date
# MAGIC - `ai_marketplace.v_model_price`: adds `effective_to` (the next price's start date)
# MAGIC - `ai_observability.v_model_call_cost`: **one row per call** with tokens, latency, status and cost
# MAGIC - `ai_observability.v_model_cost_daily`: the same, summed per day, use case, model and purpose

# COMMAND ----------

from lighthouse_poc.platform.connection import resolve_warehouse
from lighthouse_poc.platform.migrations import SchemaMigrator
from lighthouse_poc.platform.runtime import warehouse_engine

WAREHOUSE_ID = resolve_warehouse(w, profile.platform.warehouse_id)
engine = warehouse_engine(profile, names, logger, warehouse_id=WAREHOUSE_ID)
migrator = SchemaMigrator(names, logger, engine=engine)

applied, head = migrator.current(), migrator.heads()
print("Applied:", ", ".join(applied) or "base", "| head:", ", ".join(head))
if not applied:
    raise RuntimeError("These schemas were never migrated: run notebook 02 with this prefix first.")
if applied != head:
    print(migrator.offline_sql(start=applied[0]))
    run_id = migrator.upgrade("head")
    print("Upgraded to", migrator.current(), "| restore points: migration run", run_id)
else:
    print("Up to date: nothing to run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Load the rate card
# MAGIC Cost = tokens × price. Prices live in Git in `config/model_pricing.yaml` (reviewed like code) and
# MAGIC are mirrored into `model_price`: one `MERGE` inserts new rates, updates changed ones and deletes
# MAGIC removed ones. The model catalog is synced in the same step (prices must name catalog models).
# MAGIC
# MAGIC > **The shipped rates are illustrative placeholders** (`price_source = illustrative`), not a
# MAGIC > Databricks quote; Free Edition is not billed per token. Put your contract rates in the YAML
# MAGIC > (`price_source: contract`) and re-run this cell.
# MAGIC
# MAGIC **A price change** is a new row with a later `effective_from`; never edit an old row. Calls before
# MAGIC that date keep the old price, so historical cost does not move.
# MAGIC
# MAGIC The same cell refreshes `gateway_usage`, our copy of `system.ai_gateway.usage`. Reading the system
# MAGIC table costs about 2 s per query, so the views and the dashboard read the copy. The job
# MAGIC `lighthouse-platform-usage-sync` refreshes it every hour.

# COMMAND ----------

from lighthouse_poc.platform.model_catalog import ModelCatalogSync, load_model_catalog
from lighthouse_poc.platform.pricing import RateCardSync, load_rate_card

from lighthouse_poc.platform.gateway_usage import GatewayUsageSync

models = load_model_catalog()
prices = load_rate_card(models=models)
# refresh the local copy of system.ai_gateway.usage the views read (revision 0004)
print("gateway usage rows copied:", GatewayUsageSync(names, logger).sync(engine))
ModelCatalogSync(names, logger).sync(engine, models)
changed = RateCardSync(names, logger).sync(engine, prices)
print(f"{len(prices)} rates in the YAML, {changed} rows changed in {MKT}.model_price")
display(spark.sql(f"SELECT * FROM {MKT}.v_model_price ORDER BY model_key, effective_from"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 (optional). Send some synthetic traffic
# MAGIC A fresh setup has only the verification probes, so most charts would show one bar. With
# MAGIC `sample_traffic_rounds` > 0 this cell sends **real** calls through Unity Gateway with short
# MAGIC synthetic case snippets (no real data): per case one *app* call (approved app models in turn),
# MAGIC one *judge* call and one *embedding* call, all tagged with use case `claim_settlement_demo`.
# MAGIC Tokens, latency and errors are real. Errors are expected now and then (rate limits, a
# MAGIC reasoning model returning no text) and are recorded as such: the dashboard should show them.
# MAGIC Leave it at 0 to skip.

# COMMAND ----------

from lighthouse_poc.platform.runtime import catalog_chat_client, catalog_embedding_client
from lighthouse_poc.platform.sample_traffic import SampleTrafficGenerator

rounds = int(dbutils.widgets.get("sample_traffic_rounds"))
if rounds:
    generator = SampleTrafficGenerator(
        models, names, engine, logger,
        chat_factory=lambda m: catalog_chat_client(profile, m, logger),
        embed_factory=lambda m: catalog_embedding_client(profile, m, logger),
        environment="dev")
    print(generator.run(rounds=rounds))
else:
    print("Skipped (sample_traffic_rounds = 0).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4. The source: `v_model_call_cost`
# MAGIC Every chart reads this view. It has **one row per model call** and combines two records of it:
# MAGIC
# MAGIC | Record | Knows | Table |
# MAGIC |---|---|---|
# MAGIC | Ours (the application) | use case, purpose, prompt version, run, synthetic case | `model_invocation` |
# MAGIC | The gateway's | tokens, latency, HTTP status, requester (authoritative) | `system.ai_gateway.usage` |
# MAGIC
# MAGIC They are joined on the request tag `lh_event_id` that our code sends with every call
# MAGIC (`FULL OUTER JOIN`, so calls only one side knows about are kept). Gateway values win.
# MAGIC Then the price in force on the call's date is joined from `v_model_price`:
# MAGIC
# MAGIC ```
# MAGIC input_cost_usd  = input_tokens  × input_usd_per_1m_tokens  / 1,000,000
# MAGIC output_cost_usd = output_tokens × output_usd_per_1m_tokens / 1,000,000
# MAGIC total_cost_usd  = input_cost_usd + output_cost_usd
# MAGIC ```
# MAGIC
# MAGIC Things to know:
# MAGIC - **Gateway rows arrive late** (system tables lag, often 10+ minutes). Until then
# MAGIC   `gateway_confirmed = false` and our own token/latency numbers are used.
# MAGIC - **`use_case_id = 'unattributed'`**: calls the gateway saw that no marketplace application
# MAGIC   recorded (other jobs in the workspace). Their model key is looked up from the model service.
# MAGIC - **Failed calls** (`status = 'error'`, e.g. HTTP 429/503) have 0 tokens and 0 cost.
# MAGIC - **`priced = false`**: no rate for that model on that date (add one to the YAML).

# COMMAND ----------

display(spark.sql(f"""
SELECT event_time, use_case_id, purpose, model_key, status, status_code, latency_ms,
       input_tokens, output_tokens, input_usd_per_1m_tokens, output_usd_per_1m_tokens,
       input_cost_usd, output_cost_usd, total_cost_usd, gateway_confirmed, app_recorded
FROM {OBS}.v_model_call_cost ORDER BY event_time DESC LIMIT 20"""))

# Sanity check: totals per model. These are the numbers the dashboard counters must show
# (all dates, no filters).
display(spark.sql(f"""
SELECT model_key, count(*) AS calls, sum(is_error) AS errors,
       sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
       round(sum(input_cost_usd), 6) AS input_cost_usd, round(sum(output_cost_usd), 6) AS output_cost_usd,
       round(sum(total_cost_usd), 6) AS total_cost_usd, sum(CASE WHEN priced THEN 0 ELSE 1 END) AS unpriced
FROM {OBS}.v_model_call_cost GROUP BY model_key ORDER BY total_cost_usd DESC"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5. The dashboard datasets
# MAGIC A Lakeview dashboard has **datasets** (SQL queries) and **widgets** that aggregate them.
# MAGIC This dashboard uses three:
# MAGIC
# MAGIC | Dataset | Grain | Used by |
# MAGIC |---|---|---|
# MAGIC | `calls` | one row per call (last 90 days) | every counter and chart, the calls table |
# MAGIC | `latency` | model × use case × purpose (last 30 days) | latency percentile table |
# MAGIC | `rate_card` | model × price period | rate card table |
# MAGIC
# MAGIC **Why one call-grain dataset for (almost) everything:** a page filter applies to every widget
# MAGIC that reads the filtered dataset, and clicking a bar **cross-filters** the other widgets on the
# MAGIC page that read the same dataset. Keeping all charts on `calls` makes the whole page drill down
# MAGIC together. The widgets do the aggregation (`COUNT`, `SUM(input_tokens)`, `AVG(latency_ms)`...).
# MAGIC
# MAGIC **Why latency percentiles are separate:** a p95 cannot be computed by adding up smaller
# MAGIC p95s, so they are aggregated in SQL per model / use case / purpose. The model, use case and
# MAGIC purpose filters still apply to them; the date filter does not (fixed 30 days).
# MAGIC
# MAGIC The next cell prints each dataset's SQL and previews its first rows.

# COMMAND ----------

from lighthouse_poc.platform.dashboard import ObservabilityDashboard

dashboard = ObservabilityDashboard(names)
for ds in dashboard.datasets:
    print(f"--- dataset `{ds.name}`: {ds.display_name}\n    {ds.purpose}\n{ds.sql.strip()}\n")
    display(spark.sql(ds.sql).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6. The layout
# MAGIC Two pages on a 6-column grid (`x,y,w,h` below: column, row, width, height):
# MAGIC
# MAGIC **Overview**
# MAGIC 1. Filters: date range, model, use case, purpose, status
# MAGIC 2. Counters: calls, errors, input tokens, output tokens, cost (USD), average latency
# MAGIC 3. Usage: calls per day by model (line); calls by model and status (stacked bar)
# MAGIC 4. Tokens and latency: input tokens, output tokens, average latency, each by model
# MAGIC 5. Cost: input token cost, output token cost (bars by model), cost per day by model (line)
# MAGIC 6. Tables: latency p50/p95/p99/max by model; the rate card behind the cost figures
# MAGIC
# MAGIC **Model drill-down**: the same filters (single model) and counters, then cost by purpose,
# MAGIC calls by use case, calls by HTTP status code, input tokens and max latency per hour, and a
# MAGIC table of **individual calls** (time, status code, latency, tokens, cost, prompt version, case).
# MAGIC
# MAGIC The inventory below is generated from the code, so it always matches what step 7 deploys.

# COMMAND ----------

display(spark.createDataFrame(dashboard.widget_rows()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7. Create or update the dashboard, then publish
# MAGIC `DashboardPublisher.deploy`:
# MAGIC 1. creates the folder if needed;
# MAGIC 2. looks for `<folder>/<name>.lvdash.json`: **updates** it if it exists, otherwise **creates** it
# MAGIC    (re-running never makes duplicates);
# MAGIC 3. attaches the SQL warehouse that will run the dashboard queries;
# MAGIC 4. publishes it (if `publish = true`) with embedded credentials: viewers see data through your
# MAGIC    access, so share it only with people allowed to see these tables.
# MAGIC
# MAGIC > Re-running this step **overwrites the draft** with the version in code. To keep a change you
# MAGIC > made in the UI, either change `dashboard.py`, or *Clone* the dashboard in the UI and edit the copy.

# COMMAND ----------

from lighthouse_poc.platform.dashboard import DashboardPublisher

deployed = DashboardPublisher(w, logger).deploy(
    DASHBOARD_NAME, dbutils.widgets.get("dashboard_folder"), dashboard.serialized(), WAREHOUSE_ID,
    publish=dbutils.widgets.get("publish") == "true")
engine.dispose()

print(("Created" if deployed.created else "Updated"), deployed.path)
links = f'<p><a href="{deployed.draft_url}" target="_blank">Open the draft (editable)</a></p>'
if deployed.published_url:
    links += f'<p><a href="{deployed.published_url}" target="_blank">Open the published dashboard</a></p>'
displayHTML(links)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8. Use the dashboard
# MAGIC
# MAGIC **Open it**: click a link above (opens in a new tab, so this notebook stays open), or
# MAGIC sidebar → **Dashboards** → *AI Marketplace observability (…)*, or **Workspace** → the folder above.
# MAGIC
# MAGIC **Drill down**
# MAGIC | You want to... | Do this |
# MAGIC |---|---|
# MAGIC | Narrow the whole page | Use the filters at the top (date range, model, use case, purpose, status) |
# MAGIC | Focus on one model from a chart | **Click its bar** (or a legend entry / line point): every chart on the page filters to it. Click again (or *Clear*) to undo |
# MAGIC | See the individual calls | Go to the **Model drill-down** page, pick the model, scroll to the *Calls* table; sort by clicking a column header |
# MAGIC | Find failures | Status filter = `error`, then *Calls by status code* (429 = rate limited, 5xx = upstream) |
# MAGIC | See what a number is made of | Hover a chart for exact values; on a widget, the ⋮ menu → *View data* / *Download* |
# MAGIC | See exactly which price was used | The *Rate card* table on the Overview page |
# MAGIC
# MAGIC **Change it**
# MAGIC - Explore freely in the draft (*Edit draft*): add a widget, change a chart type. To keep the change
# MAGIC   for everyone, add it to `src/lighthouse_poc/platform/dashboard.py` and re-run step 7.
# MAGIC - New price: add a row to `config/model_pricing.yaml` → re-run step 2. New model: add it to
# MAGIC   `config/model_catalog.yaml` and give it a rate → step 2.
# MAGIC - Refresh: the dashboard queries live tables, so reload it. Published dashboards cache results
# MAGIC   for a while; use *Refresh* (top right). *Schedule* there sends it by e-mail on a timetable.
# MAGIC
# MAGIC **Troubleshooting**
# MAGIC | Symptom | Cause / fix |
# MAGIC |---|---|
# MAGIC | Charts empty | No calls in the date range: widen it, run notebook 02, or set `sample_traffic_rounds` and re-run step 3 |
# MAGIC | Cost is 0 for a model | No rate for it: add one to `config/model_pricing.yaml`, re-run step 2 (check `unpriced` in step 4) |
# MAGIC | `gateway_confirmed` false | Normal for recent calls: `system.ai_gateway.usage` lags; refresh later |
# MAGIC | `unattributed` use case | Gateway calls not recorded by a marketplace app (other jobs, or another schema prefix's events) |
# MAGIC | A widget shows an error | The draft was edited in the UI and a field no longer exists: re-run step 7 to restore it |
# MAGIC | `TABLE_OR_VIEW_NOT_FOUND ... v_model_call_cost` | Step 1 did not run for this prefix |
# MAGIC | Permission error on `system.ai_gateway.usage` | Only workspace admins can read it by default; ask an admin for `SELECT` |

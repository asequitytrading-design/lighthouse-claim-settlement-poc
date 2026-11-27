# Databricks notebook source
# MAGIC %md
# MAGIC # AI Marketplace platform: setup and verification
# MAGIC
# MAGIC Builds and checks the Unity Catalog data platform of the AI Marketplace
# MAGIC (`docs/ai-marketplace-platform.md`, ADR-012 and ADR-013). **Synthetic data only.**
# MAGIC
# MAGIC | Step | What it proves |
# MAGIC |---|---|
# MAGIC | 1 workspace | identity, catalog and SQL warehouse work |
# MAGIC | 2 migrations | Alembic upgrades the platform schemas to the head revision |
# MAGIC | 3 objects | every table and view exists |
# MAGIC | 4 catalog | `config/model_catalog.yaml` is synced into `ai_marketplace.model_catalog` (type-2 history) and the rate card into `model_price` |
# MAGIC | 5 gateway | each approved/candidate model answers through **Unity Gateway**, with request tags |
# MAGIC | 6 knowledge | synthetic examiner guidance is chunked and embedded through the gateway |
# MAGIC | 7 vectors | Mosaic AI Vector Search endpoint + Delta Sync index answer a gateway-embedded query |
# MAGIC | 8 dashboard | the usage views show today's calls (gateway usage rows arrive later) |
# MAGIC
# MAGIC Run it from the bundle's workspace files (`databricks bundle deploy`) or a Git folder, so
# MAGIC `src/` and `config/` sit next to `notebooks/`. Re-running is safe: every step is idempotent.

# COMMAND ----------

# MAGIC %pip install --quiet "alembic>=1.13,<2" "databricks-sqlalchemy>=2.0.8,<3" "databricks-vectorsearch>=0.57" "openai>=1.0" "pyyaml>=6"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
from pathlib import Path

PROJECT = Path.cwd().parent  # notebooks run with cwd = the notebook folder
sys.path.insert(0, str(PROJECT / "src"))

from databricks.sdk import WorkspaceClient

from lighthouse_poc.config import load_profile
from lighthouse_poc.logging_config import configure_logging
from lighthouse_poc.platform.naming import PlatformNames, dev_prefix

logger = configure_logging()
me = WorkspaceClient().current_user.me().user_name

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema_prefix", dev_prefix(me))      # personal dev schemas; "" for shared
dbutils.widgets.text("warehouse_id", "")                    # empty: first available warehouse
dbutils.widgets.text("vector_search_endpoint", "lighthouse-vs")
dbutils.widgets.dropdown("apply_migrations", "true", ["true", "false"])
dbutils.widgets.dropdown("vector_search", "true", ["true", "false"])
dbutils.widgets.dropdown("environment", "dev", ["dev", "ci", "prod"])

import os
os.environ["LIGHTHOUSE_CATALOG"] = dbutils.widgets.get("catalog").strip()
os.environ["LIGHTHOUSE_PLATFORM_PREFIX"] = dbutils.widgets.get("schema_prefix").strip()
os.environ["DATABRICKS_WAREHOUSE_ID"] = dbutils.widgets.get("warehouse_id").strip()
os.environ["LIGHTHOUSE_VS_ENDPOINT"] = dbutils.widgets.get("vector_search_endpoint").strip()
os.environ.setdefault("DATABRICKS_HOST", WorkspaceClient().config.host)

profile = load_profile("databricks")
names = PlatformNames(profile.platform.catalog, profile.platform.schema_prefix)
print("Schemas:", ", ".join(f"{names.catalog}.{s}" for s in names.schemas))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Migrations: what will run
# MAGIC Revisions live in `src/lighthouse_poc/platform/alembic/versions`. Review the SQL first
# MAGIC (the same as `lighthouse platform db sql`), then the next cell applies it.

# COMMAND ----------

from lighthouse_poc.platform.migrations import SchemaMigrator
from lighthouse_poc.platform.runtime import warehouse_engine

engine = warehouse_engine(profile, names, logger)
migrator = SchemaMigrator(names, logger, engine=engine)
print("Revisions:", " -> ".join(r.revision for r in migrator.revisions()))
print("Applied:  ", ", ".join(migrator.current()) or "base (not migrated yet)")
print(migrator.offline_sql(start=(migrator.current() or [None])[0]) if migrator.current() != migrator.heads()
      else "Up to date: nothing to run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run all verification steps

# COMMAND ----------

from lighthouse_poc.platform.verification import SetupVerifier

report = SetupVerifier(profile, names, engine, logger,
                       environment=dbutils.widgets.get("environment")).run(
    migrate=dbutils.widgets.get("apply_migrations") == "true",
    vectors=dbutils.widgets.get("vector_search") == "true")
display(spark.createDataFrame(report.rows()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Look at the results
# MAGIC The model catalog, today's usage (our events joined with `system.ai_gateway.usage`) and the
# MAGIC migration restore points. Gateway usage rows usually appear after some delay: re-run this
# MAGIC cell later and `gateway_confirmed` fills in.

# COMMAND ----------

mkt, obs = f"{names.catalog}.{names.marketplace}", f"{names.catalog}.{names.observability}"
display(spark.sql(f"SELECT model_key, task, status, roles, model_service, embedding_dimensions, verified_on "
                  f"FROM {mkt}.v_model_catalog_current ORDER BY task, status, model_key"))
display(spark.sql(f"SELECT * FROM {obs}.v_model_usage_daily WHERE event_date >= current_date() - 1 "
                  f"ORDER BY event_date DESC, calls DESC"))
display(spark.sql(f"SELECT migration_run_id, intent, count(*) AS tables, max(recorded_at) AS recorded_at "
                  f"FROM {mkt}.schema_migration_audit GROUP BY 1, 2 ORDER BY recorded_at DESC LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: prove a migration round trip
# MAGIC Downgrade the newest revision (`0003_model_cost`: drops the rate card table and the cost
# MAGIC views), then upgrade again (recreates them) and re-load the rate card from
# MAGIC `config/model_pricing.yaml` (reference data comes back from its source, not from the
# MAGIC migration). Restore points for the downgrade are printed.

# COMMAND ----------

dbutils.widgets.dropdown("demo_round_trip", "false", ["true", "false"])
if dbutils.widgets.get("demo_round_trip") == "true":
    down_run = migrator.downgrade("-1")
    print("After downgrade:", migrator.current())
    print("\n".join(migrator.restore_statements(down_run)))
    migrator.upgrade("head")
    print("After upgrade:  ", migrator.current())
    from lighthouse_poc.platform.model_catalog import load_model_catalog
    from lighthouse_poc.platform.pricing import RateCardSync, load_rate_card
    rates = RateCardSync(names, logger).sync(engine, load_rate_card(models=load_model_catalog()))
    print("Rate card re-loaded:", rates, "rows")

# COMMAND ----------

engine.dispose()
assert report.passed, "Setup NOT verified: see the FAIL rows above"
print("Setup verified.")

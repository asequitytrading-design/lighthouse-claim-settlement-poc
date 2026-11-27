# Databricks notebook source
# MAGIC %md
# MAGIC # Workspace setup for the Lighthouse demo (no laptop CLI needed)
# MAGIC The notebook route of `docs/deployment-runbook.md`: run it from the project folder in the
# MAGIC workspace (so `src/`, `config/` and `data/` sit next to `notebooks/`). Synthetic data only.
# MAGIC
# MAGIC | Step | Cell | What it does | When |
# MAGIC |---|---|---|---|
# MAGIC | 0 | Settings | catalog, schema prefix, SQL warehouse | always |
# MAGIC | 1 | Platform | migrations (schemas, synthetic claims, MCP tool function, app config), model catalog + rates, knowledge base, Vector Search index, end-to-end check | before the first deploy (Vector Search: 10-20 min the first time) |
# MAGIC | 2 | Genie | find or create the claims Genie space; writes the bundle variables file | before the first deploy |
# MAGIC | 3 | App grants | grants the app's service principal read access (Unity Catalog) | **after** the first bundle deploy |
# MAGIC
# MAGIC Every step is idempotent: re-running is safe.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.18,<4" "openai>=1.0" "pyyaml>=6" "alembic>=1.13,<2" "databricks-sqlalchemy>=2.0.8,<3" "databricks-vectorsearch>=0.57"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Step 0: settings
import json
import os
import sys
from pathlib import Path

PROJECT = Path.cwd().parent                       # notebooks run with cwd = the notebook folder
sys.path.insert(0, str(PROJECT / "src"))

from databricks.sdk import WorkspaceClient

from lighthouse_poc.cli import main as lighthouse
from lighthouse_poc.platform.naming import dev_prefix

w = WorkspaceClient()
me = w.current_user.me().user_name

dbutils.widgets.text("catalog", "workspace")                  # a catalog you may create schemas in
dbutils.widgets.text("schema_prefix", dev_prefix(me))         # same default as the bundle's dev target
dbutils.widgets.text("warehouse_id", "")                      # empty: first available SQL warehouse
dbutils.widgets.text("app_name", "lighthouse-demo-dev")
dbutils.widgets.dropdown("vector_search", "true", ["true", "false"])

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("schema_prefix").strip()
WAREHOUSE = dbutils.widgets.get("warehouse_id").strip()
os.environ.setdefault("DATABRICKS_HOST", w.config.host)
COMMON = ["--catalog", CATALOG, "--prefix", PREFIX] + (["--warehouse", WAREHOUSE] if WAREHOUSE else [])
print(f"project {PROJECT}\ncatalog {CATALOG} | prefix {PREFIX!r} | warehouse {WAREHOUSE or '(first available)'}")

# COMMAND ----------

# Step 1: platform (migrations, model catalog, knowledge, vectors, verification)
verify = ["--profile", "databricks", "platform", "verify", *COMMON]
if dbutils.widgets.get("vector_search") == "false":
    verify.append("--no-vectors")
assert lighthouse(verify) == 0, "platform verify failed: see the report above"
assert lighthouse(["--profile", "databricks", "platform", "models", "sync", *COMMON]) == 0

# COMMAND ----------

# Step 2: Genie space + bundle variables (read by `databricks bundle deploy -t dev`)
from lighthouse_poc.config import load_profile
from lighthouse_poc.platform.connection import resolve_warehouse
from lighthouse_poc.platform.genie import GenieSpaces
from lighthouse_poc.platform.naming import PlatformNames
from lighthouse_poc.logging_config import configure_logging

os.environ["LIGHTHOUSE_CATALOG"], os.environ["LIGHTHOUSE_PLATFORM_PREFIX"] = CATALOG, PREFIX
warehouse_id = resolve_warehouse(w, WAREHOUSE or None)
load_profile("databricks")                                    # validates the profile on this workspace
space = GenieSpaces(w, configure_logging()).ensure(PlatformNames(CATALOG, PREFIX), warehouse_id, None)
overrides = {"genie_space_id": space.space_id, "warehouse_id": warehouse_id, "catalog": CATALOG}
target = PROJECT / ".databricks" / "bundle" / "dev" / "variable-overrides.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(overrides, indent=2))
print(f"Genie space {space.title}: {space.space_id} ({'created' if space.created else 'existing'})")
print(f"wrote {target}:\n{json.dumps(overrides, indent=2)}")

# COMMAND ----------

# Step 3 (after the first bundle deploy): grant the app's service principal access
app = w.apps.get(dbutils.widgets.get("app_name").strip())
principal = app.service_principal_client_id
print(f"app {app.name} runs as service principal {principal}")
assert lighthouse(["--profile", "databricks", "platform", "app", "grants", *COMMON,
                   "--principal", principal, "--prompt-schema", f"{PREFIX}lighthouse"]) == 0

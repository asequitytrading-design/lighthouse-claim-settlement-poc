# Databricks notebook source
# MAGIC %md
# MAGIC # Demo app controls: start, switch live settings, stop
# MAGIC Control panel for the Lighthouse demo app (`lighthouse-demo-<target>`, Databricks Apps).
# MAGIC
# MAGIC | Step | Cell | What it does |
# MAGIC |---|---|---|
# MAGIC | 0 | Settings | pick the app, the Lakebase project and the settings to apply |
# MAGIC | 1 | Status | app state and URL, Lakebase compute state, settings stored for the app |
# MAGIC | 2 | Start | enable the Lakebase compute (if needed) and start the app |
# MAGIC | 3 | **Apply settings** | validate and store the settings; the **running** app picks them up within ~60 s, no restart |
# MAGIC | 4 | Stop | with `stop_now = yes`: stop the app and disable the Lakebase compute (Free Edition: after the demo) |
# MAGIC
# MAGIC **Live settings** (table `<catalog>.<prefix>ai_marketplace.app_config`, revision 0006):
# MAGIC
# MAGIC | Setting | Values | Effect |
# MAGIC |---|---|---|
# MAGIC | `session_store` | `sqlite` / `lakebase` | where chats and the semantic cache live; `lakebase` needs the Lakebase compute on (step 2 does it); if Lakebase is unreachable the app falls back to SQLite and says so in its health |
# MAGIC | `default_model` | an app model of the catalog | the copilot's default model (the composer can still pick another) |
# MAGIC | `semantic_cache` | `on` / `off` | answer repeated guidance questions from the cache |
# MAGIC | `cache_threshold` | 0.85 - 0.99 | similarity needed for a cache hit (lower = more hits, more risk) |
# MAGIC | `grounding_check` | `on` / `off` | answerability + verified-quote check on guidance (off only to show the difference) |
# MAGIC | `genie` | `on` / `off` | the portfolio tool (Genie) |
# MAGIC
# MAGIC How it works: the app re-reads this table at most once a minute, only when requests arrive (no
# MAGIC polling while idle), and rebuilds only what changed (store and cache, or the copilot). Check the
# MAGIC result in the app: the sidebar status tooltip, or `<app URL>/api/health` -> `live_settings`.
# MAGIC Settings only: nothing secret is stored. Synthetic data only.

# COMMAND ----------

# MAGIC %pip install --quiet "databricks-sdk>=0.38" "pyyaml>=6"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,0. Settings
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))       # project code next to notebooks/

from databricks.sdk import WorkspaceClient

from lighthouse_poc.platform.app_config import AppConfig, read_sql, upsert_sql
from lighthouse_poc.platform.model_catalog import load_model_catalog
from lighthouse_poc.platform.naming import PlatformNames, dev_prefix

w = WorkspaceClient()
me = w.current_user.me().user_name
APP_MODELS = sorted(m.key for m in load_model_catalog()
                    if "app" in m.roles and m.status in ("approved", "candidate"))
DEFAULTS = AppConfig()

dbutils.widgets.text("app_name", "lighthouse-demo-dev")
dbutils.widgets.text("lakebase_project", "lighthouse-demo")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("platform_prefix", dev_prefix(me))
dbutils.widgets.dropdown("session_store", "lakebase", ["lakebase", "sqlite"])
dbutils.widgets.dropdown("default_model", DEFAULTS.default_model, APP_MODELS)
dbutils.widgets.dropdown("semantic_cache", "on", ["on", "off"])
dbutils.widgets.dropdown("cache_threshold", DEFAULTS.cache_threshold, ["0.90", "0.92", "0.95", "0.97"])
dbutils.widgets.dropdown("grounding_check", "on", ["on", "off"])
dbutils.widgets.dropdown("genie", "on", ["on", "off"])
dbutils.widgets.dropdown("stop_now", "no", ["no", "yes"])     # guards step 4 against "Run all"

W = {k: dbutils.widgets.get(k).strip() for k in ("app_name", "lakebase_project", "catalog", "platform_prefix")}
SETTINGS = {k: dbutils.widgets.get(k) for k in AppConfig().as_dict()}
names = PlatformNames(W["catalog"], W["platform_prefix"])
ENDPOINT = f"/api/2.0/postgres/projects/{W['lakebase_project']}/branches/production/endpoints/primary"
print("App:", W["app_name"], "| settings table:", names.table("app_config"))
print("Settings to apply:", SETTINGS)

# COMMAND ----------

# DBTITLE 1,1. Status
def lakebase_state() -> dict:
    try:
        status = w.api_client.do("GET", ENDPOINT).get("status", {})
        return {"state": status.get("current_state"), "disabled": bool(status.get("disabled"))}
    except Exception as exc:
        return {"state": f"unavailable ({type(exc).__name__})", "disabled": True}


def show_status():
    app = w.apps.get(W["app_name"])
    print(f"App      : {app.name} | {app.app_status.state.value if app.app_status else '?'} | "
          f"compute {app.compute_status.state.value if app.compute_status else '?'}")
    print(f"URL      : {app.url}")
    print(f"Lakebase : {W['lakebase_project']} | {lakebase_state()}")
    try:
        display(spark.sql(read_sql(names, W["app_name"])))
    except Exception as exc:
        print(f"No stored settings yet ({type(exc).__name__}): run notebook 02 / `lighthouse platform db upgrade` for revision 0006")


show_status()

# COMMAND ----------

# DBTITLE 1,2. Start (Lakebase compute if needed, then the app)
import time


def wait(predicate, what, timeout=900):
    started = time.time()
    while not predicate():
        if time.time() - started > timeout:
            raise TimeoutError(f"{what} not ready after {timeout} s")
        time.sleep(10)


if SETTINGS["session_store"] == "lakebase" and lakebase_state()["disabled"]:
    print("enabling the Lakebase compute ...")
    w.api_client.do("PATCH", ENDPOINT, query={"update_mask": "spec.disabled"}, body={"spec": {"disabled": False}})
    wait(lambda: not lakebase_state()["disabled"], "Lakebase")
app = w.apps.get(W["app_name"])
if not app.app_status or app.app_status.state.value != "RUNNING":
    print("starting the app ...")
    w.apps.start(W["app_name"])
    wait(lambda: (w.apps.get(W["app_name"]).app_status.state.value == "RUNNING"), "the app")
show_status()
print("Warm it up before presenting: open the app once, or run `python scripts/demo_env.py up` locally.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Apply settings to the running app
# MAGIC Change the widgets at the top, then run the next cell. Values are validated before anything is
# MAGIC written (unknown models or out-of-range thresholds are refused). The app applies them on its next
# MAGIC request after at most ~60 s: open or refresh the app, then check `live_settings` in its health.
# MAGIC
# MAGIC Tips for the demo: switch `default_model` to show model management live; set `semantic_cache = off`
# MAGIC then `on` and ask the same guidance question twice to show the latency drop; `grounding_check = off`
# MAGIC shows why grounding matters (ask the pet insurance question), then set it back to `on`.

# COMMAND ----------

# DBTITLE 1,3. Apply settings
if SETTINGS["session_store"] == "lakebase" and lakebase_state()["disabled"]:
    print("Note: Lakebase compute is disabled; the app will fall back to SQLite until step 2 enables it.")
statement = upsert_sql(names, W["app_name"], SETTINGS, me, set(APP_MODELS))
spark.sql(statement).collect()
print(f"Stored {len(SETTINGS)} settings for {W['app_name']} as {me}.")
display(spark.sql(read_sql(names, W["app_name"])))
print("The running app applies them within ~60 s on its next request (check /api/health -> live_settings).")

# COMMAND ----------

# DBTITLE 1,4. Stop (app, then Lakebase compute): set stop_now = yes first
if dbutils.widgets.get("stop_now") == "yes":
    w.apps.stop(W["app_name"])
    w.api_client.do("PATCH", ENDPOINT, query={"update_mask": "spec.disabled"}, body={"spec": {"disabled": True}})
    print("Stopped. Set stop_now back to no.")
else:
    print("Not stopping: set the widget stop_now = yes and run this cell to stop the app and Lakebase.")
show_status()

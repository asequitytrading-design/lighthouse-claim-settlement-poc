"""Start or stop the demo environment on Databricks (Free Edition: keep it off when not in use).

  python scripts/demo_env.py status      app and Lakebase state, app URL
  python scripts/demo_env.py up          start Lakebase, start the app, warm it up
  python scripts/demo_env.py down        stop the app, stop Lakebase

Uses your Databricks CLI profile (DATABRICKS_CONFIG_PROFILE, default lighthouse). The app must be
deployed once with the bundle (session_store=lakebase for the demo, see apps/lighthouse_demo/README.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

APP = os.environ.get("LIGHTHOUSE_DEMO_APP", "lighthouse-demo-dev")
INSTANCE = os.environ.get("LIGHTHOUSE_LAKEBASE_INSTANCE", "lighthouse-demo")


def client():
    os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", "lighthouse")
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _endpoint() -> str:
    # Lakebase on Free Edition is an autoscaling project; its compute is the branch endpoint
    return f"/api/2.0/postgres/projects/{INSTANCE}/branches/production/endpoints/primary"


def status(w) -> dict:
    app = w.apps.get(APP)
    try:
        endpoint = w.api_client.do("GET", _endpoint()).get("status", {})
        db_state, db_disabled = endpoint.get("current_state"), bool(endpoint.get("disabled"))
    except Exception as exc:             # project deleted or Lakebase unavailable
        db_state, db_disabled = f"unavailable ({type(exc).__name__})", True
    return {"app": APP, "app_state": getattr(app.app_status.state, "value", None) if app.app_status else None,
            "compute": getattr(app.compute_status.state, "value", None) if app.compute_status else None,
            "url": app.url, "lakebase": INSTANCE, "lakebase_state": db_state, "lakebase_disabled": db_disabled}


def set_lakebase(w, stopped: bool) -> None:
    """Disable (stop) or enable the Lakebase compute. Free Edition does not allow changing the
    auto-suspend timeout or stopping the instance through the database-instance API."""
    w.api_client.do("PATCH", _endpoint(), query={"update_mask": "spec.disabled"},
                    body={"spec": {"disabled": stopped}})


def wait(predicate, what: str, timeout: int = 900) -> None:
    started = time.time()
    while not predicate():
        if time.time() - started > timeout:
            sys.exit(f"timed out waiting for {what}")
        time.sleep(10)


def warmup(w, url: str) -> dict:
    token = w.config.authenticate()["Authorization"]
    request = urllib.request.Request(f"{url}/api/warmup", method="POST", headers={"Authorization": token})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["status", "up", "down"])
    action = parser.parse_args().action
    w = client()
    if action == "up":
        print(f"starting Lakebase {INSTANCE} ...")
        set_lakebase(w, stopped=False)
        wait(lambda: not status(w)["lakebase_disabled"], "Lakebase")
        print(f"starting app {APP} ...")
        w.apps.start(APP)
        wait(lambda: status(w)["app_state"] == "RUNNING", "the app")
        result = warmup(w, status(w)["url"])
        print("warm-up:", "ready" if result["ready"] else "NOT ready",
              {k: v["status"] for k, v in result["steps"].items()})
    elif action == "down":
        print(f"stopping app {APP} ...")
        w.apps.stop(APP)
        print(f"stopping Lakebase {INSTANCE} ...")
        set_lakebase(w, stopped=True)
    print(json.dumps(status(w), indent=2))


if __name__ == "__main__":
    main()

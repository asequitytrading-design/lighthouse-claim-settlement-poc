"""Databricks Apps entry point: serve lighthouse-api (and the built UI) on the app's port.

Databricks Apps runs this from the bundle root with the app's service principal credentials in
the environment (DATABRICKS_HOST, DATABRICKS_CLIENT_ID/SECRET) and the port in
DATABRICKS_APP_PORT. The package is imported from src/ (no install step needed).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

host = os.environ.get("DATABRICKS_HOST", "")
if host and not host.startswith("http"):              # Apps provides the bare hostname
    os.environ["DATABRICKS_HOST"] = f"https://{host}"

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("lighthouse_poc.api.app:create_app", factory=True, host="0.0.0.0",
                port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")), log_level="info")

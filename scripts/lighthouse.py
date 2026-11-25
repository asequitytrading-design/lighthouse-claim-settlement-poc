"""Job entry point for the lighthouse CLI (Databricks spark_python_task or plain python).

Same commands as the installed ``lighthouse`` console script, for runtimes where the package
is not installed (the bundle syncs src/ next to this file):
  python scripts/lighthouse.py --profile databricks eval --prompt 3 --baseline @champion ...
"""
import sys
from pathlib import Path

try:
    PROJECT = Path(__file__).resolve().parents[1]
except NameError:  # some job runtimes execute scripts without __file__
    PROJECT = Path(sys.argv[0]).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lighthouse_poc.cli import main  # noqa: E402

if __name__ == "__main__":
    # Exit only on failure: Databricks serverless runs scripts in an IPython runner where
    # even sys.exit(0) raises SystemExit and marks the task as failed.
    exit_code = main(sys.argv[1:])
    if exit_code:
        sys.exit(exit_code)

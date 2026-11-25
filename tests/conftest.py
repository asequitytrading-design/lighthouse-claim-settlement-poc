"""Keep unit tests hermetic: no writes to a real MLflow server or the repo directory."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="lighthouse-tests-")
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_TMP}/mlflow.db"

import mlflow  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _disable_tracing():
    mlflow.tracing.disable()
    yield
    mlflow.tracing.enable()

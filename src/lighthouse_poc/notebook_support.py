"""Helpers for Databricks notebooks that drive the `lighthouse` CLI (notebook 05).

The notebook runs the same commands a developer runs in a terminal (`lighthouse try`, `eval`,
`promote`), in-process, so the code path, MLflow lineage and usage tracking are identical.
"""
from __future__ import annotations

import io
import json
import shlex
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable

from .platform.model_catalog import load_model_catalog

MODEL_ENV = {"claims-small": "UNITY_GATEWAY_SMALL_MODEL", "claims-medium": "UNITY_GATEWAY_MEDIUM_MODEL",
             "claims-judge": "UNITY_GATEWAY_JUDGE_MODEL"}


@dataclass(frozen=True)
class CliResult:
    command: str
    exit_code: int
    output: str

    @property
    def data(self) -> Any:
        """Parsed JSON output (commands run with --json)."""
        start = self.output.find("{")
        if start < 0:
            raise ValueError(f"No JSON in the output of: {self.command}")
        return json.loads(self.output[start:])


def run_cli(args: list[str], profile: str = "databricks", echo: bool = True,
            main: Callable | None = None) -> CliResult:
    """Run `lighthouse --profile <profile> <args>` and return its exit code and output."""
    if main is None:
        from .cli import main
    argv = ["--profile", profile, *args]
    command = "lighthouse " + " ".join(shlex.quote(a) for a in argv)
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            code = main(argv, out=buffer)
    except SystemExit as exc:          # argparse usage errors
        code = int(exc.code or 0)
    result = CliResult(command, int(code or 0), buffer.getvalue())
    if echo:
        print(f"$ {command}\n{result.output}" + (f"[exit code {result.exit_code}]" if result.exit_code else ""))
    return result


def model_service_env(entries=None) -> dict[str, str]:
    """UNITY_GATEWAY_*_MODEL values from the governed model catalog (what the bundle job passes)."""
    services = {e.key: e.model_service for e in (entries or load_model_catalog())}
    return {env: services[key] for key, env in MODEL_ENV.items() if key in services}


def experiment_url(host: str, experiment_id: str, run_id: str | None = None) -> str:
    base = f"{host.rstrip('/')}/ml/experiments/{experiment_id}"
    return f"{base}/runs/{run_id}" if run_id else base

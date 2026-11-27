"""Seed the local MLflow UI with the Lighthouse building blocks (idempotent; local stack only).

Creates only what is missing and reports every change:
  1. experiment  /lighthouse-ui-sample
  2. AI Gateway endpoints for the Kong routes (kong-claims-small / -medium / -judge); the app
     endpoints' usage tracking points at the experiment so Playground calls become traces
     there (the judge endpoint keeps its own experiment)
  3. prompt claim_intake_triage v1 and v2 (chat prompts, as the Playground saves them),
     alias @candidate -> v2, and both prompts linked to the experiment (Prompts tab)
  4. evaluation dataset claim_intake_triage_golden (11 synthetic cases)
  5. judges on the Judges page: custom code judges (gates/rubric, src/lighthouse_poc/ui_judges.py)
     and LLM judges on gateway:/kong-claims-judge (Guidelines, ExpectationsGuidelines, Safety)

Never sets @champion: approval is a human step (lighthouse promote).
  python scripts/seed_ui_demo.py
"""
import argparse
import json
import sys
from pathlib import Path

try:
    PROJECT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT = Path(sys.argv[0]).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import mlflow  # noqa: E402
import requests  # noqa: E402

from lighthouse_poc.config import load_dotenv  # noqa: E402
from lighthouse_poc.prompt_eval import upsert_eval_dataset  # noqa: E402
from lighthouse_poc.rubric import load_spec  # noqa: E402

SAMPLE = PROJECT / "samples" / "claim_intake_triage"
EXPERIMENT_TAG = "_mlflow_experiment_ids"
ENDPOINTS = {  # name -> (Kong route, pinned model, tracks into the demo experiment)
    "kong-claims-small": ("claims-small", "qwen3.5-4b-32k:latest", True),
    "kong-claims-medium": ("claims-medium", "qwen3.5-9b-32k:latest", True),
    "kong-claims-judge": ("claims-judge", "gemma4-e4b-32k:latest", False),
}


class Gateway:
    def __init__(self, base: str):
        self.base = f"{base}/api/3.0/mlflow/gateway"

    def call(self, method: str, path: str, **kwargs) -> dict:
        response = requests.request(method, f"{self.base}/{path}", timeout=60, **kwargs)
        if not response.ok:
            raise SystemExit(f"gateway {path}: HTTP {response.status_code} {response.text[:300]}")
        return response.json()

    def endpoints(self) -> dict:
        return {e["name"]: e for e in self.call("GET", "endpoints/list").get("endpoints", [])}

    def secrets(self) -> dict:
        return {s["secret_name"]: s for s in self.call("GET", "secrets/list").get("secrets", [])}


def ensure_endpoints(gateway: Gateway, experiment_id: str, kong_key: str | None, log) -> None:
    existing, secrets = gateway.endpoints(), gateway.secrets()
    for name, (route, model, track_here) in ENDPOINTS.items():
        endpoint = existing.get(name)
        if endpoint is None:
            if not kong_key:
                raise SystemExit("KONG_API_KEY is not set (.env); needed to create gateway endpoints")
            secret = secrets.get(name) or gateway.call("POST", "secrets/create", json={
                "secret_name": name, "provider": "openai", "secret_value": {"api_key": kong_key},
                "auth_config": {"api_base": f"http://kong:8000/ai/{route}"}})["secret"]
            definition = gateway.call("POST", "model-definitions/create", json={
                "name": name, "secret_id": secret["secret_id"], "provider": "openai",
                "model_name": model})["model_definition"]
            body = {"name": name, "usage_tracking": True, "model_configs": [{
                "model_definition_id": definition["model_definition_id"],
                "linkage_type": "PRIMARY", "weight": 1}]}
            if track_here:
                body["experiment_id"] = experiment_id
            gateway.call("POST", "endpoints/create", json=body)
            log(f"created endpoint {name} -> Kong /ai/{route} ({model})")
        elif track_here and (endpoint.get("experiment_id") != experiment_id
                             or not endpoint.get("usage_tracking")):
            gateway.call("POST", "endpoints/update", json={
                "endpoint_id": endpoint["endpoint_id"], "usage_tracking": True,
                "experiment_id": experiment_id})
            log(f"endpoint {name}: usage tracking now -> experiment {experiment_id} "
                f"(was {endpoint.get('experiment_id')})")
        else:
            log(f"endpoint {name}: ok")


def chat_template(system_file: str) -> list:
    return [{"role": "system", "content": (SAMPLE / system_file).read_text(encoding="utf-8")},
            {"role": "user", "content": (SAMPLE / "user_prompt.txt").read_text(encoding="utf-8")}]


def link_prompt(client, name: str, experiment_id: str, log) -> None:
    prompt = client.get_prompt(name)
    if prompt is None:
        return
    ids = [i for i in (prompt.tags.get(EXPERIMENT_TAG) or "").strip(",").split(",") if i]
    if experiment_id not in ids:
        client.set_prompt_tag(name, EXPERIMENT_TAG, "," + ",".join(ids + [experiment_id]) + ",")
        log(f"prompt {name}: linked to experiment {experiment_id}")


def ensure_prompts(client, experiment_id: str, log) -> None:
    name = "claim_intake_triage"
    versions = client.search_prompt_versions(name) if client.get_prompt(name) else []
    wanted = [("system_prompt_v1.txt", "v1: sample triage prompt (synthetic)"),
              ("system_prompt_v2.txt", "v2: no time words or commitments in reply_draft; "
                                       "decision_request takes precedence")]
    for system_file, note in wanted[len(versions):]:
        version = mlflow.genai.register_prompt(
            name=name, template=chat_template(system_file), commit_message=note,
            tags={"use_case": "claim_intake_triage", "synthetic": "true"})
        log(f"registered {name} version {version.version}: {note}")
    try:
        client.get_prompt_version_by_alias(name, "candidate")
        log(f"prompt {name}: alias candidate already set")
    except Exception:
        client.set_prompt_alias(name, "candidate", 2)
        log(f"prompt {name}: alias candidate -> 2 (champion stays a human decision)")
    for prompt_name in (name, "claim_settlement_review"):
        link_prompt(client, prompt_name, experiment_id, log)


def ensure_judges(experiment_id: str, log, replace: bool = False) -> None:
    from mlflow.genai.scorers import (ExpectationsGuidelines, Guidelines, Safety, delete_scorer,
                                      list_scorers)

    from lighthouse_poc.ui_judges import CODE_JUDGES
    registered = {s.name for s in list_scorers(experiment_id=experiment_id)}
    judge_model = "gateway:/kong-claims-judge"
    guidelines = load_spec("claim_intake_triage").judges["guidelines"]
    llm_judges = [Guidelines(name="triage_guidelines", guidelines=list(guidelines), model=judge_model),
                  ExpectationsGuidelines(name="triage_case_guidelines", model=judge_model),
                  Safety(name="triage_safety", model=judge_model)]
    for judge in list(CODE_JUDGES) + llm_judges:
        if judge.name in registered and not replace:
            log(f"judge {judge.name}: ok")
            continue
        if judge.name in registered:
            delete_scorer(name=judge.name, experiment_id=experiment_id, version="all")
        judge.register(name=judge.name, experiment_id=experiment_id)
        log(f"{'re-registered' if judge.name in registered else 'registered'} judge {judge.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--experiment", default="/lighthouse-ui-sample")
    parser.add_argument("--replace-judges", action="store_true",
                        help="Delete and re-register the judges (after changing ui_judges.py)")
    args = parser.parse_args()
    if not args.tracking_uri.startswith(("http://localhost", "http://127.0.0.1")):
        raise SystemExit("seed_ui_demo.py is for the local stack only")
    load_dotenv(PROJECT / ".env")
    import os
    # The client also refuses to load registered code judges without this (local stack only;
    # the server runs with the same setting, see docker-compose.yml).
    os.environ.setdefault("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    experiment = mlflow.set_experiment(args.experiment)
    log = lambda message: print(f"- {message}")  # noqa: E731
    print(f"experiment {args.experiment} (id {experiment.experiment_id})")
    ensure_endpoints(Gateway(args.tracking_uri), experiment.experiment_id,
                     os.environ.get("KONG_API_KEY"), log)
    ensure_prompts(client, experiment.experiment_id, log)
    records = [json.loads(line) for line in (SAMPLE / "golden.jsonl").read_text().splitlines()
               if line.strip()]
    upsert_eval_dataset("claim_intake_triage_golden", experiment.experiment_id, records,
                        {"source_file": "golden.jsonl", "synthetic": "true"})
    log(f"dataset claim_intake_triage_golden: {len(records)} records merged")
    ensure_judges(experiment.experiment_id, log, replace=args.replace_judges)
    print(f"Open {args.tracking_uri}/#/experiments/{experiment.experiment_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

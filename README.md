# Lighthouse Claim Settlement POC

A Python-first starter for prompt versioning and evaluation for a synthetic life and health claim settlement preparation assistant. It spans short-term disability, long-term disability, life, accident, critical illness, and dental claims. The assistant organizes evidence for a human examiner; it does not determine coverage, eligibility, diagnosis, or benefit amounts.

## Two ways to demonstrate

- **Local development:** run the open-source MLflow server with Docker Compose and use its actual MLflow UI at `http://localhost:5000`. This supports local prompt/evaluation development with a SQL-backed tracking store. Unity Catalog integrations and Unity Gateway are Databricks workspace capabilities, not part of the local container.
- **Databricks showcase:** use the actual MLflow UI in the workspace for prompt versions and evaluation runs; route model calls through Unity Gateway. Use the synthetic claim data only.
- **Browser concept:** open `app/demo.html` to click through a seeded lifecycle simulator. It borrows familiar MLflow workspace patterns but is not the same UI, does not persist data, and does not invoke models. It exists to explain the POC flow before a live workspace walkthrough.

## Local MLflow

Requirements: Docker with Compose and Python 3.10+. Start Docker Desktop, then run `docker compose up -d` and open `http://localhost:5000`. The service binds to localhost and stores its database/artifacts in named Docker volumes. In a second terminal, create a Python environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/register_prompt.py --name local.claim_settlement_review
python scripts/evaluate.py --prompt-name local.claim_settlement_review --prompt-version 1 --model-service local.mock.claim-settlement-v1 --model-mode mock --scorer-mode deterministic
```

The local mock run exercises prompt loading, inference plumbing, deterministic schema/boundary checks, and MLflow tracking. It does not measure model quality. To run LLM judges, choose `--scorer-mode llm-judges` and configure a judge provider/model; judge credentials and routing are separate from the app model. For Databricks, route the app model through Unity Gateway and explicitly configure/verify the judge route as well.

Stop MLflow with `docker compose down`; named volumes retain the state. Never expose this unauthenticated development server publicly or place real claim data in it.

Use local MLflow to validate prompt registration, versioning, experiment tracking, evaluation datasets, evaluation runs, and trace review. Local behavior and UI can differ from the Databricks-hosted experience, and local records do not sync to a workspace.

## Databricks model calls

Set `DATABRICKS_HOST`, `DATABRICKS_TOKEN` (or configure supported OAuth), and `UNITY_GATEWAY_MODEL` to a permitted fully qualified model service. The starter's Unity Gateway adapter uses the workspace route `/ai-gateway/mlflow/v1`; actual service names and permissions depend on the workspace. Never commit a real token. For local tests without Databricks connectivity, use `scripts/evaluate.py --model-mode mock`; the mock is deterministic wiring help, not evidence of model quality. A local MLflow container does not emulate Unity Gateway.

## Python-first, serverless-ready structure

Business logic lives under `src/lighthouse_poc`. The notebook is an optional interactive walkthrough, not the application architecture or the only execution method. Databricks supports Python script and Python wheel tasks on serverless jobs; a Git-backed script/package is the path toward an MVP. Keep the package modular and the notebook thin. Free Edition availability, quotas, region, package support, and Unity Gateway access must be checked in the actual workspace.

## Included

- `CLAUDE.md`: engineering rules and safety boundary.
- `data/`: six synthetic source cases and matching golden evaluation cases.
- `src/lighthouse_poc/`: prompt registry, evaluation, logging, model adapter, typed exceptions.
- `scripts/evaluate.py`: Python script entry point for a serverless job.
- `notebooks/`: optional guided Databricks walkthrough.
- `docs/`: design, implementation plan/book, demo flow, and KPI contract.
- `app/demo.html`: clickable lifecycle simulator, clearly labeled as simulated.
- `docker-compose.yml`: local open-source MLflow server.

Databricks Free Edition is personal, non-commercial, serverless-only and quota-limited. Use synthetic data only and an organization-approved workspace for client-facing or commercial demonstrations. See [Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations).

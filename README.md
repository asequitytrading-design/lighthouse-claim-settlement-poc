# Lighthouse Claim Settlement POC

A Python-first starter for prompt versioning and evaluation for a synthetic life and health claim settlement preparation assistant. It spans short-term disability, long-term disability, life, accident, critical illness, and dental claims. The assistant organizes evidence for a human examiner; it does not determine coverage, eligibility, diagnosis, or benefit amounts.

## Two ways to demonstrate

- **Local development:** run the open-source MLflow server with Docker Compose and use its actual MLflow UI at `http://localhost:5000`. This supports local prompt/evaluation development with a SQL-backed tracking store. Unity Catalog integrations and Unity Gateway are Databricks workspace capabilities, not part of the local container.
- **Databricks showcase:** use the actual MLflow UI in the workspace for prompt versions and evaluation runs; route model calls through Unity Gateway. Use the synthetic claim data only.
- **Browser concept:** open `app/demo.html` to click through a seeded lifecycle simulator. It borrows familiar MLflow workspace patterns but is not the same UI, does not persist data, and does not invoke models. It exists to explain the POC flow before a live workspace walkthrough.

## Local stack: MLflow + Kong AI gateway + Ollama

Requirements: Docker Desktop (4 GB+ memory recommended), Ollama with the models in `kong/kong.yaml`, Python 3.10+.

```bash
cp .env.example .env            # set KONG_API_KEY and KONG_PG_PASSWORD
docker compose up -d            # MLflow :5000, Kong proxy :8100, Admin API :8101, Manager :8102
docker compose run --rm kong-seed
python3 -m venv .venv && source .venv/bin/activate && python -m pip install -e '.[dev]'
python scripts/register_prompt.py --profile local
python scripts/evaluate.py --profile local --prompt-version 1 --model mock           # plumbing only
python scripts/evaluate.py --profile local --prompt-version 1 --model claims-medium  # qwen3.5-9b via Kong
python -m pytest
```

Environment differences live in `config/profiles/{local,databricks}.yaml`; code uses logical model names (`claims-small`, `claims-medium`, `claims-judge`, `mock`) and the gateway picks the physical model. See [docs/local-development.md](docs/local-development.md) and [docs/decisions.md](docs/decisions.md). The mock validates plumbing only, not model quality. Never expose the local stack or place real claim data in it.

## Databricks model calls

Use `--profile databricks`. Authentication uses databricks-sdk unified auth (runtime identity, OAuth or token); set `UNITY_GATEWAY_{SMALL,MEDIUM,JUDGE}_MODEL` to permitted fully qualified model services. The starter's Unity Gateway adapter uses the workspace route `/ai-gateway/mlflow/v1`; actual service names and permissions depend on the workspace. Never commit a real token. For local tests without Databricks connectivity, use `scripts/evaluate.py --model mock`; the mock is deterministic wiring help, not evidence of model quality. A local MLflow container does not emulate Unity Gateway.

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
- `docker-compose.yml`: local MLflow, Kong OSS gateway (Postgres-backed) and decK seeding.
- `config/profiles/`: environment profiles (local, databricks).
- `kong/kong.yaml`: Kong routes/plugins seed.
- `tests/`: offline unit tests.

Databricks Free Edition is personal, non-commercial, serverless-only and quota-limited. Use synthetic data only and an organization-approved workspace for client-facing or commercial demonstrations. See [Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations).

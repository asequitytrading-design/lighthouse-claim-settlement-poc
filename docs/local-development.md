# Local development with MLflow Docker

1. Install Docker with Compose.
2. From the repository root, run `docker compose up -d`.
3. Open the actual local MLflow UI at `http://localhost:5000`.
4. Develop prompts/evaluation with synthetic data. The Python entry point defaults to the deterministic mock adapter (`--model-mode mock`); it validates the flow but does not measure model quality. Use `MLFLOW_TRACKING_URI=http://localhost:5000` so the client writes runs to this local server. MLflow judges may require a configured judge model; for a fully offline loop, use deterministic/custom scorers instead of model-backed judges.
5. Stop the service with `docker compose down`. Compose named volumes preserve the local database and artifacts.

The compose file runs the official MLflow container on `127.0.0.1:5000`, with a SQLite tracking backend and local artifact volume. Local MLflow supports prompt versioning and evaluation development; evaluation datasets require a SQL-backed tracking server, which SQLite provides. The server is unauthenticated and must remain local. Never use real insurance information or expose it to a network.

Local MLflow does not include the Databricks workspace MLflow environment, Unity Catalog registry integrations, workspace identity, or Unity Gateway. Those are validated in the Databricks stage. For Databricks, use `DATABRICKS_HOST`, secure OAuth/token configuration, and a qualified approved Unity Gateway model service. See the [MLflow Docker guide](https://mlflow.org/docs/latest/docker/), [MLflow prompt guide](https://mlflow.org/docs/latest/prompts), [evaluation datasets](https://mlflow.org/docs/latest/genai/datasets/), and [Unity Gateway quickstart](https://docs.databricks.com/aws/en/unity-gateway/quickstart).

# Claude review and setup handoff

Please review this synthetic-data proof of concept before adding implementation features.

## Review focus

1. Confirm local Docker MLflow setup and the distinction between local open-source MLflow and Databricks workspace capabilities.
2. Review prompt creation/versioning and evaluation lifecycle against MLflow GenAI APIs at the dependency version in `pyproject.toml`.
3. Review the LLM judge setup (`Correctness`, `Safety`, `ExpectationsGuidelines`) and the separately configured judge-model route. Do not assume it is routed through Unity Gateway without validating the provider configuration.
4. Check that all Unity Gateway requests use a qualified model-service name, credentials are never committed, and logs/traces do not expose claim narratives or secrets.
5. Check the insurance safety boundary: the assistant may prepare an examiner handoff but must not decide eligibility, coverage, diagnosis, or benefit amounts.
6. Review whether the Python script can run as a Databricks serverless Git-backed job, keeping the notebook optional.
7. Identify any API/runtime mismatches, missing error handling, or KPI updates needed before the live Databricks walkthrough.

## Local setup order

1. Start Docker Desktop.
2. Run `docker compose up -d` and open `http://localhost:5000`.
3. Create a local Python environment and install the project in editable mode.
4. Register the bundled synthetic prompt in the local MLflow server.
5. Run the golden evaluation with the mock model for plumbing checks. This is not a model-quality result.
6. Only after local review, configure a Databricks workspace, approved Unity Gateway model service, and judge model route.

Use synthetic data only. Do not put client information or credentials in this repository.

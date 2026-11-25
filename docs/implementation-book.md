# Implementation book: local MLflow and Databricks

## 1. Choose the execution environment

Use local Docker MLflow for fast development and UI familiarization. It stores state locally and does not provide Unity Catalog or Unity Gateway. Use Databricks for the real workspace showcase, Unity Catalog-backed capabilities where enabled, and model requests routed through Unity Gateway. Neither environment should receive real insurance claim data in this POC.

## 2. Start local MLflow

Install Docker Desktop or Docker Engine with Compose. From the project root, run `docker compose up -d` and open `http://localhost:5000`. The compose file starts the official MLflow image with a SQLite backend and persistent local artifact/database volumes. Use this UI to inspect prompt versions, experiments, evaluation records and traces supported by the installed MLflow version. This is a development service; it has no authentication and binds only to localhost. Do not expose it to a network.

Local evaluation datasets need a SQL-backed tracking server; this project uses SQLite. The local instance is independent from Databricks and its records do not sync. See [MLflow GenAI datasets](https://mlflow.org/docs/latest/genai/datasets/), [MLflow Docker](https://mlflow.org/docs/latest/docker/), and [MLflow Prompt Registry](https://mlflow.org/docs/latest/prompts).

## 3. Set up Git and Databricks

Create a private GitHub repository and add this project. Clone it into a Databricks Git folder for interactive work. For repeatable jobs, configure the job to read the remote Git repository at a pinned branch/commit. Use synthetic files only. Check workspace MLflow version, Unity Catalog support, region, permissions, and serverless package availability before assuming any feature exists. Databricks supports serverless Python script and Python wheel job tasks as well as notebook tasks. See [Python script tasks](https://docs.databricks.com/aws/en/jobs/tasks/python-script) and [serverless jobs](https://docs.databricks.com/aws/en/jobs/run-serverless-jobs).

## 4. Run Python, with the notebook as an option

The reusable application code belongs under `src/lighthouse_poc`; the notebook is a convenient guided entry point, not a requirement. For the POC, the engineer may run the thin notebook from the Git folder. For a more repeatable and MVP-like path, use `scripts/evaluate.py` as a serverless Python script task from Git, or package a Python wheel job. The entry point accepts explicit prompt name/version, model-service name, experiment, and manifest versions. Pin dependencies compatible with the actual serverless environment; do not assume local versions match workspace runtime.

## 5. Connect through Unity Gateway

Set `DATABRICKS_HOST` to the workspace URL and configure workspace OAuth or a securely supplied `DATABRICKS_TOKEN`. Set the model to a fully qualified Unity Gateway service name such as `system.ai.<approved-service>`. Grant the user or job identity the required catalog/schema access and `EXECUTE` on that service. The adapter calls `https://<workspace>/ai-gateway/mlflow/v1` using the OpenAI-compatible chat API. Confirm the selected model service, supported region, authentication, permissions, and quota in the workspace. The sample model names in the UI are illustrative. See the [Unity Gateway quickstart](https://docs.databricks.com/aws/en/unity-gateway/quickstart) and [migration/client guide](https://docs.databricks.com/aws/en/ai-gateway/migrate-to-unity-gateway).

For local application development, run `MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/evaluate.py --prompt-name local.claim_settlement_review --prompt-version 1 --model-service local.mock.claim-settlement-v1 --model-mode mock` after registering a prompt. The mock validates data flow and output handling only, not model quality. If a scorer requires a judge model, configure one or use deterministic scorers for offline development. For a real model call, use Unity Gateway only with authorized workspace access. Never save tokens in Git, notebooks, prompt text, or logs.

## 6. Register and test prompts

Use a qualified prompt name when the workspace supports the intended registry, such as `main.default.claim_settlement_review`. Register the template with owner, tags, and change note. Treat each version as immutable; a change creates a new version. Test the candidate in the MLflow prompt playground (or the simulator for a guided explanation), then explicitly load the version under evaluation. Compare against the approved baseline. See [MLflow Prompt Registry](https://mlflow.org/docs/latest/prompts) and Databricks [prompt registry](https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/).

## 7. Maintain the golden set and evaluate

The six synthetic cases in `data/golden_eval_dataset.jsonl` cover disability, life, accident, critical illness, and dental claims. Keep stable case IDs, expected behavior, source note, and dataset version. The starter evaluation runs three LLM judges: `Correctness` against expected facts, `Safety`, and `ExpectationsGuidelines` using the domain-specific per-case rubric stored in each golden record. Separate deterministic validation in the assistant enforces schema and the “Not determined” boundary. Set `MLFLOW_GENAI_JUDGE_DEFAULT_MODEL` to an explicitly approved judge model route. MLflow supports gateway-backed judge models. The Unity Gateway app-model route does not automatically configure the judge route: set `MLFLOW_GENAI_JUDGE_DEFAULT_MODEL` to a supported configured judge model, then validate that its provider is governed through Unity Gateway and confirm judge calls in gateway usage records. Do not assume the application model route also governs judge calls. Run the same dataset/model/scorer/judge configuration for baseline and candidate; inspect case-level feedback and judge rationales, not only aggregate scores. LLM judge feedback is review evidence, never the release gate by itself. Record prompt version, model service, dataset version, scorer config, Git commit, and MLflow run ID. See [MLflow gateway-backed judges](https://mlflow.org/docs/latest/genai/governance/ai-gateway/endpoints/query-endpoints/), [MLflow evaluation datasets](https://mlflow.org/docs/latest/genai/datasets/), [built-in LLM judges](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges), and [guidelines judges](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges/guidelines).

## 8. Review and publish

A human product owner/examiner reviews failures, boundary checks, trace content and the exact manifest. Record reviewer identity, rationale, decision, time and run reference. Only then update the agent's approved prompt reference. Do not treat an alias or score alone as an approval. For the POC, the approval record is an explicit workflow step; do not assume a preview/experimental deployment approval feature is enabled.

## 9. Live demo sequence

1. Open the actual Databricks MLflow UI and show the registered prompt and its versions.
2. Open candidate v3, explain its change note, and try it in Playground with a synthetic case.
3. Show the versioned golden dataset and cases.
4. Run evaluation against a Unity Gateway model service; compare candidate and baseline runs.
5. Inspect a case-level result, scorer feedback, and trace. Confirm gateway usage/permissions where accessible.
6. Review the candidate manifest and record approve/reject; explain the distinction between this POC's human release gate and model inference routing.

Use `app/demo.html` before the live workspace walkthrough only as a clickable, seeded simulator. Its controls update mock page state, but the screen is not MLflow and does not write workspace data.

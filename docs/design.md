# POC design: prompt registry and evaluation framework

## Objective and boundaries

Demonstrate prompt authoring/versioning and evaluation for a claims preparation assistant using synthetic life and health claims. The assistant organizes supplied facts and evidence for a human examiner. It never determines coverage, eligibility, exclusions, diagnosis, causation, fraud, entitlement, claim validity, or benefit amounts.

## Capability mapping

| Capability | Databricks/MLflow POC surface | Outcome |
|---|---|---|
| Prompt Registry | MLflow Prompt Registry, prompt versions, tags, aliases, playground | Draft, test, compare, register immutable versions and bind to evaluations |
| Evaluation Framework | MLflow GenAI evaluation, versioned datasets, scorers/judges, run comparisons, traces | Evaluate exact prompt/model/dataset combination; inspect case-level evidence |
| Model Management | Unity Catalog model services and model records | Discover permitted model services, pin selected service/version where available, track lifecycle |
| Observability | MLflow tracing plus Unity Gateway usage/metrics where exposed | Inspect request path, quality, latency, errors, tokens/cost where available |
| AI Gateway | **Unity Gateway selected** | Govern model routing, identity, permissions, policies and usage for POC calls |

The POC implements the first two capabilities. Model requests use Unity Gateway. The HTML is a standalone workflow simulator with seeded content and a familiar MLflow-inspired layout; it is not Databricks MLflow and does not call it. The live demo uses the actual workspace MLflow UI.

## Lifecycle

1. Engineer creates a prompt for a new agent or selects an existing agent, with owner, tags, and a change reason.
2. MLflow records a new immutable prompt version; prior approved version remains available.
3. Engineer uses the playground against a permitted Unity Gateway model service and synthetic cases.
4. Versioned golden data evaluates the candidate and baseline under comparable settings. Deterministic gates check schema and the non-decision rule; MLflow LLM judges assess correctness, safety, and case-specific expectations/guidelines, returning feedback and rationale.
5. Engineer reviews case outputs, hard boundary checks, scorer feedback, traces, and differences.
6. Human approver records a decision tied to exact versions and run evidence.
7. Only an approved prompt reference is made available to the agent; rejection leaves the current approved reference unchanged.

## Architecture and extension path

```text
Local developer: Python package -> local MLflow Docker + SQLite -> mock model adapter
Databricks: Git source -> Python package / optional notebook -> MLflow workspace UI and tracking
                                              -> Unity Gateway -> qualified model service
```

Reusable logic stays in `src/lighthouse_poc`. The notebook is optional, thin, and useful for interactive exploration. `scripts/evaluate.py` is the Python script entry point; Databricks supports serverless Python script and wheel jobs, suitable for the extendable package path. The actual Free Edition workspace may limit packages, job features, region support, or quotas; verify before implementation.

The local open-source MLflow server offers an actual local MLflow interface for development, prompt management and evaluation. Use a SQL backend (SQLite is configured for this local POC). Local records do not sync to Databricks; Unity Catalog-integrated registry behavior and Unity Gateway are workspace features. Keep the provider adapter injectable so local evaluations can use a deterministic mock and workspace runs use Unity Gateway.

## Release manifest

Bind each decision to application/Git version, prompt name and version, model service, dataset name and version, scorer names/config, MLflow run ID, reviewer, rationale, and timestamp. An alias is a convenient pointer, not the approval record by itself.

## Data and logging

Use the six synthetic cases only. Traces and logs should avoid full narratives, sensitive health/death details, personal identifiers, credentials, and policy text. Log case ID, prompt/model/dataset/app version, run/trace ID, duration, and error category.

## Databricks references

[MLflow Prompt Registry](https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/), [GenAI evaluation](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/eval-examples), [Unity Gateway quickstart](https://docs.databricks.com/aws/en/unity-gateway/quickstart), [Python script jobs](https://docs.databricks.com/aws/en/jobs/tasks/python-script), [serverless jobs](https://docs.databricks.com/aws/en/jobs/run-serverless-jobs). Feature availability varies by workspace, region, permissions, and runtime.

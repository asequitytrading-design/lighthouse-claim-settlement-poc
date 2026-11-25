# Implementation plan

| Phase | Work | Evidence |
|---|---|---|
| 0. Local environment | Start Docker MLflow, confirm UI, run against synthetic data and mock model | Reproducible local setup; no secret or claim data |
| 1. Databricks readiness | Confirm workspace MLflow version, Unity Catalog, Unity Gateway service access, region, serverless packages and quotas | Environment note; authorized model service |
| 2. Package foundation | Keep domain code in `src`; establish structured logging, typed errors, config/provider adapters, KPI definitions | Python-first package; notebook optional |
| 3. Prompt Registry | Create a prompt for a new or existing agent; create immutable versions, tags/change notes; test and manage candidate/approved/archived lifecycle | Prompt history, agent binding, candidate artifact |
| 4. Golden evaluation | Load six cases, version dataset, run deterministic gates plus Correctness, Safety, and per-case ExpectationsGuidelines LLM judges; compare runs and inspect rationales | Per-case judge feedback, hard-check results and traces |
| 5. Review gate | Record reviewer/rationale/decision and exact release manifest | Auditable approve/reject action |
| 6. Serverless execution | Run package as Python script or wheel job; retain notebook for exploration if useful | Reproducible Git-based run |
| 7. Demo | Use actual Databricks MLflow UI for live evidence; use HTML only for guided concept walkthrough | Clear separation of simulated and real results |

Databricks supports serverless Python script and Python wheel job tasks. Free Edition is serverless-only and quota-limited, so verify enabled capabilities and actual package versions. Unity Gateway is the selected POC model route; validate region, service identity, model permissions, and any usage limits. Start with a small six-case run. Use synthetic examples only.

## Completion criteria

The six cases run with traceable outputs; schema and non-decision checks are visible; exact prompt/model/dataset/scorer versions are reproducible; reviewer can approve or reject; each feature/fix has KPI coverage; the live workspace demo distinguishes actual evidence from the standalone simulator.

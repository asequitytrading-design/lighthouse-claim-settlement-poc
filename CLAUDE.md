# Lighthouse POC engineering guide

## Objective

Build an extendable Python-first proof of concept for prompt authoring/versioning and evaluation for a claim-settlement preparation assistant. Engineers create prompt versions, test with approved models, evaluate app/prompt/model combinations against a versioned golden set, inspect evidence, and record human approval before publishing.

## Scope and safety

Synthetic cases cover short-term disability, long-term disability, life, accident, critical illness, and dental. The assistant summarizes supplied facts, lists evidence received, flags case-specific missing/conflicting items, and drafts a handoff for a human claims examiner. It must never decide coverage, eligibility, exclusions, diagnosis, causation, fraud, entitlement, claim validity, or benefit amount. No real policyholder/client data is permitted.

## Engineering rules

- Follow OOP and SOLID: focused classes, clear interfaces, dependency injection, and provider adapters.
- Keep reusable business logic in `src/lighthouse_poc`; notebooks are optional thin exploration/demo entry points. Prefer a package and Python script/wheel job path as the POC evolves toward an MVP.
- Environment differences live only in `config/profiles/*.yaml` (see `docs/decisions.md` ADR-001). Code refers to logical model names; Kong (local, OSS + Postgres) and Unity Gateway (Databricks) are reached through the single OpenAI-compatible `GatewayChatClient`. Kong is the organisation's selected AI gateway; Unity Gateway is the Databricks POC route.
- Unity Gateway is the selected Databricks model routing and governance layer. Use qualified model-service names and workspace OAuth/secret-managed credentials. Keep local model calls behind an injectable adapter; use a mock when Databricks access is unavailable.
- Structured logging and typed exception handling are mandatory. Preserve causal chains; do not swallow errors or fabricate success.
- Log synthetic case ID, trace/run ID, prompt/model/dataset/app versions, latency and error category where available. Avoid full claim narratives, health details and secrets in logs.
- Each module, feature, or bug fix adds or updates KPIs in `docs/metrics.md`: name, definition, unit, collection point, baseline/target, owner.
- Validate output JSON and required fields. `benefit_determination` must equal `Not determined`.
- A human examiner makes all settlement decisions. The POC is administrative and non-decisional.
- Never commit credentials or actual client data.
- Keep the standalone HTML clearly marked as a simulator. Live claims about run scores must come from the Databricks MLflow UI.

Update this guide as architecture and conventions settle. Record major decisions in `docs/decisions.md` when introduced.

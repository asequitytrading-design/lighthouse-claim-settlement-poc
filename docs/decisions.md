# Architecture decisions

## ADR-001: Environment profiles with one OpenAI-compatible gateway client (2026-11-25)

**Context.** The POC must validate prompt registry and evaluation locally (Ollama models
behind Kong, the organisation's selected AI gateway) and run unchanged on Databricks
(workspace MLflow, Unity Gateway, Free Edition model services).

**Decision.**
- `config/profiles/<env>.yaml` holds everything environment-specific: tracking URI,
  experiment, prompt namespace, gateway kind/base URL/auth, and a map of **logical model
  names** (`claims-small`, `claims-medium`, `claims-judge`, `mock`) to gateway routes or
  model services. Secrets and workspace values come from `${ENV}` references.
- One `GatewayChatClient` speaks the OpenAI chat-completions API to either gateway.
  Auth is an injected `AuthProvider` (`api_key_header` for Kong, `databricks_sdk`
  unified auth for Databricks), evaluated per request so OAuth tokens can refresh.
- The gateway, not the code, picks the physical model. Kong's `ai-proxy` pins it per route
  (the client sends an empty `model`); on Databricks the profile names the model service.
  Runs record the model that actually served each call (`lighthouse.served_models`).

**Consequences.** Swapping a model is a gateway or profile change. Code and tests run
offline with `mock`. A Databricks profile loads without Unity Gateway variables and fails
with a `ConfigurationError` naming the missing variable only when a model is used.

## ADR-002: Kong OSS in database mode for local development (2026-11-25)

**Decision.** Run Kong 3.9.3 with Postgres 16 (`docker-compose.yml`), seeded from
`kong/kong.yaml` with decK `gateway apply`. Postgres holds the live configuration so
routes and plugins can be edited in Kong Manager or the Admin API and persist.
Plugins: `key-auth` (service), `ai-proxy` (per route, statistics on, payloads off),
`correlation-id` and `file-log` (global).

**Consequences.** Git holds the seed, not necessarily the live state; `deck gateway dump`
captures edits for review. Host ports 8100–8102 avoid clashes with local dev servers.
Kong runs 2 nginx workers to fit Docker Desktop's default 2 GB VM next to MLflow.

## ADR-003: Judge model separate from the application model family (2026-11-25)

**Decision.** Local judge route `claims-judge` → gemma4-e4b; qwen3.5-9b is the alternate.
gpt-oss:20b was ruled out as too heavy for the development machine.

**Consequences.** Judge results remain review evidence only; calibrate against SME labels
(see `docs/metrics.md`, LLM judge agreement) before relying on them.

## ADR-004: LLM judges through the same gateway, credentials off call arguments (2026-11-25)

**Context.** MLflow built-in judges call their model directly using an `openai:/` URI and
`OPENAI_API_BASE`/`OPENAI_API_KEY`. During `mlflow.genai.evaluate`, MLflow auto-traces
OpenAI SDK calls and stores call arguments in trace spans. A first implementation passed
the Kong key per call in `extra_headers` and it was persisted in 14 local trace spans.

**Decision.**
- `JudgeBinding` (src/lighthouse_poc/scorers.py) resolves the profile's judge alias to a
  model URI, base URL, credential and inference params. Provider variables are set only for
  the duration of the evaluation and restored afterwards.
- Kong judge routes declare `upstream_model` equal to the model Kong pins, so a drift between
  the profile and the gateway fails loudly (HTTP 400) instead of mislabelling judge lineage.
- Application gateway credentials live on the SDK client (`default_headers` for Kong,
  callable `api_key` for bearer/OAuth). Only `X-Correlation-ID` is a per-call header.
- Ollama's missing JSON mode is handled in Kong (`pre-function`), not in application code.
- Deterministic gates always run alongside judges; judges never gate alone.

**Remediation performed.** Rotated the local Kong key, deleted the old credential, deleted the
14 affected traces and vacuumed the local MLflow database; verified the old key is absent
from MLflow storage and the new key is absent after a full judge run.

**Consequences.** Judge results are comparable between environments (same scorers, profile
decides the route). Local judge runs are slow (~6 min for 6 cases) because one Ollama host
alternates between app and judge models.

## ADR-005: Databricks deployment via Asset Bundle and GitHub Actions (2026-11-25)

**Decision.**
- `databricks.yml` + `resources/lighthouse.yml` deploy a Unity Catalog schema, an MLflow
  experiment and a two-task serverless job (environment version 5): `register_prompt`
  (idempotent, `--if-changed`) publishes the version as a task value consumed by `evaluate`.
- Scripts run from synced workspace files (Git-backed source, no wheel yet); profile inputs are
  passed as `--env KEY=VALUE` script arguments because job-level environment variables are a
  Beta feature that requires environment version 5+ and may not be enabled on Free Edition.
  Only non-secret values are passed this way; model calls use databricks-sdk unified auth.
- The MLflow registry URI is a profile setting (`databricks-uc` on Databricks).
- Job parameters default to `app_model=mock, scorer_mode=deterministic` so the first run needs
  no model access; judges default to the Databricks-managed judge until Unity Gateway is set up.
- Targets: `dev` (development mode, per-user prefixes) and `ci` (the deploying identity's home
  folder; `/Workspace/Shared` fails `validate --strict` because every user can write there) for
  GitHub Actions. CI authenticates as service principal `lighthouse-ci` with OAuth M2M on Free
  Edition (no account admin, so no OIDC federation policy), uses GitHub OIDC when only the client
  ID is set, and falls back to a token secret; actions are pinned to commit SHAs.

**Open questions to verify in a real workspace.** Unity Gateway availability on Free Edition and
whether it exposes chat completions (the quickstart shows the Responses API); service principal
and federation support on Free Edition; managed-judge availability; schema deletion behaviour
with prompts.

**Consequences.** Offline checks cover schema validity (CLI JSON schema) and bundle/code
consistency (tests/test_databricks_bundle.py); `databricks bundle validate` needs a workspace.

**Verified on Free Edition (2026-11-25, CLI 1.24.0, target dev).** Catalog `workspace`; the
bundle created the schema, experiment and job; serverless environment version 5 installed the
dependencies; prompts registered in Unity Catalog; the mock/deterministic run passed 6/6 with
lineage params, linked prompt version and 6 traces. Fixes found by the first deploy:
- `${workspace.host}` resolves empty when the host comes from a CLI profile; scripts now resolve
  `DATABRICKS_HOST` from databricks-sdk auth (`runtime.load_runtime_profile`).
- Serverless runs scripts in an IPython runner where `sys.exit(0)` raises `SystemExit` and fails
  the task; scripts exit only on failure.
- Unity Catalog has no implicit `@latest` prompt alias (NOT_FOUND), which was misread as a
  missing prompt and created identical versions 1–3; `latest()` now uses
  `search_prompt_versions`. Re-runs report `Unchanged`. The three duplicate versions were
  reduced to version 3 (identical template, no aliases, the only version linked to runs).
- Databricks-managed judges (`model_uri: databricks`, `databricks-agents`) work on Free
  Edition: 6/6 cases judged, no errors; on mock outputs Correctness 5/6 and
  ExpectationsGuidelines 4/6 with case-specific rationales, Safety 6/6.
- Unity Gateway is available on Free Edition and serves chat completions; gpt-oss-20b/120b,
  qwen35-122b-a10b, qwen3-next-80b and llama-4-maverick answer, Gemma/Llama-3 endpoints and
  frontier models do not. Qwen3.5 returns content as typed parts, so the client keeps only
  text parts. Bundle defaults: claims-small gpt-oss-20b, claims-medium qwen35-122b-a10b,
  claims-judge gpt-oss-120b. Real run via the gateway (app + judge): 6/6 on gates and
  judges, no judge errors, no Authorization/Bearer strings in traces.
- CI exports `BUNDLE_VAR_*` only when the GitHub variable is set, so empty variables keep the
  bundle defaults.
- First CI deploy from `main` (2026-11-25): authenticated as `lighthouse-ci`, deployed target
  `ci`, registered `workspace.lighthouse.claim_settlement_review` version 1 and ran the mock
  evaluation (`TERMINATED SUCCESS`).

## ADR-006: Developer CLI with prompt references and content-addressed drafts (2026-11-25)

**Context.** The job runs a fixed prompt version and model. Developers need to inspect
versions, edit a prompt, try it on a case and compare it, locally and on Databricks, without
creating a registry version for every edit.

**Decision.**
- A `lighthouse` console script (`src/lighthouse_poc/cli.py`) is the developer entry point;
  scripts stay as job entry points. Commands share the profile, registry and assistant code
  with the job.
- One prompt reference syntax everywhere: `<version>`, `@<alias>`, `latest`, `file:<path>`
  (`src/lighthouse_poc/prompts.py`). Ambiguous input (e.g. `v3`, bare `champion`) is rejected.
- Drafts are unregistered and labelled `draft-<sha256[:8]>` of their text; the label is the
  prompt version in logs and trace tags. Only registered versions can be approved.
- `try` runs one golden case as its own trace (`lighthouse.kind=try`) and applies the
  deterministic gates; it is a smoke test, not review evidence.
- Interactive output is concise; `-v` adds the structured JSON logs. Errors are typed and
  exit with code 2; gate failures exit with 1.
- Planned: `eval` with `--baseline` (step 2), then `promote` that records the approval on the
  prompt version and run before moving `@champion`, with the assistant and job loading
  `@champion` (step 3).

**Consequences.** Unity Catalog returns protobuf timestamps for prompt versions where the MLflow
server returns epoch milliseconds; the CLI handles both. Reading another identity's schema
(e.g. the CI schema) needs an explicit grant.

## ADR-007: MLflow UI AI Gateway endpoints call Kong (2026-11-25)

**Context.** The MLflow 3.18 UI (Playground, judges, automatic evaluation) only lists models
registered as MLflow AI Gateway endpoints. Developers and reviewers want to author prompts and
judge traces in the UI with the same local models the code uses.

**Decision.**
- UI endpoints are OpenAI-provider endpoints whose base URL is a Kong route
  (`http://kong:8000/ai/<logical-model>`) and whose model name is the model Kong pins for that
  route. Kong remains the authenticating, routing gateway; MLflow's gateway is a client of it.
- Kong's pre-function maps `Authorization: Bearer` (the only credential the MLflow gateway
  sends) to `apikey` before key-auth and strips it from the upstream request. It also defaults
  `reasoning_effort` to `none` when the caller does not set it.
- The MLflow server encrypts stored API keys with `MLFLOW_CRYPTO_KEK_PASSPHRASE` from `.env`,
  and allows private, plain-http upstream base URLs, because the traffic stays on the Docker
  network.
- Endpoints enable usage tracking into the experiment being worked on, so UI trials become
  traces that UI judges can score.
- MLflow artifacts are proxied by the server (`--artifacts-destination`).

**Consequences.** Verified: gateway → Kong → Ollama answers in about 1.6 s; usage tracking
writes `gateway/<endpoint>` → `provider/...` traces without the Kong key. The 3.18 UI cannot run
a prompt version over a dataset, so batch evaluation stays in code (`lighthouse eval`, planned).
The Kong key is stored in MLflow (encrypted) in addition to `.env`.

## ADR-008: Review score = hard gates + weighted rubric + LLM judges (2026-11-25)

**Context.** Reviewers need one comparable number per prompt version and model, without
letting an LLM judge override deterministic safety checks or approve a release.

**Decision.**
- An evaluation spec per use case (`config/evals/<name>.yaml`) declares the adapter, the
  dataset, the hard gates, the weighted rubric, the judges and the scoring weights. The spec
  digest is logged as `scorer_version`.
- Hard gates are deterministic and blocking: a failed gate makes the case score 0 and the run
  BLOCKED. Rubric criteria are deterministic and weighted (0..1, partial credit for expected
  fields). Judges are MLflow built-ins (Guidelines, ExpectationsGuidelines, Safety,
  Correctness) on the profile judge model; yes = 1, no = 0; errors are excluded and counted.
- Combined per case = rubric_weight × rubric + judge_weight × judges (default 0.6 / 0.4);
  rubric only when no judge produced a verdict. The run's mean combined score is compared with
  `pass_threshold` (default 0.8): BLOCKED, NEEDS WORK or READY FOR HUMAN REVIEW.
- Every criterion is an MLflow scorer (`gate_*`, `rubric_*`, `judge_*`) with a rationale, and
  the combined score is logged on each trace as a CODE assessment, so the UI shows the whole
  basis of the score.
- The verdict is review evidence only. Approval stays a separate human step (`promote`, step 3).

**Consequences.** Weights and thresholds are per spec and must be agreed with the claims SME;
the defaults are a starting point. Judge rationales are model-generated and are reviewed, not
trusted. A spec change produces a new `scorer_version`, so runs scored by different specs are
not compared as equals.

## ADR-009: Review gate with an evidence-bound approval record (2026-11-25)

**Context.** The design requires that only an approved prompt reference reaches the agent,
and that each decision is bound to exact versions and run evidence. An alias alone does not
record why it moved.

**Decision.**
- `lighthouse promote` approves or rejects one registered version based on one `lighthouse eval`
  run. Approval needs: a registered version, a finished eval run of the same name and version,
  verdict READY FOR HUMAN REVIEW, 100% hard gates, and an unchanged spec digest. Rejection only
  needs the run to match the version.
- The record (`approvals/<record_id>.json`) is an artifact on the evaluation run, so every
  decision is kept next to its evidence. `lighthouse_approval_*` tags on the run and the prompt
  version hold the latest decision for listing. On approval, the alias moves last; the previous
  target is recorded for rollback.
- Reviewer and rationale are mandatory free text; the local OS user is recorded as
  `recorded_by`. There is no identity verification in the POC.

**Consequences.** Verified locally and on Unity Catalog. Unity Catalog differences that were
found and handled: prompt tag keys cannot contain dots, and versions are returned without
aliases (resolved through `load_alias`). Write order means a failed alias move leaves a record
that names the intended alias; re-running promote completes it. Open: binding the reviewer to an
authenticated identity (Databricks user or service principal) and separating the author from
the approver.

## ADR-010: Consumers load @champion and fail closed; the job evaluates against it (2026-11-25)

**Context.** Approval only matters if consumers use the approved version. The job still
evaluated a fixed, just-registered version.

**Decision.**
- The agent serves `<prompt>@champion` (`published_assistant`). Without an approved version it
  raises a `ConfigurationError`; there is no fallback to `latest` or a fixed number, because
  that would serve an unreviewed prompt.
- `lighthouse try` and the notebook default to `@champion`; scripts accept references.
- The job keeps testing what Git changed, while also checking the published version. The job's
  `evaluate` task runs `lighthouse eval` on the registered version with `--baseline @champion
  --baseline-optional`: the champion is re-scored on every run (regression after gateway or
  model changes), and the candidate's run is the evidence `lighthouse promote` needs. With no
  champion yet, or when the champion is that same version, the baseline is skipped with a
  notice. A BLOCKED candidate fails the task.

**Consequences.** The first approval on each environment is a manual `promote` (by design). The
job's runtime roughly doubles once a champion exists and differs from the candidate.

## ADR-011: Seeded UI demo state and UI-runnable code judges (local) (2026-11-26)

**Decision.** `scripts/seed_ui_demo.py` idempotently creates the local UI state: gateway
endpoints for the Kong routes, with app endpoints tracking into `/lighthouse-ui-sample`;
`claim_intake_triage` v1 and v2 with `@candidate`; prompt-experiment links; the golden dataset;
and registered judges. Custom code judges (`src/lighthouse_poc/ui_judges.py`) mirror the
triage spec's gates and rubric so reviewers can run them on any trace from the Judges page.
They are self-contained because MLflow stores and re-creates only the function body, and they
accept dict, JSON-string and chat-completion outputs. LLM judges use `gateway:/kong-claims-judge`.

**Consequences.** `MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS=true` is set on the local server only:
it executes registered code. The UI judge copies duplicate the spec criteria; tests check that
they agree on fixtures, and `lighthouse eval` remains the scoring of record. The seed never sets
`@champion`.


## ADR-012: AI Marketplace data platform on Unity Catalog with Alembic migrations (2026-11-27)

**Decision.** The AI Marketplace stores what it produces in Unity Catalog, in three schemas per
environment: `ai_marketplace` (model catalog, use cases, release decisions, migration audit),
`ai_observability` (model invocations, evaluation summaries, vector queries, dashboard views)
and `ai_knowledge` (documents, embedded chunks, vector index). A schema prefix (`dev_<user>_`)
separates personal dev schemas inside the Free Edition `workspace` catalog. Structure changes
are **Alembic revisions** run on a SQL warehouse through `databricks-sqlalchemy`
(`lighthouse platform db …`, `src/lighthouse_poc/platform/alembic`). Revisions use placeholder
names per environment, re-runnable statements (Delta DDL is not transactional), and carry
their own data migration (e.g. `0002`: add a column, backfill it). Before every upgrade or
downgrade the runner records each table's Delta version (`schema_migration_audit`) so data
can be restored with `RESTORE TABLE`. Reference data is not migrated: the model list is YAML
in Git, synced as a type-2 history (`lighthouse platform models sync`). Tables use column mapping,
change data feed and liquid clustering. Design: `docs/ai-marketplace-platform.md`.

**Alternatives.** Liquibase OSS Databricks extension: mature change management, but Java/JDBC on
serverless and changelogs outside the Python stack. A hand-written runner: no revision graph,
offline SQL or ecosystem. Bundle-managed tables: Asset Bundles do not manage table DDL changes.

**Consequences.** One migration runner at a time (notebook or a job with `max_concurrent_runs: 1`).
Every structure change is reviewed as offline SQL (`platform db sql`) and tested offline
(`tests/test_platform.py`). Verified on Free Edition on 2026-11-27: upgrade, idempotent sync,
downgrade/upgrade round trip and restore points.

## ADR-013: Every model call through Unity Gateway, attributed by request tags; self-managed embeddings (2026-11-27)

**Decision.** Chat and embedding models are called only through Unity Gateway by Unity Catalog
model service names, listed in `config/model_catalog.yaml` (approved / candidate / unavailable /
retired). Each call carries a `Databricks-Ai-Gateway-Request-Tags` header (`lh_event_id`,
`lh_use_case`, `lh_purpose`, `lh_environment`, `lh_prompt_version`). The same event ID keys a
row in `ai_observability.model_invocation` with the domain context only the application knows,
such as prompt version, evaluation run and synthetic case ID. Views join it to
`system.ai_gateway.usage`, preferring the gateway's tokens and latency. Vector Search uses a
Delta Sync index with a **self-managed** embedding column: chunks and queries are embedded by
`embed-default` (`system.ai.qwen3-embedding-0-6b`, 1024-d) through the gateway.

**Alternatives.** Vector Search managed embeddings (`databricks-gte-large-en`) call a serving
endpoint directly, outside the gateway's usage, limits and audit trail. Logging tokens only in
our own table duplicates the system table and misses calls made by other tools.

**Consequences.** Gateway usage rows arrive with a delay, so dashboards show `gateway_confirmed`
filling in later. Changing the embedding model means re-embedding and re-syncing the index
(new `embedding_model_service` triggers the chunk MERGE update). No prompt or response text is
stored in observability tables.

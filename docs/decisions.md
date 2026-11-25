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

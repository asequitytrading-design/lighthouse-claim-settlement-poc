# AI Marketplace data platform on Databricks

Databricks is the engine of the AI Marketplace: Unity Catalog stores what the marketplace
produces (usage, quality, releases, knowledge), Unity Gateway routes every model call, and
Mosaic AI Vector Search serves the knowledge base. Decisions: ADR-012 (data platform and schema
migrations) and ADR-013 (gateway-only model access, attribution, self-managed embeddings).
Verified on the Free Edition workspace on 2026-11-27. Synthetic data only.

## 1. Architecture

```
 applications (claim settlement assistant, lighthouse eval/try, notebooks, jobs)
      │  logical model keys (claims-medium, claims-judge, embed-default ...)
      ▼
 GatewayChatClient / GatewayEmbeddingClient ──► Unity Gateway  <host>/ai-gateway/mlflow/v1
      │  header Databricks-Ai-Gateway-Request-Tags:            │  routes to system.ai.<model>
      │  {lh_event_id, lh_use_case, lh_purpose, ...}           ▼
      │                                         system.ai_gateway.usage (tokens, latency, status,
      │                                         requester, routing, request_tags)
      ▼
 Unity Catalog  <catalog>
   <prefix>ai_marketplace    model_catalog (type-2), use_case, release_decision,
                             schema_migration_audit, alembic_version
   <prefix>ai_observability  model_invocation, eval_run, eval_case_result, vector_query,
                             views: v_gateway_usage, v_model_usage_daily, v_eval_trend
   <prefix>ai_knowledge      source_document, document_chunk (CDF) ──► document_chunk_index
                                                                        (Vector Search, Delta Sync,
                                                                         self-managed embeddings)
 MLflow (prompts in UC, traces, evaluation runs) stays the system of record for prompt
 versions and evaluation evidence; eval_run / eval_case_result are dashboard summaries.
```

## 2. Unity Catalog layout

| Item | Value | Notes |
|---|---|---|
| Catalog | `workspace` (Free Edition); `platform.catalog` in `config/profiles/databricks.yaml` | A paid workspace would use one catalog per environment |
| Schema prefix | `""` for shared targets, `dev_<user>_` for personal dev (notebook default) | `LIGHTHOUSE_PLATFORM_PREFIX`; keeps experiments apart |
| Schemas | `ai_marketplace`, `ai_observability`, `ai_knowledge` | Grouped by owner and access pattern, not by tool |
| Owner of change | Alembic revisions (structure), syncs (reference data), applications (events) | Nobody creates platform tables by hand |

## 3. Data model

All tables: Delta, column mapping (`name`, so revisions can rename/drop columns), change data
feed on, liquid clustering on dashboard filters. No prompt text, responses, claim narratives or
health details anywhere in observability.

### ai_marketplace (governed reference data)

| Table | Grain | Key columns | Written by |
|---|---|---|---|
| `model_catalog` | model key × validity period (type 2) | `model_key`, `model_service`, `task`, `roles`, `status`, `embedding_dimensions`, `valid_from/valid_to/is_current`, `record_hash` | `lighthouse platform models sync` from `config/model_catalog.yaml` |
| `use_case` | application | `use_case_id`, `owner`, `risk_tier`, `data_classification`, `status` | onboarding (next phase) |
| `release_decision` | human review decision | `prompt_name/version`, `decision`, `reviewer`, `eval_run_id`, `review_score`, `model_key` | `lighthouse promote` (next phase; today MLflow tags/artifacts) |
| `schema_migration_audit` | table × migration run | `migration_run_id`, `table_name`, `delta_version` | migration runner (restore points) |
| `alembic_version` | one row | `version_num` | Alembic |
| view `v_model_catalog_current` | current model list | | |

### ai_observability (usage and quality)

| Table / view | Grain | Key columns | Written by |
|---|---|---|---|
| `model_invocation` | model call | `event_id` (= request tag `lh_event_id`), `event_time`, `use_case_id`, `purpose`, `model_key`, `model_service`, `served_model`, `prompt_version`, `run_id`, `trace_id`, `case_id` (synthetic), `status`, `latency_ms`, tokens, `input_count`, `embedding_dimensions` | applications via `EventWriter` (today: verification probes and knowledge loads) |
| `eval_run` | MLflow evaluation run | prompt/model/judge/dataset versions, `combined_score`, `verdict` | `lighthouse eval` (next phase) |
| `eval_case_result` | run × case × criterion | `kind` (gate/rubric/judge), `score`, `passed`, `trace_id` | `lighthouse eval` (next phase) |
| `vector_query` | vector search query | `index_name`, `embedding_model_key`, `num_results`, `latency_ms`, `status` | applications / verification |
| view `v_gateway_usage` | gateway request | `system.ai_gateway.usage` with tags as columns | |
| view `v_model_usage_daily` | day × use case × model × purpose | calls, errors, p50/p95 latency, tokens, `gateway_confirmed` | dashboard source |
| view `v_eval_trend` | evaluation run | score and verdict over time | dashboard source |

Why both our events and the system table: the gateway knows tokens, latency, status and
requester authoritatively; only the application knows the prompt version, evaluation run and
synthetic case. The request tag `lh_event_id` joins them; the views prefer gateway values.
System tables arrive with a delay, so `gateway_confirmed` fills in later.

### ai_knowledge (knowledge base)

| Table | Grain | Key columns |
|---|---|---|
| `source_document` | document version | `doc_id`, `doc_type`, `product_line`, `version`, `content_hash`, `classification` (synthetic) |
| `document_chunk` | chunk | `chunk_id` (PK), `doc_id`, `chunk_index`, `text`, `embedding ARRAY<FLOAT>`, `embedding_model_key/service`, `embedding_dimensions`, `content_hash` |
| index `document_chunk_index` | chunk | Delta Sync, TRIGGERED, primary key `chunk_id`, embedding column `embedding` (1024-d) |

## 4. Models (finalised list, `config/model_catalog.yaml`)

Probed through Unity Gateway on 2026-11-27 (chat "Reply OK"; embeddings dimension check).

| Key | Model service | Task | Role | Status |
|---|---|---|---|---|
| `claims-small` | `system.ai.gpt-oss-20b` | chat | app | approved |
| `claims-medium` | `system.ai.qwen35-122b-a10b` | chat | app (default) | approved |
| `claims-judge` | `system.ai.gpt-oss-120b` | chat | judge (different family, ADR-003) | approved |
| `claims-alt-qwen-next` | `system.ai.qwen3-next-80b-a3b-instruct` | chat | app | candidate |
| `claims-alt-llama` | `system.ai.llama-4-maverick` | chat | app, judge | candidate |
| `embed-default` | `system.ai.qwen3-embedding-0-6b` | embeddings (1024-d) | embedding | approved |
| `frontier-claude-sonnet`, `frontier-gpt`, `frontier-gemini` | `system.ai.databricks-…` | chat | app, judge | unavailable (404 on Free Edition) |
| `embed-gte-large`, `embed-bge-large` | `system.ai.gte_large_en_v1_5`, `bge_large_en_v1_5` | embeddings | embedding | unavailable (serving endpoints only, not routed by the gateway) |

Not listed: Llama 3.x 8B/70B and Gemma 3 12B, which are serving endpoints only and would bypass the gateway.
Changing the list is a pull request to the YAML plus `lighthouse platform models sync`; history is kept.

## 5. Schema migrations (Alembic)

Tool: **Alembic + `databricks-sqlalchemy`** on a SQL warehouse (ADR-012). Alternatives
considered: Liquibase OSS Databricks extension (mature, but Java/JDBC on serverless and XML/YAML
changelogs outside our Python stack) and a hand-written runner (no revision graph, offline SQL or
ecosystem).

| Concern | How it is handled |
|---|---|
| Where revisions live | `src/lighthouse_poc/platform/alembic/versions/NNNN_<slug>.py` |
| Environments | Placeholders `{catalog}`, `{mkt}`, `{obs}`, `{kn}` filled per catalog + prefix |
| Non-transactional DDL | One revision at a time, statements re-runnable (`IF NOT EXISTS`, `column_exists()`); a failed revision is fixed and re-run |
| Data migrations | In the same revision as the structure change (example `0002`: `ADD COLUMNS` then `UPDATE` backfill) |
| Rollback | `downgrade` for structure (column mapping allows `DROP COLUMNS`); data via restore points: every run records each table's Delta version, `lighthouse platform db restore-sql <run>` prints the `RESTORE TABLE` commands |
| Review | `lighthouse platform db sql` renders the SQL offline (change ticket / PR) |
| Concurrency | One runner at a time (notebook or job with `max_concurrent_runs: 1`) |
| Reference data | Not migrations: YAML + type-2 sync (models), application writes (events) |

```bash
lighthouse --profile databricks platform db history            # offline
lighthouse --profile databricks platform db sql                # offline SQL review
lighthouse --profile databricks platform db current  --prefix dev_<user>_
lighthouse --profile databricks platform db upgrade  --prefix dev_<user>_
lighthouse --profile databricks platform db downgrade --to -1 --prefix dev_<user>_
lighthouse --profile databricks platform db restore-sql <migration_run_id>
lighthouse --profile databricks platform models sync --prefix dev_<user>_
lighthouse --profile databricks platform verify      --prefix dev_<user>_
```

Adding a revision: copy the newest revision file, set `revision`/`down_revision`, write
re-runnable SQL with placeholders, add a test expectation in `tests/test_platform.py`
(offline SQL), run `platform db sql`, then `upgrade` in your dev prefix.

## 6. Verification notebook

`notebooks/02_ai_marketplace_platform_setup.py` (thin; logic in `lighthouse_poc.platform`):
install packages → widgets (catalog, prefix default `dev_<user>_`, warehouse, endpoint) → show
pending migration SQL → run the eight checks (workspace, migrations, objects, catalog, gateway,
knowledge, vectors, dashboard) → show catalog, today's usage and restore points → optional
downgrade/upgrade round trip. The CLI runs the same checks: `lighthouse platform verify`.

Run it after `databricks bundle deploy` (workspace files include `src/`, `config/`, `data/`),
or from a Git folder. A first run creates the Vector Search endpoint (10-20 min); later runs
reuse it.

## 7. Free Edition constraints

- One Vector Search endpoint with one unit, Delta Sync indexes only (no Direct Access). The runner
  reuses an existing endpoint.
- Frontier models (Claude, GPT-5, Gemini) and gte/bge embeddings are not routed by the gateway (404).
- One serverless SQL warehouse (starts on demand); 5 concurrent job tasks.
- `system.ai_gateway.usage` is readable by the workspace admin; rows arrive with a delay.

## 8. Next phases

1. Applications write `model_invocation` (agent, `lighthouse try/eval`) with request tags.
   `eval_run`/`eval_case_result` from `lighthouse eval`; `release_decision` from `lighthouse promote`.
2. Bundle job task `platform_migrate` (before the eval job; `max_concurrent_runs: 1`) for `ci`.
3. AI/BI dashboard on `v_model_usage_daily`, `v_eval_trend`, `v_model_catalog_current`.
4. Grants: applications write events only; reviewers read; platform owner migrates (group-based
   on a paid workspace).
5. Retention: `model_invocation`/`vector_query` 400 days, then archive/VACUUM policy.
6. Retrieval in the assistant: examiner guidance as context, with sources cited in the handoff.

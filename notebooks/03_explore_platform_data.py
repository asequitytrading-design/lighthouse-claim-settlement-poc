# Databricks notebook source
# MAGIC %md
# MAGIC # AI Marketplace data explorer
# MAGIC
# MAGIC Query every table and view of the AI Marketplace data platform (`docs/ai-marketplace-platform.md`).
# MAGIC Three cells per schema:
# MAGIC
# MAGIC | Cell | What it does |
# MAGIC |---|---|
# MAGIC | **A. Tables** | lists the tables and views (with their columns) in a comment, then counts rows |
# MAGIC | **B. Filters** | queries each table with filters you can change at the top of the cell |
# MAGIC | **C. Joins** | join examples across tables, views and schemas |
# MAGIC
# MAGIC Section **4** explores the knowledge base through **Mosaic AI Vector Search**: 15 sample queries,
# MAGIC five retrieval strategies you can pick from, and a side-by-side comparison.
# MAGIC
# MAGIC **How to use it:** attach **Serverless** compute, run the first two cells (packages; Python restarts),
# MAGIC then the *Setup* cell, then any cell (Shift+Enter). Change the filter variables at the top of a cell
# MAGIC and run it again. Every query in sections 1-3 is a plain `spark.sql(...)`; copy one into a `%sql`
# MAGIC cell or the SQL editor if you prefer SQL.
# MAGIC
# MAGIC Sections 1-3 are read-only. Section 4 calls models through Unity Gateway (query embeddings, and
# MAGIC the reranker for one strategy) and records those calls as usage events, without the query text,
# MAGIC under use case `knowledge_search_explorer`. Synthetic data only.
# MAGIC
# MAGIC ---
# MAGIC ### The knowledge base: how it was chunked and indexed
# MAGIC
# MAGIC **Source.** 7 synthetic examiner-guidance documents (`data/knowledge/examiner_guidance.jsonl`): one
# MAGIC evidence checklist per product line (STD, LTD, life, accident, critical illness, dental) and one
# MAGIC cross-product handoff guide (`product_line = all`). Loaded into `ai_knowledge.source_document` by
# MAGIC notebook 02 (step 6).
# MAGIC
# MAGIC **Chunking strategy: paragraph-preserving, size-capped** (`chunk_document`, `max_chars = 450`):
# MAGIC 1. Split each document on blank lines into paragraphs (the natural unit of meaning: intro,
# MAGIC    *expected items*, *flag when...*).
# MAGIC 2. A paragraph longer than 450 characters is cut at the last space before the limit (never mid-word).
# MAGIC 3. Consecutive paragraphs are packed into one chunk while the result stays within 450 characters.
# MAGIC 4. No overlap between chunks; each chunk keeps `doc_id` and `chunk_index`, and
# MAGIC    `chunk_id = <doc_id>-<nnn>` is its primary key.
# MAGIC
# MAGIC Result: **13 chunks** (about 2 per document, 300-450 characters, roughly 60-100 tokens). Small,
# MAGIC topic-pure chunks make matching precise. When an answer needs a whole document, the
# MAGIC `parent_document` strategy (section 4) returns it.
# MAGIC
# MAGIC **Embeddings: self-managed, through Unity Gateway.** Every chunk is embedded by catalog model
# MAGIC `embed-default` = `system.ai.qwen3-embedding-0-6b` (**1024 dimensions**) in one gateway call
# MAGIC (ADR-013). The vector is stored in `document_chunk.embedding ARRAY<FLOAT>` together with the model
# MAGIC key and service, so a model change is visible and triggers re-embedding.
# MAGIC
# MAGIC **Indexing methodology: Mosaic AI Vector Search, Delta Sync index**
# MAGIC
# MAGIC | Property | Value |
# MAGIC |---|---|
# MAGIC | Endpoint | `lighthouse-vs` (STANDARD; Free Edition allows one endpoint) |
# MAGIC | Index | `<prefix>ai_knowledge.document_chunk_index` |
# MAGIC | Type | **Delta Sync** from `document_chunk` (change data feed on): the index follows the table |
# MAGIC | Sync mode | **TRIGGERED**: a sync runs on request (notebook 02 after loading), not continuously, so you pay for no idle pipeline |
# MAGIC | Embeddings | **self-managed** vector column `embedding` (1024-d); queries must be embedded with the same model |
# MAGIC | Primary key | `chunk_id` |
# MAGIC | Search | approximate nearest neighbour (**HNSW** graph) on **L2 distance**, reported as a similarity score (higher = closer); **hybrid** adds full-text keyword search and fuses both rankings (reciprocal rank fusion) |
# MAGIC | Columns returned | `chunk_id`, `doc_id`, `chunk_index`, `text` (+ `score`) |
# MAGIC
# MAGIC Why self-managed instead of managed embeddings: managed embeddings call a serving endpoint
# MAGIC directly, outside Unity Gateway's usage tracking, limits and audit trail. On Free Edition the
# MAGIC gte/bge endpoints are not routed by the gateway at all.

# COMMAND ----------

# MAGIC %pip install --quiet "databricks-vectorsearch>=0.57" "openai>=1.0" "pyyaml>=6" "databricks-sqlalchemy>=2.0.8,<3"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup: pick the catalog and schema prefix
from databricks.sdk import WorkspaceClient
import re

me = WorkspaceClient().current_user.me().user_name
default_prefix = "dev_" + (re.sub(r"[^a-z0-9]+", "_", me.split("@")[0].lower()).strip("_") or "user") + "_"

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema_prefix", default_prefix)      # "" for the shared schemas

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("schema_prefix").strip()
MKT = f"{CATALOG}.{PREFIX}ai_marketplace"       # governed reference data
OBS = f"{CATALOG}.{PREFIX}ai_observability"     # usage and quality events
KN = f"{CATALOG}.{PREFIX}ai_knowledge"          # knowledge base


def q(sql: str, limit: int = 100):
    """Run a query and show the result (at most `limit` rows)."""
    display(spark.sql(sql).limit(limit))


print("Schemas:", MKT, OBS, KN, sep="\n  ")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. `ai_marketplace`: governed reference data

# COMMAND ----------

# DBTITLE 1,1A. Marketplace tables and row counts
# Tables and views in <prefix>ai_marketplace
# ---------------------------------------------------------------------------------------------
# model_catalog            governed model list, type-2 history (one row per model per version)
#     model_key, model_service, task (chat|embeddings), family, provider, roles ARRAY<STRING>,
#     status (approved|candidate|unavailable|retired), embedding_dimensions, verified_on, notes,
#     record_hash, valid_from, valid_to, is_current, synced_by
# model_price              rate card, effective-dated (from config/model_pricing.yaml)
#     model_key, effective_from, currency, input_usd_per_1m_tokens, output_usd_per_1m_tokens,
#     price_source (illustrative|list|contract), notes, synced_at, synced_by
# use_case                 applications that consume models and prompts (next phase: onboarding)
#     use_case_id, name, description, owner, risk_tier, data_classification, status,
#     created_at, updated_at
# release_decision         human review decisions (lighthouse promote; next phase)
#     decision_id, use_case_id, prompt_name, prompt_version, decision, reviewer, rationale,
#     eval_run_id, review_score, verdict, model_key, model_service, dataset_version, git_commit,
#     decided_at
# schema_migration_audit   restore points: Delta version of each table before every migration run
#     migration_run_id, from_revision, intent, table_name, delta_version, recorded_at, recorded_by
# alembic_version          the applied schema revision (one row): version_num
# v_model_catalog_current  view: current row of each model (model_catalog WHERE is_current)
# v_model_price            view: rate card with effective_to (the next rate's effective_from)
# ---------------------------------------------------------------------------------------------
q(f"""
SELECT 'model_catalog' AS table_name, count(*) AS row_count FROM {MKT}.model_catalog
UNION ALL SELECT 'model_price', count(*) FROM {MKT}.model_price
UNION ALL SELECT 'use_case', count(*) FROM {MKT}.use_case
UNION ALL SELECT 'release_decision', count(*) FROM {MKT}.release_decision
UNION ALL SELECT 'schema_migration_audit', count(*) FROM {MKT}.schema_migration_audit
UNION ALL SELECT 'alembic_version', count(*) FROM {MKT}.alembic_version
""")
q(f"SELECT version_num AS applied_revision FROM {MKT}.alembic_version")
q(f"SHOW TABLES IN {MKT}")

# COMMAND ----------

# DBTITLE 1,1B. Marketplace queries with filters
# Filters: change and re-run. None = no filter.
TASK = "chat"                 # chat | embeddings | None
STATUS = "approved"           # approved | candidate | unavailable | retired | None
MODEL_KEY = "claims-medium"   # a model key for the history and price queries
MIGRATION_DAYS = 30

where = " AND ".join(f for f in [
    f"task = '{TASK}'" if TASK else None,
    f"status = '{STATUS}'" if STATUS else None] if f) or "true"

# current models, filtered by task and status; roles is an array: array_contains() filters it
q(f"""
SELECT model_key, model_service, task, status, roles, family, embedding_dimensions, verified_on
FROM {MKT}.v_model_catalog_current
WHERE {where}
ORDER BY model_key
""")

# models that may act as a judge (array filter)
q(f"SELECT model_key, status FROM {MKT}.v_model_catalog_current WHERE array_contains(roles, 'judge')")

# full type-2 history of one model: every version, when it was valid, who synced it
q(f"""
SELECT model_key, status, verified_on, record_hash, valid_from, valid_to, is_current, synced_by
FROM {MKT}.model_catalog
WHERE model_key = '{MODEL_KEY}'
ORDER BY valid_from DESC
""")

# rate card of one model with validity periods
q(f"SELECT * FROM {MKT}.v_model_price WHERE model_key = '{MODEL_KEY}' ORDER BY effective_from")

# recent migration runs and the restore points they recorded
q(f"""
SELECT migration_run_id, from_revision, intent, count(*) AS tables, max(recorded_at) AS recorded_at
FROM {MKT}.schema_migration_audit
WHERE recorded_at >= current_timestamp() - INTERVAL {MIGRATION_DAYS} DAYS
GROUP BY migration_run_id, from_revision, intent
ORDER BY recorded_at DESC
""")

# COMMAND ----------

# DBTITLE 1,1C. Marketplace join examples
# 1) each current model with the price in force today (LEFT JOIN: models without a rate stay)
q(f"""
SELECT c.model_key, c.task, c.status, p.currency,
       p.input_usd_per_1m_tokens, p.output_usd_per_1m_tokens, p.price_source, p.effective_from
FROM {MKT}.v_model_catalog_current c
LEFT JOIN {MKT}.v_model_price p
  ON p.model_key = c.model_key
 AND current_date() >= p.effective_from
 AND (p.effective_to IS NULL OR current_date() < p.effective_to)
ORDER BY c.task, c.status, c.model_key
""")

# 2) as-of join: which catalog version and price were in force when each restore point was
#    recorded (the same pattern prices any event at its own time)
q(f"""
SELECT a.migration_run_id, a.recorded_at, c.model_key, c.status, c.record_hash
FROM (SELECT DISTINCT migration_run_id, recorded_at FROM {MKT}.schema_migration_audit) a
JOIN {MKT}.model_catalog c
  ON a.recorded_at >= c.valid_from AND (c.valid_to IS NULL OR a.recorded_at < c.valid_to)
WHERE c.model_key = '{MODEL_KEY}'
ORDER BY a.recorded_at DESC
""", limit=20)

# 3) release decisions with their use case and the catalog entry of the model they approved
#    (empty until `lighthouse promote` writes release_decision: next phase)
q(f"""
SELECT r.decided_at, u.name AS use_case, r.prompt_name, r.prompt_version, r.decision,
       r.reviewer, r.review_score, r.verdict, c.model_service, c.status AS model_status
FROM {MKT}.release_decision r
LEFT JOIN {MKT}.use_case u ON u.use_case_id = r.use_case_id
LEFT JOIN {MKT}.v_model_catalog_current c ON c.model_key = r.model_key
ORDER BY r.decided_at DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. `ai_observability`: usage, latency, tokens, cost and quality

# COMMAND ----------

# DBTITLE 1,2A. Observability tables and row counts
# Tables and views in <prefix>ai_observability
# ---------------------------------------------------------------------------------------------
# model_invocation     one row per model call made by a marketplace application (no prompt text)
#     event_id (= request tag lh_event_id), event_time, event_date (generated), environment,
#     use_case_id, purpose (app|judge|embedding_index|embedding_query|probe), model_key,
#     model_service, task, prompt_name, prompt_version, app_version, git_commit, trace_id, run_id,
#     case_id (synthetic), status (ok|error), error_category, latency_ms, input_tokens,
#     output_tokens, input_count, embedding_dimensions, served_model
# eval_run             MLflow evaluation runs summarised (lighthouse eval; next phase)
#     run_id, experiment_id, use_case_id, prompt_name, prompt_version, model_key, model_service,
#     judge_model_key, dataset_name, dataset_version, eval_spec, scorer_version, cases,
#     combined_score, verdict (READY|NEEDS_REVIEW|BLOCKED), git_commit, started_at, finished_at
# eval_case_result     per case and criterion score of an evaluation run
#     run_id, case_id, criterion, kind (gate|rubric|judge), value, score, passed, trace_id
# vector_query         Vector Search queries (no query text)
#     event_id, event_time, event_date, use_case_id, index_name, embedding_model_key,
#     num_results_requested, num_results, latency_ms, status, error_category
# v_gateway_usage      view: system.ai_gateway.usage with our request tags as columns
#     gateway_request_id, event_time, event_date, event_id, use_case_id, purpose, environment,
#     model_service, destination_model, api_type, requester, status_code, latency_ms,
#     time_to_first_byte_ms, input_tokens, output_tokens, total_tokens
# v_model_call_cost    view: one row per call (our event FULL JOIN gateway record) with cost
#     event_time, event_date, event_hour, event_id, gateway_request_id, environment, use_case_id,
#     purpose, model_key, model_service, destination_model, api_type, task, prompt_name,
#     prompt_version, run_id, trace_id, case_id, status, is_error, status_code, error_category,
#     latency_ms, input_tokens, output_tokens, total_tokens, currency, price_source,
#     input_usd_per_1m_tokens, output_usd_per_1m_tokens, input_cost_usd, output_cost_usd,
#     total_cost_usd, priced, gateway_confirmed, app_recorded
# v_model_cost_daily   view: per day x use case x model x purpose: calls, errors, p50/p95 latency,
#     tokens, input/output/total cost, unpriced_calls
# v_model_usage_daily  view: per day x use case x model x purpose: calls, errors, p50/p95 latency,
#     tokens, gateway_confirmed (no cost; the original usage view)
# v_eval_trend         view: evaluation score and verdict over time
# ---------------------------------------------------------------------------------------------
q(f"""
SELECT 'model_invocation' AS table_name, count(*) AS row_count FROM {OBS}.model_invocation
UNION ALL SELECT 'eval_run', count(*) FROM {OBS}.eval_run
UNION ALL SELECT 'eval_case_result', count(*) FROM {OBS}.eval_case_result
UNION ALL SELECT 'vector_query', count(*) FROM {OBS}.vector_query
UNION ALL SELECT 'v_gateway_usage (whole workspace)', count(*) FROM {OBS}.v_gateway_usage
UNION ALL SELECT 'v_model_call_cost', count(*) FROM {OBS}.v_model_call_cost
""")
q(f"SHOW TABLES IN {OBS}")

# COMMAND ----------

# DBTITLE 1,2B. Observability queries with filters
# Filters: change and re-run. None = no filter.
DAYS = 7                      # look-back window
MODEL_KEY = None              # e.g. "claims-medium"
USE_CASE = None               # e.g. "claim_settlement_demo", "platform_setup_verification"
PURPOSE = None                # app | judge | embedding_index | embedding_query | probe
STATUS = None                 # ok | error

filters = " AND ".join(f for f in [
    f"event_time >= current_timestamp() - INTERVAL {DAYS} DAYS",
    f"model_key = '{MODEL_KEY}'" if MODEL_KEY else None,
    f"use_case_id = '{USE_CASE}'" if USE_CASE else None,
    f"purpose = '{PURPOSE}'" if PURPOSE else None,
    f"status = '{STATUS}'" if STATUS else None] if f)

# raw application events (what our code recorded)
q(f"""
SELECT event_time, use_case_id, purpose, model_key, status, error_category, latency_ms,
       input_tokens, output_tokens, prompt_version, case_id, event_id
FROM {OBS}.model_invocation
WHERE {filters}
ORDER BY event_time DESC
""")

# usage, latency, tokens and cost per model in the window (same filters)
q(f"""
SELECT model_key, count(*) AS calls, sum(is_error) AS errors,
       round(avg(latency_ms)) AS avg_latency_ms,
       percentile_approx(latency_ms, 0.95) AS p95_latency_ms,
       sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
       round(sum(input_cost_usd), 6) AS input_cost_usd,
       round(sum(output_cost_usd), 6) AS output_cost_usd,
       round(sum(total_cost_usd), 6) AS total_cost_usd
FROM {OBS}.v_model_call_cost
WHERE {filters}
GROUP BY model_key
ORDER BY total_cost_usd DESC
""")

# failed calls with the gateway status code (429 = rate limited, 5xx = upstream error)
q(f"""
SELECT event_time, model_key, use_case_id, purpose, status_code, error_category, latency_ms
FROM {OBS}.v_model_call_cost
WHERE event_time >= current_timestamp() - INTERVAL {DAYS} DAYS AND status = 'error'
ORDER BY event_time DESC
""")

# daily trend (pre-aggregated view)
q(f"""
SELECT * FROM {OBS}.v_model_cost_daily
WHERE event_date >= current_date() - {DAYS}
ORDER BY event_date DESC, total_cost_usd DESC
""")

# vector search queries
q(f"""
SELECT event_time, index_name, embedding_model_key, num_results_requested, num_results,
       latency_ms, status
FROM {OBS}.vector_query
WHERE event_time >= current_timestamp() - INTERVAL {DAYS} DAYS
ORDER BY event_time DESC
""")

# COMMAND ----------

# DBTITLE 1,2C. Observability join examples
# 1) our event vs the gateway's record of the same call (join key: request tag lh_event_id).
#    Gateway rows arrive with a delay; gateway_* columns stay NULL until they do.
q(f"""
SELECT i.event_time, i.model_key, i.purpose, i.status,
       i.latency_ms AS app_latency_ms, g.latency_ms AS gateway_latency_ms,
       i.input_tokens AS app_input_tokens, g.input_tokens AS gateway_input_tokens,
       i.output_tokens AS app_output_tokens, g.output_tokens AS gateway_output_tokens,
       g.status_code, g.destination_model, g.requester IS NOT NULL AS has_requester
FROM {OBS}.model_invocation i
LEFT JOIN {OBS}.v_gateway_usage g ON g.event_id = i.event_id
WHERE i.event_time >= current_timestamp() - INTERVAL 7 DAYS
ORDER BY i.event_time DESC
""")

# 2) cross-schema: each event with the catalog version in force at call time (as-of join on
#    the type-2 history) and the price in force on the call date -> cost per call
q(f"""
SELECT i.event_time, i.model_key, c.status AS catalog_status_then, c.family,
       i.input_tokens, i.output_tokens, p.input_usd_per_1m_tokens, p.output_usd_per_1m_tokens,
       round((coalesce(i.input_tokens, 0) * p.input_usd_per_1m_tokens
            + coalesce(i.output_tokens, 0) * p.output_usd_per_1m_tokens) / 1e6, 8) AS cost_usd
FROM {OBS}.model_invocation i
LEFT JOIN {MKT}.model_catalog c
  ON c.model_key = i.model_key
 AND i.event_time >= c.valid_from AND (c.valid_to IS NULL OR i.event_time < c.valid_to)
LEFT JOIN {MKT}.v_model_price p
  ON p.model_key = i.model_key
 AND i.event_date >= p.effective_from AND (p.effective_to IS NULL OR i.event_date < p.effective_to)
ORDER BY i.event_time DESC
""")

# 3) gateway traffic that no application recorded, by model service: other workloads in the
#    workspace (jobs, notebooks) or another schema prefix. Anti join = "in the gateway, not ours".
q(f"""
SELECT g.model_service, g.use_case_id, count(*) AS calls, sum(g.input_tokens) AS input_tokens,
       min(g.event_time) AS first_seen, max(g.event_time) AS last_seen
FROM {OBS}.v_gateway_usage g
LEFT ANTI JOIN {OBS}.model_invocation i ON i.event_id = g.event_id
GROUP BY g.model_service, g.use_case_id
ORDER BY calls DESC
""")

# 4) evaluation runs with their failed criteria (empty until lighthouse eval writes them)
q(f"""
SELECT r.finished_at, r.prompt_version, r.model_key, r.combined_score, r.verdict,
       e.criterion, e.kind, count(*) AS failed_cases
FROM {OBS}.eval_run r
JOIN {OBS}.eval_case_result e ON e.run_id = r.run_id
WHERE NOT e.passed
GROUP BY ALL
ORDER BY r.finished_at DESC, failed_cases DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `ai_knowledge`: documents, chunks and the vector index

# COMMAND ----------

# DBTITLE 1,3A. Knowledge tables and row counts
# Tables in <prefix>ai_knowledge
# ---------------------------------------------------------------------------------------------
# source_document        knowledge-base documents (synthetic examiner guidance)
#     doc_id, title, doc_type, product_line (std|ltd|life|accident|critical_illness|dental|all),
#     version, source_uri, content_hash, classification (synthetic), status, ingested_at
# document_chunk         embedded chunks; the source of the Delta Sync vector index
#     chunk_id (PK), doc_id, chunk_index, text, char_count, embedding ARRAY<FLOAT> (1024-d),
#     embedding_model_key, embedding_model_service, embedding_dimensions, content_hash,
#     embedded_at
# document_chunk_index   Mosaic AI Vector Search index (Delta Sync, self-managed embeddings).
#     Not a table: query it with the vector_search() SQL function (cell 3C).
# ---------------------------------------------------------------------------------------------
q(f"""
SELECT 'source_document' AS table_name, count(*) AS row_count FROM {KN}.source_document
UNION ALL SELECT 'document_chunk', count(*) FROM {KN}.document_chunk
UNION ALL SELECT 'document_chunk (embedded)', count(*) FROM {KN}.document_chunk WHERE embedding IS NOT NULL
""")
q(f"SHOW TABLES IN {KN}")

# COMMAND ----------

# DBTITLE 1,3B. Knowledge queries with filters
# Filters: change and re-run. None = no filter.
PRODUCT_LINE = "life"         # std | ltd | life | accident | critical_illness | dental | all | None
DOC_ID = "kb-life-evidence"   # a document for the chunk query
TEXT_CONTAINS = "beneficiary" # case-insensitive text filter on chunks

q(f"""
SELECT doc_id, title, doc_type, product_line, version, classification, status, ingested_at
FROM {KN}.source_document
{f"WHERE product_line = '{PRODUCT_LINE}'" if PRODUCT_LINE else ""}
ORDER BY doc_id
""")

# chunks of one document; the 1024 floats are summarised by size() instead of shown
q(f"""
SELECT chunk_id, chunk_index, char_count, size(embedding) AS dimensions, embedding_model_key,
       embedded_at, text
FROM {KN}.document_chunk
WHERE doc_id = '{DOC_ID}'
ORDER BY chunk_index
""")

# keyword filter over all chunks
q(f"""
SELECT doc_id, chunk_index, text
FROM {KN}.document_chunk
WHERE lower(text) LIKE lower('%{TEXT_CONTAINS}%')
ORDER BY doc_id, chunk_index
""")

# COMMAND ----------

# DBTITLE 1,3C. Knowledge join examples and vector search
INDEX = f"{KN}.document_chunk_index"

# 1) chunks with their document (title, product line) and chunk statistics per document
q(f"""
SELECT d.product_line, d.doc_id, d.title, count(c.chunk_id) AS chunks,
       sum(c.char_count) AS characters, max(c.embedded_at) AS last_embedded,
       first(c.embedding_model_key) AS embedding_model
FROM {KN}.source_document d
LEFT JOIN {KN}.document_chunk c ON c.doc_id = d.doc_id
GROUP BY d.product_line, d.doc_id, d.title
ORDER BY d.product_line, d.doc_id
""")

# 2) cross-schema: embedding model of the chunks vs the catalog (dimensions must match)
q(f"""
SELECT c.embedding_model_key, m.model_service, m.status, m.embedding_dimensions AS catalog_dims,
       min(size(c.embedding)) AS min_dims, max(size(c.embedding)) AS max_dims, count(*) AS chunks
FROM {KN}.document_chunk c
LEFT JOIN {MKT}.v_model_catalog_current m ON m.model_key = c.embedding_model_key
GROUP BY ALL
""")

# 3) vector search: "more like this chunk". The query vector is a stored chunk embedding, so no
#    model call is made. (A free-text query needs an embedding through Unity Gateway first:
#    see notebook 02 / lighthouse_poc.platform.knowledge.) Joined back to the document title.
q(f"""
SELECT v.doc_id, v.chunk_index, round(v.search_score, 4) AS score, d.title, v.text
FROM vector_search(
       index => '{INDEX}',
       query_vector => (SELECT embedding FROM {KN}.document_chunk
                        WHERE doc_id = '{DOC_ID}' AND chunk_index = 0),
       num_results => 5) v
JOIN {KN}.source_document d ON d.doc_id = v.doc_id
ORDER BY score DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Vector search explorer: sample queries and retrieval strategies
# MAGIC
# MAGIC Sections 1-3 query the tables directly. This section asks the **index** questions in plain
# MAGIC language, the way the assistant will. A query is embedded through Unity Gateway with the same
# MAGIC model as the chunks (`embed-default`), then searched with the strategy you pick.
# MAGIC
# MAGIC | Strategy | How it retrieves | Strong at | Cost |
# MAGIC |---|---|---|---|
# MAGIC | `dense` | ANN on the query embedding only | paraphrases, synonyms ("passed away" ~ "death certificate") | 1 index query |
# MAGIC | `hybrid` | keyword (full-text) + vector, rankings fused | exact terms, codes, field names (`benefit_determination`) | 1 index query |
# MAGIC | `filtered_hybrid` | a rules-based product router adds a metadata filter (`doc_id IN` that product's documents + the cross-product guide), then hybrid | stopping other products' look-alike checklists from ranking first | 1 index query; routing rules to maintain |
# MAGIC | `parent_document` | hybrid over chunks, grouped per document (best chunk), returns the **whole document** | giving the model complete context | 1 index query, larger context |
# MAGIC | `llm_rerank` | hybrid top 8 re-ordered by `claims-small` through the gateway | putting the best passage first | +1 model call per query (latency, tokens) |
# MAGIC
# MAGIC Databricks also offers a built-in reranker; it is **not enabled on Free Edition** ("Reranking is not
# MAGIC yet enabled for this workspace"), so `llm_rerank` uses an approved catalog model instead.
# MAGIC
# MAGIC **How the comparison scores strategies:** each of the 15 sample queries
# MAGIC (`data/knowledge/sample_queries.jsonl`, synthetic) names the document that answers it. Scoring is per
# MAGIC document: **hit@1** = the right document ranked first, **hit@3** = in the top 3, **MRR** = average of
# MAGIC 1 / rank of the right document. Latency is search time (plus the rerank call), excluding the
# MAGIC shared query embedding. Queries are tagged `keyword`, `paraphrase` or `exact_term`, so you can see
# MAGIC which kind each strategy handles.
# MAGIC
# MAGIC Cells: **4A** setup (pick a strategy in the widgets), **4B** sample queries, **4C** run the selected
# MAGIC strategy (or your own query), **4D** compare all five strategies and get a recommendation.

# COMMAND ----------

# DBTITLE 1,4A. Vector search setup (pick a strategy in the widgets)
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))       # project code next to notebooks/

from lighthouse_poc.config import load_profile
from lighthouse_poc.logging_config import configure_logging
from lighthouse_poc.platform.knowledge import vector_search_client
from lighthouse_poc.platform.model_catalog import load_model_catalog
from lighthouse_poc.platform.naming import PlatformNames
from lighthouse_poc.platform.retrieval import (KnowledgeCorpus, build_explorer, load_sample_queries,
                                               rank_matrix, recommend, result_rows, summarise)
from lighthouse_poc.platform.runtime import (catalog_chat_client, catalog_embedding_client,
                                             warehouse_engine)

STRATEGIES = ["dense", "hybrid", "filtered_hybrid", "parent_document", "llm_rerank"]
dbutils.widgets.dropdown("strategy", "hybrid", STRATEGIES)
dbutils.widgets.dropdown("top_k", "3", ["1", "3", "5"])
dbutils.widgets.text("custom_query", "")                 # empty = run the 15 sample queries
dbutils.widgets.dropdown("record_usage", "true", ["true", "false"])
dbutils.widgets.text("vector_search_endpoint", "lighthouse-vs")

logger = configure_logging()
os.environ.setdefault("DATABRICKS_HOST", WorkspaceClient().config.host)
os.environ["LIGHTHOUSE_CATALOG"], os.environ["LIGHTHOUSE_PLATFORM_PREFIX"] = CATALOG, PREFIX
profile = load_profile("databricks")
names = PlatformNames(CATALOG, PREFIX)

# what the strategies need besides the index: product lines (router) and chunk texts (parent docs)
corpus = KnowledgeCorpus.from_rows(
    [r.asDict() for r in spark.sql(f"SELECT doc_id, title, product_line FROM {KN}.source_document").collect()],
    [r.asDict() for r in spark.sql(f"SELECT doc_id, chunk_index, text FROM {KN}.document_chunk").collect()])

models = {m.key: m for m in load_model_catalog()}
embed_model, rerank_model = models["embed-default"], models["claims-small"]
index = vector_search_client().get_index(
    endpoint_name=dbutils.widgets.get("vector_search_endpoint").strip(), index_name=names.vector_index)
explorer = build_explorer(index, names.vector_index, corpus,
                          embed_model, catalog_embedding_client(profile, embed_model, logger),
                          rerank_model, catalog_chat_client(profile, rerank_model, logger), logger)
sample_queries = load_sample_queries()

# grounded answers (cell 4E): answer only from the retrieved knowledge, else "I do not have this information"
from lighthouse_poc.platform.grounding import GroundedAnswerer
answerer = GroundedAnswerer(rerank_model, catalog_chat_client(profile, rerank_model, logger), logger)
explorer.event_sources.append(answerer)          # its model calls are recorded with the others


def record_usage():
    """Write this run's model calls and index queries to ai_observability (no query text)."""
    if dbutils.widgets.get("record_usage") != "true":
        return
    engine = warehouse_engine(profile, names, logger)
    try:
        print("usage events written:", explorer.flush_events(names, engine, logger))
    finally:
        engine.dispose()


print(f"Index {names.vector_index}: {index.describe().get('status', {}).get('indexed_row_count')} chunks;"
      f" {len(corpus.product_lines)} documents; {len(sample_queries)} sample queries")
print("Selected strategy:", dbutils.widgets.get("strategy"))

# COMMAND ----------

# DBTITLE 1,4B. The sample queries
# 15 synthetic questions an examiner might ask, each with the document that answers it.
# style: keyword (shares words with the document), paraphrase (different words, same meaning),
# exact_term (a field name or code).
display(spark.createDataFrame([q.__dict__ for q in sample_queries]))

# COMMAND ----------

# DBTITLE 1,4C. Run the selected strategy (or your own query)
from lighthouse_poc.platform.retrieval import SampleQuery

strategy = dbutils.widgets.get("strategy")
top_k = int(dbutils.widgets.get("top_k"))
custom = dbutils.widgets.get("custom_query").strip()
queries = [SampleQuery("custom", custom)] if custom else sample_queries

results = explorer.run([strategy], queries, k=top_k)
print(f"Strategy {strategy}: {explorer.strategies[strategy].summary}")
if not custom:
    [s] = summarise(results)
    print(f"hit@1 {s.hit_at_1:.0%} | hit@{min(3, top_k)} {sum(r.hit_at(3) for r in results) / len(results):.0%}"
          f" | MRR {s.mrr:.3f} | avg latency {s.avg_latency_ms:.0f} ms")
# one row per retrieved chunk (or document); expected = it is the document that answers the query
display(spark.createDataFrame(result_rows(results)))
record_usage()

# COMMAND ----------

# DBTITLE 1,4D. Compare all strategies on the sample queries
results = explorer.run(STRATEGIES, sample_queries, k=3)
summaries = summarise(results)
display(spark.createDataFrame([s.as_row() for s in summaries]))

# where each strategy ranked the right document, per query (1 = first; null = not in the top 3)
display(spark.createDataFrame(rank_matrix(results)))

best, best_free = recommend(summaries), recommend(summaries, allow_extra_calls=False)
print(f"Best overall:                   {best.strategy} (MRR {best.mrr:.3f}, hit@1 {best.hit_at_1:.0%}, "
      f"{best.avg_latency_ms:.0f} ms, {best.extra_model_calls} extra model calls)")
print(f"Best without extra model calls: {best_free.strategy} (MRR {best_free.mrr:.3f}, "
      f"hit@1 {best_free.hit_at_1:.0%}, {best_free.avg_latency_ms:.0f} ms)")
record_usage()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading the comparison
# MAGIC
# MAGIC First run (2026-11-27, dev prefix, top 3):
# MAGIC
# MAGIC | Strategy | hit@1 | hit@3 | MRR | avg latency | extra calls |
# MAGIC |---|---|---|---|---|---|
# MAGIC | dense | 0.93 | 1.00 | 0.956 | ~420 ms | 0 |
# MAGIC | hybrid | 0.87 | 1.00 | 0.911 | ~390 ms | 0 |
# MAGIC | filtered_hybrid | 0.93 | 1.00 | 0.956 | ~390 ms | 0 |
# MAGIC | parent_document | 0.87 | 1.00 | 0.911 | ~440 ms | 0 |
# MAGIC | llm_rerank | **1.00** | 1.00 | **1.000** | ~2,400 ms | 15 |
# MAGIC
# MAGIC - **Every strategy finds the right document in its top 3.** They differ only in what comes
# MAGIC   **first**, which matters when the assistant uses just the top passage.
# MAGIC - **`llm_rerank` is the most accurate**: it fixed both rank-1 misses. It costs about 6× the latency
# MAGIC   and one model call (tokens) per query.
# MAGIC - **`dense` and `filtered_hybrid` tie** among strategies with no extra calls (MRR 0.956), but they
# MAGIC   miss different queries. `dense` put the **dental** checklist first for an LTD question (q03:
# MAGIC   "treating ... records" looks like dental "treatment"). That is a wrong-product answer. Routing
# MAGIC   "elimination period" to LTD fixed it in `filtered_hybrid`. `filtered_hybrid` missed q15, ranking the
# MAGIC   generic handoff guide 3rd, because no product was named and the router added no filter.
# MAGIC   `recommend()` breaks the tie on latency, which is noise at ~400 ms, so the printed "best without
# MAGIC   extra calls" can flip between the two from run to run.
# MAGIC - **`hybrid` was not better than `dense` here**: with 13 short chunks, keyword matching on common
# MAGIC   words ("records", "ends") adds noise. It earns its place on exact terms and codes
# MAGIC   (q14 `benefit_determination`).
# MAGIC - **`parent_document`** ranks like hybrid (it groups hybrid's results by document) but gives the
# MAGIC   model whole documents. Choose it for *what goes into the prompt*, not for ranking.
# MAGIC
# MAGIC **Recommendation for the assistant:** `filtered_hybrid` as the default. It ties `dense` on the score,
# MAGIC but its miss is benign while `dense`'s miss surfaces another product's checklist, and it keeps
# MAGIC hybrid's exact-term matching. Return parent documents as context. Add `llm_rerank` only where the
# MAGIC first passage must be right and about 2 s per query is acceptable.
# MAGIC
# MAGIC **Caveats:** 15 queries over 7 documents is a smoke test, not a benchmark. One query changes a score
# MAGIC by about 0.07, and the reranker's answers can vary between runs. Before choosing for production,
# MAGIC grow the query set (more paraphrases, ambiguous and multi-document questions, real examiner
# MAGIC phrasing reviewed by an SME) and re-run 4D. Your run's table above is the result to trust.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4E. One query, step by step (showcase and negative testing)
# MAGIC Edit `QUERY`, `STRATEGY` and `TOP_K` at the top of the next cell and run it (Shift+Enter). It
# MAGIC shows each step of the retrieval: the embedding call, what the strategy did (routing filter,
# MAGIC rerank...), the ranked results, a **relevance check**, and, when the query is one of the 15
# MAGIC samples, whether the expected document was found.
# MAGIC
# MAGIC **Negative testing.** Vector search *always* returns the nearest chunks, even for "best pizza in
# MAGIC Rome", so "got results" never means "found an answer". The relevance check uses the **dense
# MAGIC similarity of the best chunk**, whatever strategy you pick. Hybrid scores are rank-fused and
# MAGIC normalised, so the top hit scores about 1.0 even for nonsense. Calibrated on this index:
# MAGIC
# MAGIC | Dense top score | Verdict | Seen for |
# MAGIC |---|---|---|
# MAGIC | below 0.50 | **NO RELEVANT KNOWLEDGE** | pizza, wifi router, cricket, bread, capital cities, weather (0.42-0.52) |
# MAGIC | 0.50-0.55 | **BORDERLINE** | weakest sample paraphrases (0.53-0.54), "latest football results" (0.52) |
# MAGIC | 0.55 and above | **RELEVANT** | most sample queries (0.55-0.72) |
# MAGIC
# MAGIC Try these: an off-topic question (should say NO RELEVANT KNOWLEDGE), a typo-heavy sample, a
# MAGIC decision-seeking question ("Is this claim fraudulent and should we deny it?". It scores RELEVANT,
# MAGIC and the handoff guide that says the assistant never decides fraud ranks 2nd, behind the accident
# MAGIC checklist), and an **out-of-scope insurance**
# MAGIC question ("pet insurance claim for a broken dog leg", which scored 0.61). That last one is the
# MAGIC known gap: it looks relevant to embeddings, so a score threshold cannot reject it. Scope rules
# MAGIC (the product router) must. **Step 6 closes that gap**: see the grounding table below.
# MAGIC
# MAGIC ### Step 6: grounded answer ("answer only from what we embedded")
# MAGIC Retrieval finds passages; grounding decides whether they actually answer the question. Raw
# MAGIC similarity can never reach 0 (any English sentence scores about 0.4 against these chunks), so the
# MAGIC **grounding score** is built from three checks, and each one can only lower it:
# MAGIC
# MAGIC | Check | How | Refuses when |
# MAGIC |---|---|---|
# MAGIC | 1. Retrieval confidence | dense similarity of the best chunk mapped to 0..1: `(sim - 0.50) / (0.70 - 0.50)`, clipped | below 0.50 the score is **exactly 0** and no model is called (pizza, weather...) |
# MAGIC | 2. Answerability | `claims-small` (through Unity Gateway, tagged `grounded_answer`) reads the retrieved passages and answers **only** from them, citing passage numbers and quoting the supporting sentences, or replies "not answerable" | the passages are about something else (pet insurance looks like a claim question but no passage covers it) |
# MAGIC | 3. Quote verification | code checks every quote appears **verbatim** in a passage it cites | the model invented its support |
# MAGIC
# MAGIC `grounding score = confidence of the best cited chunk x share of quotes verified`, and **0 with the
# MAGIC answer "I do not have this information in the knowledge base."** whenever a check refuses. The model
# MAGIC is also told never to decide coverage, fraud, eligibility or amounts, so a decision-seeking question
# MAGIC gets what the guidance says about *who* decides. Any model failure refuses too (fail closed).

# COMMAND ----------

# DBTITLE 1,4E. Run ONE query with ONE strategy (edit the three lines below)
# The 15 sample queries (copy one into QUERY, or type your own):
#   q01  What documents should a short-term disability claim file contain?                    -> kb-std-evidence
#   q02  The last day worked on the claimant form does not match what the employer reported   -> kb-std-evidence
#   q03  Treating provider records stop before the elimination period ends                    -> kb-ltd-evidence
#   q04  A functional capacity evaluation is mentioned but nobody attached it                 -> kb-ltd-evidence
#   q05  Which papers do we need after the insured person has passed away?                    -> kb-life-evidence
#   q06  One of the named beneficiaries has no proof of identity                              -> kb-life-evidence
#   q07  Date of birth on the certificate is different from the policy record                 -> kb-life-evidence
#   q08  The injury date in the incident report and the first doctor visit disagree           -> kb-accident-evidence
#   q09  Claimant referred to a police report that was never sent to us                       -> kb-accident-evidence
#   q10  Specialist report refers to test results that are not in the file                    -> kb-ci-evidence
#   q11  The illness named on the claim form differs from what the consultant confirmed       -> kb-ci-evidence
#   q12  A procedure on the itemised statement has no treatment date                          -> kb-dental-evidence
#   q13  The amount charged is higher than the estimate sent before treatment                 -> kb-dental-evidence
#   q14  What value must benefit_determination always have?                                   -> kb-handoff-format
#   q15  How should the note for the human examiner be organised and what must it never decide? -> kb-handoff-format
#
# Negative tests to try:
#   "What is the best pizza in Rome?"                    off-topic           -> NO RELEVANT KNOWLEDGE; step 6 score 0
#   "Is this claim fraudulent and should we deny it?"    decision-seeking    -> RELEVANT; handoff guide (never decides) ranks 2nd
#   "pet insurance claim for a broken dog leg"           out-of-scope product -> scores RELEVANT; step 6 refuses
#   "benifit determinaton value?"                         typos               -> still finds the handoff guide first (dense)
#
# Strategies: dense | hybrid | filtered_hybrid | parent_document | llm_rerank   (see the table in section 4)

QUERY = "What is the best pizza in Rome?"
STRATEGY = "filtered_hybrid"
TOP_K = 3

from lighthouse_poc.platform.retrieval import run_single

report = run_single(explorer, QUERY, STRATEGY, k=TOP_K, samples=sample_queries)
r = report.result
print(f"Query     : {report.query.query}  [{report.query.query_id}]")
print(f"Strategy  : {STRATEGY}: {explorer.strategies[STRATEGY].summary}")
print(f"Step 1    : query embedded with {embed_model.key} ({embed_model.model_service}, "
      f"{embed_model.embedding_dimensions}-d) through Unity Gateway"
      + (f" in {report.embedding_ms} ms" if report.embedding_ms else " (cached from an earlier run)"))
print(f"Step 2    : products detected by the router: {report.routed_products or 'none'}"
      f"{'  | ' + r.note if r.note else ''}")
print(f"Step 3    : {len(r.hits)} results in {r.latency_ms} ms"
      f"{f', {r.model_calls} extra model call(s)' if r.model_calls else ''}")
display(spark.createDataFrame(result_rows([r], snippet=220)) if r.hits else spark.createDataFrame(
    [{"result": "no results"}]))
score = "n/a" if report.dense_top_score is None else f"{report.dense_top_score:.3f}"
print(f"Step 4    : relevance = {report.relevance}  (best dense similarity {score} -> "
      f"{report.dense_top_doc}; thresholds 0.50 / 0.55)")
print(f"Step 5    : {report.expected_check()}")

from lighthouse_poc.platform.grounding import ground

g = ground(answerer, report)                 # step 6: answer only from the retrieved passages
print(f"Step 6    : grounding score = {g.grounding_score:.2f}  (retrieval confidence "
      f"{g.retrieval_confidence:.2f}; {'model checked' if g.model_called else 'no model call needed'}"
      + (f", {g.latency_ms} ms" if g.latency_ms else "") + ")")
print(f"  Answer  : {g.answer}")
if g.grounded:
    print(f"  Sources : passages {g.citations} -> {', '.join(g.cited_chunks)}")
    for quote in g.verified_quotes:
        print(f"  Quote OK: \"{quote}\"")
else:
    print(f"  Why     : {g.refusal_reason}")
for quote in g.unverified_quotes:
    print(f"  Quote NOT found in the passages: \"{quote}\"")
record_usage()   # the embedding (and rerank) calls are tracked like every other model call


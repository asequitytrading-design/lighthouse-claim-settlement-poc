# Lighthouse demo: thin UI, chat agent with MCP, API layer (design and plan)

Status: design agreed 2026-11-27, demo **Monday 2026-11-30** (fixed). Source of the ask: the
planning meeting (2026-11-27): an agent that
stitches 2-3 tools through MCP, connects to the Lakehouse (Unity Catalog), shows model switching
and evaluation results, behind a thin UI, on dummy data.

## 1. Decisions

| # | Decision | Notes |
|---|---|---|
| D1 | Demo Monday, scope tiers below | Must / Should / Later |
| D2 | **Our own agent** is primary; **Agent Bricks** side by side if it works on Free Edition | build-vs-buy evidence for Arjun |
| D3 | **Lakebase** only if accessible, **started for the demo and stopped otherwise** (protect the Free Edition account); local dev on **Postgres + pgvector** (Docker) with the same SQL | SQLite only as a last fallback |
| D4 | **Synthetic data only** | real data later in the client's environment and subscription |
| D5 | **Own conversational UI** with a current chat design (not stock Streamlit) | React + TypeScript + Tailwind, built and served by the API |
| D6 | UI runs locally now; hosting decided later (Databricks Apps is available) | one process: API serves the built UI |

## 2. Feasibility (checked on the workspace 2026-11-27)

| Capability | Result |
|---|---|
| Tool calling via Unity Gateway | **32/32 correct** (4 models x 4 scenarios x 2): right tool, valid args, no tool for small talk; median 1.1-2.1 s; one 62 s outlier (llama) -> timeout + retry |
| Managed MCP | vector search server lists our index tool; Genie server lists `query_space` / `poll_response`; `system.ai` functions server lists 16 tools |
| Genie | available (starter space exists); we create one over synthetic claim tables |
| Databricks Apps | API available, no apps yet (later hosting) |
| Lakebase | API answers (no instances); creation to be tried once, then stopped |
| Agent Bricks | API answers (no tiles); to be tried once from the UI |

## 3. Demo story: claims examiner copilot

1. Examiner: "Prepare the handoff for claim CSET-LTD-002." Agent calls **get_claim_case** (UC
   function over synthetic `claim_case`, through managed UC-functions MCP) and
   **search_guidance** (managed Vector Search MCP on our knowledge index, grounded: refuses with
   "I do not have this information" when the guidance does not cover it), then drafts the handoff
   with the `@champion` prompt. Never decides coverage, fraud or amounts.
2. "How many LTD claims are missing a functional capacity evaluation?" -> **ask_portfolio** (Genie
   over synthetic claim tables, managed Genie MCP).
3. Switch the model in the composer, ask again: answer, latency and cost per message differ.
4. **Dashboard** page: the calls just made, per model: tokens, latency, cost (tracked through
   Unity Gateway tags, activity `demo`).
5. **Evals** page: latest evaluation runs and per-case scores; link to MLflow.
6. (If available) the same question to the **Agent Bricks** agent, side by side.
7. Negative test: an off-topic or out-of-scope question -> grounded refusal.

## 4. Architecture

```
Browser (localhost)                 lighthouse-api (FastAPI, local)                 Databricks (Free Edition)
React chat UI  ── /api/* (JSON, ──► routers: chat (SSE), playground, models,  ──►  Unity Gateway (models, tags)
(built, served     SSE streams)     usage, evals, health                            Managed MCP: UC functions,
 by the API)                        services: AgentService, PlaygroundService,         Vector Search, Genie
                                    UsageService, EvalService, SessionStore,        SQL warehouse (UC tables)
                                    SemanticCache                                   Lakebase (demo only)
                                    reuses src/lighthouse_poc (gateway, usage
                                    tracker, grounding, retrieval, platform)       local: Postgres+pgvector
```

Principles: the UI talks only to the API; the API reuses the package (no logic in routers);
every model call goes through the usage tracker (activity `demo`, purpose `app` / `tool` /
`grounded_answer`); identities: local = the developer's Databricks CLI login; hosted = app
service principal (later, on-behalf-of user where available).

## 5. API contract (v1)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | workspace, gateway, warehouse, MCP, store reachability |
| GET | `/api/models` | approved app models from the governed catalog (key, family, price) |
| POST | `/api/chat` | `{session_id?, message, model}` -> **SSE**: `token`, `tool_call`, `tool_result`, `citation`, `usage`, `done`, `error` |
| GET | `/api/sessions`, `/api/sessions/{id}` | conversation list and history (session store) |
| POST | `/api/playground/complete` | `{model, system?, prompt, params}` or `{prompt_ref, variables}` -> text, tokens, latency, cost |
| GET | `/api/usage/summary?hours=` | calls, tokens, cost, p50/p95 per model (from `v_model_call_cost`, cached 60 s) |
| GET | `/api/usage/recent?limit=` | latest calls (no text) |
| GET | `/api/evals/runs`, `/api/evals/runs/{run_id}` | `eval_run` / `eval_case_result` + MLflow links |
| Later | `/api/prompts`, `/api/eval`, `/api/promote` | prompt registry and eval in the UI (promote stays a human action) |

## 6. Agent design

- Loop: model (selected key, via `GatewayChatClient` extended with `tools`) -> tool calls ->
  MCP client calls -> results back -> final answer; max 4 tool rounds; per-call timeout 45 s,
  one retry; the whole turn streams events to the UI.
- Tools (MCP, managed by Databricks; our agent is the MCP client):
  - `get_claim_case(case_id)`: UC SQL function `<schema>.get_claim_case` over `claim_case`
  - `search_guidance(query)`: Vector Search MCP on `ai_knowledge.document_chunk_index`, then our
    grounding check (score, verified quotes) before the answer may use it
  - `ask_portfolio(question)`: Genie space "Lighthouse claims (synthetic)"
- Guardrails: system prompt with the non-decision rule; output check that no decision language
  appears (reuse rubric criterion); tool results never stored in logs; case IDs only.
- Abstraction `ToolProvider` (MCP now; direct SDK fallback if an MCP server is unavailable).

## 7. Data (synthetic) in Unity Catalog

- `<prefix>ai_knowledge` exists (documents, chunks, index).
- New `<schema>.claim_case` (~40 synthetic cases across the six products, built from the golden
  set pattern) + `claim_document` (documents received / missing) + UC function `get_claim_case`.
  Created by an Alembic revision (0005) in the marketplace schemas, or in the prompt schema for
  the demo (decide in phase 1; the revision is preferred).
- Genie space over `claim_case` / `claim_document` (created once via API).

## 8. Session store and semantic cache (D3)

- `SessionStore` interface: Postgres implementation (local Docker `pgvector/pgvector:pg16`;
  Lakebase in the demo) and an in-memory/SQLite fallback.
- `SemanticCache` for **guidance questions only** (never case-specific answers): key = query
  embedding + knowledge index version + model; hit when cosine >= 0.95; entries invalidated on
  re-index; the dashboard shows hits, tokens and cost saved.
- Lakebase lifecycle: `scripts/lakebase.py start|stop|status` (instance stopped when not in use).

## 9. UI (D5)

Modern conversational layout (in the style of current chat products):
- Left rail: new chat, conversation list, navigation (Chat, Playground, Dashboard, Evals).
- Chat: streaming markdown answers; **tool cards** (tool, arguments, duration, result summary,
  expandable); **citations** with verified quotes; per-message footer: model, tokens, latency,
  cost; composer with model picker and suggested prompts; light/dark.
- Playground: model(s) side by side, system/user prompt, params, output, tokens/cost.
- Dashboard: KPI tiles and charts from `/api/usage/*`; link to the Lakeview dashboard.
- Evals: runs table, per-case scores, verdict chips, MLflow links.
- Stack: Vite + React + TypeScript + Tailwind; built into `app/web/dist` and served by FastAPI.

## 10. Implementation plan

| Phase | Work | Exit check |
|---|---|---|
| 0 | Branch, design doc, tool-calling check | done (32/32) |
| 1 | Synthetic claim data + UC function + Genie space; MCP client + `ToolProvider`; tool round trip from Python | **done 2026-11-27**: all three tools live via managed MCP |
| 2 | `AgentService` (loop, streaming events, guardrails, usage tags) + tests with fake tools | **done 2026-11-27**: 6 demo questions live (incl. fraud, pet insurance, off-topic refusals) |
| 3 | FastAPI app: chat SSE, models, playground, usage, evals, health; tests with fakes | **done 2026-11-27**: all endpoints live locally (port 8787) + `/api/warmup` |
| 4 | React UI: chat (streaming, tool cards, citations), playground, dashboard, evals | **done 2026-11-27**: all four pages verified (screenshots, light/dark) |
| 5 | Session store + semantic cache on local Postgres; Lakebase start/stop; try it once | cache hit shown; Lakebase started/stopped |
| 6 | Agent Bricks attempt (Knowledge Assistant on the guidance docs) + side-by-side panel | works, or documented as unavailable |
| 7 | Rehearsal: demo script, recorded fallback, rate-limit plan, README | dry run under 10 minutes |

Must = phases 1-4 and 7; Should = 5 and the Evals page; Later = 6 if time, prompt registry/eval
actions in the UI, hosting on Databricks Apps.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Free Edition rate limits (429) or slow calls live | small model default, timeouts, retry once, recorded fallback video |
| Latency outliers (62 s seen) | per-call timeout, show progress events, switch model |
| Lakebase or Agent Bricks unavailable | local Postgres; mark Agent Bricks "not available on Free Edition" |
| Capacity use on the free account | Lakebase stopped outside the demo; no CI deploys (manual only) |
| Decision language in answers | system rule + rubric check on the final answer |

## 12. KPIs (to add in `docs/metrics.md` as built)

Tool-call accuracy (baseline 32/32), agent turn latency p50/p95, tool latency per tool,
grounded-refusal rate on out-of-scope questions, cache hit rate and cost saved, cost per demo turn.

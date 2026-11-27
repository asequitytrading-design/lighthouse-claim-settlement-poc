# Lighthouse demo script (Monday 2026-11-30, ~10 minutes)

What Arjun (engagement lead) asked for (planning meeting,
2026-11-27): an agent that stitches 2-3 tools through MCP, connects to the Lakehouse (Unity
Catalog), shows model switching and evaluation results, behind a thin UI, on dummy data. All data
here is synthetic. Timings are from the dry run on Databricks Apps (2026-11-27).

## Before the demo (T-30 min)

```bash
python scripts/demo_env.py up        # Lakebase on, app on, warm-up (~5 min); prints the URL
```
- Open the URL in a browser logged in to the workspace; check the sidebar dot says **Databricks connected**.
- Optional: approve a prompt version (notebook 05 step 9) so the handoff uses the governed `@champion` template.
- Keep `apps/lighthouse_demo/fallback/*.png` open in another window (screenshots of every step).
- Default model `claims-alt-qwen-next` (composer). Avoid `claims-medium` live (reasoning model, p95 > 2 min).

## Flow

| # | Page / action | Say | Expect (dry run) |
|---|---|---|---|
| 1 | **Copilot** → suggestion "Missing documents + guidance" | "One question, two governed tools: the case file comes from a Unity Catalog function, the guidance from Vector Search, both through Databricks-managed **MCP** servers." Expand both tool cards. | ~8 s; case file: 3 missing; guidance with a **verified quote** chip and citation; cache hit on repeat |
| 2 | Same chat: "Prepare the handoff for claim CSET-LTD-002." | "Drafting uses the **approved prompt** from the Unity Catalog prompt registry (`@champion`); without an approved version it fails closed and writes from the case facts only." | ~8 s; sections + "Benefit determination: Not determined" |
| 3 | **New chat** → "How many long-term disability claims are missing a functional capacity evaluation?" | "Portfolio questions go to **Genie** over the same Unity Catalog tables: it writes the SQL." Open the card: SQL + rows. | ~11 s; **5** |
| 4 | Same chat: "Which of those came in through the broker channel?" | "Follow-ups query again; the copilot never answers from memory (a fact check sends unverified answers back)." | ~13 s; **2** (CSET-LTD-026, CSET-LTD-032) |
| 5 | **New chat** → "Is claim CSET-LIFE-003 fraudulent? Should we deny it?" | "It prepares, it never decides: guardrails enforce the boundary." Point at the green guardrail chips. | ~5 s; first sentence "a human claims examiner does", then the file facts |
| 6 | "We have a pet insurance claim for a broken dog leg. Which documents do we need?" | "Grounding: not in our knowledge, so it says so instead of improvising." | ~1-2 s; "I do not have this information in the knowledge base." |
| 7 | Composer model → `claims-alt-llama`, repeat question 1 | "**Model management**: same agent, another model, one click; the gateway routes and tracks every call." | ~8-10 s; same facts, different cost |
| 8 | **Playground**: 3 models, same prompt, Run | "Compare candidates side by side before switching: answer, latency, tokens, cost." | ~5 s |
| 9 | **Dashboard** (24 h) | "Observability: every call of the platform, per model, activity and cost, from Unity Catalog (gateway usage + app events)." Point at the cache tile. | loads ~4 s |
| 10 | **Evaluations** → click a run → MLflow link | "**Evaluation framework**: hard gates + rubric + LLM judges per prompt version; verdicts are evidence for a human reviewer; promote is a separate human step." | READY / BLOCKED runs, per-case matrix |

Close: "Everything runs on Databricks as the engine (Apps, Unity Gateway, Unity Catalog, MCP, Genie,
Vector Search, Lakebase, MLflow); the same code moves to the client's subscription with their
data."

## If something goes wrong

| Symptom | Do |
|---|---|
| "Retrying (empty model reply)" | wait; it retries once by itself |
| A turn > 30 s or an error | ask again, or switch the model to `claims-small`; then show the fallback screenshot |
| Genie slow (> 20 s) | talk through the SQL card while it runs; fallback `2-portfolio-genie.png` |
| App unavailable (502) | `python scripts/demo_env.py status`; local fallback: `apps/lighthouse_demo/run_local.sh` (same backends) |
| Health shows `sqlite (fallback ...)` | Lakebase off: chats work, history is not kept; fine for the demo |

## After the demo

```bash
python scripts/demo_env.py down      # stop the app, disable Lakebase compute (protects the free account)
```

## Talking points mapped to the meeting

- **Model management / decommissioning risk**: governed model catalog + Unity Gateway routing; switch per request; Playground to compare a replacement; evaluation runs per model before promoting.
- **Evaluation framework (hybrid)**: deterministic gates + weighted rubric + LLM judges, versioned golden set, CI gate; verdict is evidence, promote is human.
- **Observability / production monitoring**: every call tagged (use case, activity, prompt version, run), cost per model, latency percentiles; the same tables feed drift monitoring later.
- **Prompt registry**: versions and `@champion` in Unity Catalog; the agent serves only the approved version.
- **Knowledge / context**: grounded retrieval with refusal, semantic cache for repeated guidance questions.
- **Build vs buy**: own agent on managed Databricks building blocks (MCP, Genie, Vector Search, Lakebase); Agent Bricks comparison is the next step.

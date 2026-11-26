# Project ledger

Running record of progress, environment state and open actions. Update it at the end of each
session; newest entries first. Synthetic data only.

## Resume here (last updated 2026-11-26)

**Branch:** `feat/lighthouse-cli`, pushed, **no PR yet**. The user tests the UI demo flow and a
dev deployment manually, then asks for the PR.

**Next session, start with:**
1. Ask how the manual UI walkthrough went (`docs/mlflow-ui-guide.md`, "Demo script") and fix
   what they hit.
2. When asked, raise the PR for `feat/lighthouse-cli` → `main`. Merging deploys target `ci` (see
   action A4).
3. Then take up the open actions below in the order the user chooses.

Quick re-entry commands:
```bash
docker compose up -d && python scripts/seed_ui_demo.py        # local stack + UI demo state
python -m pytest -q                                            # 166 tests, offline
lighthouse prompt list --prompt-name claim_intake_triage
export DATABRICKS_CONFIG_PROFILE=lighthouse                    # Databricks CLI profile
```

## Open action items

| # | Action | Owner | Notes |
|---|---|---|---|
| A1 | Practise the UI demo: Playground edit → save v3 → Run judges → `lighthouse eval` → compare runs → `lighthouse promote` | User | Steps in `docs/mlflow-ui-guide.md`; local stack only |
| A2 | Raise the PR for `feat/lighthouse-cli` (7 commits) after the user's manual validation | Claude, when asked | The user merges from the CLI (`gh pr merge <n> --merge`) |
| A3 | First `@champion` approval: local `claim_settlement_review` / `claim_intake_triage`, and Databricks dev (`dev_asequitytrading_lighthouse.claim_settlement_review` v3, evidence run `16f8f14342704ea6900c48f4a41cb6c3`, mock) | User (human decision) | Until then the agent and `lighthouse try` fail closed by design. Prefer a real-model run as evidence (A5) |
| A4 | Target `ci` after the merge: schema `workspace.lighthouse` belongs to SP `lighthouse-ci`; the user gets `PERMISSION_DENIED` reading it, so the first `ci` approval needs a grant to the user or a promote run as the SP | User + Claude | Decide on the grant (USE SCHEMA, EXECUTE, MANAGE on prompts) or an SP-run promote |
| A5 | Real-model job run on dev: `databricks bundle run lighthouse_prompt_eval --params app_model=claims-medium,scorer_mode=llm-judges,judge_model=claims-judge` | Claude, when asked | Gives READY evidence based on real models, not the mock |
| A6 | Verify SP `lighthouse-ci` can call the Unity Gateway model services (real-model CI runs) | Claude | Not verified; mock CI works |
| A7 | Rotate the `lighthouse-ci` OAuth secret before **2027-02-23** | User | Commands in `docs/databricks-setup.md` step 7 |
| A8 | Bind the reviewer to an authenticated identity and enforce author ≠ approver (ADR-009, open) | Claude | `--reviewer` is free text today |
| A9 | Agree the rubric/judge weights (0.6/0.4) and the 0.8 threshold with a claims SME (ADR-008) | User | Per spec in `config/evals/` |
| A10 | `lighthouse prompt register` CLI command (planned in the developer guide) | Claude | Registration is via `scripts/register_prompt.py` or the Playground today |
| A11 | Explore the Databricks UI equivalent (AI Playground, prompts, judges on Unity Gateway); the UI seed is local only | Claude + User | `seed_ui_demo.py` refuses non-local tracking URIs |
| A12 | Clean-up (ask first): local prompts `gapcheck_claim_settlement_review`, `local.claim_settlement_review`; local experiments `/gapcheck-eval`, `/lighthouse-error-path-check`, `/Shared/lighthouse-claim-settlement-poc`, `gateway/zz-check`, `gateway/zz-check-claims-small` (created by Claude's tests), `gateway/kong-claims-small`/`-medium` (unused since the seed), test runs in `/lighthouse-eval-dev`; the `lighthouse-account` profile in `~/.databrickscfg` (no account access) | Claude, when asked | The two `zz-check` experiments are Claude's own leftovers |
| A13 | Optional: rotate `KONG_API_KEY`; Claude's output once showed its first 6 hex characters | User | Local-only key |
| A14 | Optional: raise Docker Desktop memory to 4 GB (2 GB caused an MLflow OOM earlier) | User | |

## Environment state (2026-11-26)

**Local stack** (`docker compose`): MLflow 3.18 at http://localhost:5000 (SQLite, proxied
artifacts, AI Gateway with encrypted keys, custom code judges enabled), Kong 3.9.3 + Postgres
(proxy :8100, Admin :8101, Manager :8102), Ollama models qwen3.5-4b-32k, qwen3.5-9b-32k,
gemma4-e4b-32k (judge).
- Experiment `/lighthouse-ui-sample` (id 10): prompts `claim_intake_triage` v1/v2 (`@candidate` →
  v2) and `claim_settlement_review` v1 linked; dataset `claim_intake_triage_golden`; 9 registered
  judges; evaluation runs v2 `002e680a…` (93.4%, READY) and v1 `9448c5f7…` (89.8%, READY).
- Gateway endpoints `kong-claims-small` / `-medium` (track into exp 10), `kong-claims-judge`.
- No `@champion` anywhere locally.

**Databricks Free Edition** (host `dbc-79305b3a-397f`, CLI profile `lighthouse`):
- dev target deployed from branch head `17206e1`; job run `299740435896324` SUCCESS (mock,
  deterministic; prompt v3 unchanged; baseline skipped because there is no champion).
- ci target deployed from `main` (PR #1 and #2 merged) by SP `lighthouse-ci` via OAuth M2M;
  prompt `workspace.lighthouse.claim_settlement_review` v1.
- GitHub environment `databricks`: `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, secret
  `DATABRICKS_CLIENT_SECRET` (expires 2027-02-23).

## Progress log

### 2026-11-26
- Seeded the local MLflow UI (`scripts/seed_ui_demo.py`, ADR-011): gateway endpoints, triage
  prompts v1/v2, dataset, 6 code judges (`ui_judges.py`) and 3 LLM judges on
  `gateway:/kong-claims-judge`; verified the UI loop (Playground-style call → trace → judges).
- Fixed: registered code judges must be self-contained (MLflow stores only the function body);
  clients need `MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS` too.
- Deployed dev from the branch and ran the job: SUCCESS, READY, baseline skipped (no champion).

### 2026-11-25
- Consumers load `@champion` and fail closed; the job evaluates each registered version against
  `@champion` (ADR-010).
- `lighthouse promote`: evidence-bound approval record, then alias move (ADR-009); verified
  locally and on Unity Catalog (UC tag keys reject dots; UC omits aliases on versions).
- `lighthouse eval`: hard gates + weighted rubric + LLM judges → combined review score and verdict,
  `--baseline` comparison (ADR-008); unsafe draft BLOCKED, safe draft READY.
- MLflow UI AI Gateway → Kong (Bearer→apikey and reasoning_effort shims in Kong; ADR-007); triage
  sample prompt and 11-case golden set.
- MLflow artifacts proxied by the server (the old local path broke every client upload).
- `lighthouse` CLI: prompt list/show/diff, cases, try; content-addressed drafts (ADR-006).
- CI deploys as SP `lighthouse-ci` (OAuth M2M; no OIDC on Free Edition); `ci` root path in the
  SP's home folder; PR #1 and #2 merged, first `main` deploy green.
- Earlier the same day: config profiles, Kong gateway, LLM judges, Databricks bundle, Unity
  Gateway real-model runs (ADR-001 to ADR-005).

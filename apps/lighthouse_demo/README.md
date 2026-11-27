# Lighthouse demo app (claims copilot, playground, dashboard, evaluations)

Thin UI over the `lighthouse-api` (FastAPI, `src/lighthouse_poc/api`). Synthetic data only.
Design and plan: [`docs/demo-thin-ui.md`](../../docs/demo-thin-ui.md).

## Run locally

```bash
apps/lighthouse_demo/run_local.sh          # builds the UI once, serves everything on http://localhost:8787
apps/lighthouse_demo/run_local.sh --dev    # API + hot-reload UI on http://localhost:5173
curl -X POST localhost:8787/api/warmup     # before a demo: loads the copilot, touches every backend
```

Prerequisites: `.venv` with `pip install -e '.[platform,app]'`, Node 20+, Databricks CLI profile
`lighthouse` (override with `DATABRICKS_CONFIG_PROFILE`), platform schemas at head
(`lighthouse platform db upgrade --prefix dev_<you>_`) and the Genie space
(`lighthouse platform genie ensure --prefix dev_<you>_`).

## Pages

| Page | URL | What it shows |
|---|---|---|
| Copilot | `#chat`, `#chat/<session>` | streaming answers, tool cards (case file, guidance, Genie, handoff), verified quotes, citations, model/tokens/latency/cost per answer |
| Playground | `#playground` | the same prompt on up to 4 governed models side by side |
| Dashboard | `#dashboard` | calls, tokens, cost, latency per model and activity from the Unity Catalog views |
| Evaluations | `#evals` | evaluation runs, verdicts, per-case criteria, MLflow links |

## Settings (environment)

See `src/lighthouse_poc/api/settings.py`: `LIGHTHOUSE_SESSION_STORE` (`sqlite` | `postgres`),
`LIGHTHOUSE_SQLITE_PATH`, `LIGHTHOUSE_PG_DSN`, `LIGHTHOUSE_APP_USE_CASE`, `LIGHTHOUSE_DASHBOARD_URL`.

## On Databricks (Databricks Apps + Lakebase)

```bash
# once: put your workspace IDs in .databricks/bundle/dev/variable-overrides.json (gitignored):
#   {"genie_space_id": "<from lighthouse platform genie ensure>", "warehouse_id": "<SQL warehouse id>"}
# once: deploy the app (Lakebase sessions + cache) and grant its identity access
databricks bundle deploy -t dev --var genie_space_id=<space> --var warehouse_id=<wh> --var session_store=lakebase
lighthouse --profile databricks platform app grants --prefix dev_<you>_ --principal <app SP id> --prompt-schema dev_<you>_lighthouse

python scripts/demo_env.py up       # enable Lakebase, start the app, warm it up (prints the URL)
python scripts/demo_env.py down     # stop the app and disable the Lakebase compute (Free Edition)
python scripts/demo_env.py status
```

## Live settings (no restart)

Notebook `notebooks/06_demo_app_controls.py`: status, start, **apply settings** (session store, default model, semantic cache + threshold, grounding check, Genie) into `<prefix>ai_marketplace.app_config` (revision 0006), stop. The running app re-reads them at most once a minute when requests arrive; see `/api/health` -> `live_settings`.

## Before the demo

Full script with timings and fallbacks: [`docs/demo-script.md`](../../docs/demo-script.md); screenshots of every step in `fallback/`.


1. Warm up (`POST /api/warmup`): cold calls take up to ~8 s, Genie 10-20 s.
2. For the governed handoff, approve a prompt version (notebook 05 step 9); without `@champion`
   the `draft_handoff` tool fails closed and the copilot builds the handoff from the case file.

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

## Before the demo

1. Warm up (`POST /api/warmup`): cold calls take up to ~8 s, Genie 10-20 s.
2. For the governed handoff, approve a prompt version (notebook 05 step 9); without `@champion`
   the `draft_handoff` tool fails closed and the copilot builds the handoff from the case file.

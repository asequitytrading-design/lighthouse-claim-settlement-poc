# Local development: MLflow + Kong AI gateway + Ollama

The local stack runs the same code path as Databricks. Only the profile changes:
`config/profiles/local.yaml` (Kong → Ollama) vs `config/profiles/databricks.yaml`
(Unity Gateway → workspace model services).

```text
scripts/evaluate.py --profile local --model claims-medium
   │  logical model name, apikey header, X-Correlation-ID = MLflow trace id
   ▼
Kong OSS 3.9 (localhost:8100/ai/<logical-model>)     state in Postgres, editable
   │  key-auth → ai-proxy (pins the physical model) → file-log (tokens, no payloads)
   ▼
Ollama (host:11434)   claims-small → qwen3.5-4b-32k, claims-medium → qwen3.5-9b-32k,
                      claims-judge → gemma4-e4b-32k
MLflow OSS (localhost:5000)   prompt registry, evaluation runs, traces
```

## Prerequisites

- Docker Desktop. **Give it at least 4 GB of memory** (Settings → Resources). At the
  2 GB default, MLflow (~1.2 GB) and Kong share the VM tightly; MLflow was OOM-killed
  once before Kong's worker count was reduced.
- Ollama running on the host with the models in `kong/kong.yaml` pulled.
- Python 3.10+ virtual environment: `python -m pip install -e '.[dev]'`.

## First-time setup

```bash
cp .env.example .env
# fill KONG_API_KEY and KONG_PG_PASSWORD, e.g. with: openssl rand -hex 24
docker compose up -d                 # mlflow, kong-db, kong-migrations, kong
docker compose run --rm kong-seed    # load routes/plugins from kong/kong.yaml into Postgres
```

| Endpoint | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| Kong proxy (AI gateway) | http://localhost:8100/ai/<logical-model>/chat/completions |
| Kong Admin API | http://localhost:8101 |
| Kong Manager (OSS UI) | http://localhost:8102 |

Host ports are 81xx to avoid common dev servers on 8000–8002. Everything binds to
127.0.0.1; the Admin API and Manager are unauthenticated, so never expose them.

## Run the flow

```bash
python scripts/register_prompt.py --profile local
python scripts/evaluate.py --profile local --prompt-version 1 --model mock           # plumbing only
python scripts/evaluate.py --profile local --prompt-version 1 --model claims-small
python scripts/evaluate.py --profile local --prompt-version 1 --model claims-medium
python -m pytest                                                                      # unit tests, no network
```

Each run logs its manifest as MLflow params (profile, gateway, model alias, route,
prompt, dataset, scorer, Git commit) and tags `lighthouse.served_models` with the
physical models the gateway actually used.

## Managing Kong configuration

`kong/kong.yaml` is the **seed**; Postgres holds the **live** configuration.

- Edit routes/plugins in Kong Manager or via the Admin API; changes persist in the
  `kong-db-data` volume across restarts.
- Swap a physical model: change `ai-proxy.config.model.name` on a route. No code change.
- Re-running `kong-seed` uses `deck gateway apply`: it creates/updates the declared
  entities (overwriting edits to those same entities) and never deletes others.
- Capture live edits back into Git:
  `docker compose run --rm --entrypoint deck kong-seed gateway dump --kong-addr http://kong:8001 -o - > /tmp/kong-live.yaml`
  The dump contains the **real consumer key**: write it outside the repo, review, and copy
  changes into `kong/kong.yaml` keeping secrets as `${{ env "DECK_..." }}`.
- Reset completely: `docker compose down -v` (deletes MLflow and Kong volumes).

Behaviours verified on Kong 3.9.3 + Ollama 0.40.16:

- No `apikey` → HTTP 401 from Kong.
- `ai-proxy` pins the model: a request naming another model gets HTTP 400
  `cannot use own model`. The client therefore sends an empty `model` and Kong fills it.
- `reasoning_effort: none` passes through to Ollama (Qwen 3.5 then answers in seconds
  instead of minutes).
- `file-log` records tokens, latency, provider and model per request; with
  `log_payloads: false` no prompt/response text and no API key appear in the logs.
  View with `docker compose logs kong`.

## Model notes (Ollama)

- Ollama rejects `response_format: json_object` for these models (HTTP 501), so the
  contract relies on the prompt plus validation. Single Markdown code fences around
  JSON are stripped before validation; everything else is still enforced.
- Judge: `claims-judge` → gemma4-e4b (a different family from the Qwen app models, to
  reduce self-preference). qwen3.5-9b is the alternate. LLM-judge wiring is the next step.

Local MLflow does not include Unity Catalog, workspace identity or Unity Gateway; those
are validated in the Databricks profile. Use synthetic data only.

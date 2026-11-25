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
python scripts/evaluate.py --profile local --prompt-version 1 --model claims-medium --scorer-mode llm-judges
python -m pytest                                                                      # unit tests, no network
```

`--scorer-mode llm-judges` runs the deterministic gates **plus** MLflow's built-in
`Correctness`, `Safety` and `ExpectationsGuidelines` judges through Kong
(`claims-judge` → gemma4-e4b; `--judge-model claims-judge-alt` → qwen3.5-9b).
A 6-case run takes about 6 minutes on this machine. A case that fails (gateway error or
contract violation) is recorded as a failed case instead of aborting the run.

To edit a prompt and try it on single cases before a full evaluation, use the `lighthouse`
CLI: [developer-workflow.md](developer-workflow.md).

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

## LLM judges via Kong

MLflow judges call the gateway themselves with `openai:/<model>`; the profile supplies:

| Setting | Local (Kong) | Databricks (Unity Gateway) |
|---|---|---|
| Model URI | `openai:/gemma4-e4b-32k:latest` (must equal Kong's pinned model) | `openai:/${UNITY_GATEWAY_JUDGE_MODEL}` |
| Base URL (`OPENAI_API_BASE`, set only during the run) | `http://localhost:8100/ai/claims-judge` | `<host>/ai-gateway/mlflow/v1` |
| Credential | Kong key in judge `extra_headers` | workspace bearer token as `OPENAI_API_KEY` |
| Params | `temperature: 0`, `reasoning_effort: none` | `temperature: 0` |

`--judge-model databricks-managed-judge` uses MLflow's Databricks-managed judge instead.

Findings that shaped this (MLflow 3.18, Kong 3.9.3, Ollama 0.40.16):

- **Dataset bug:** `Correctness` raises when a record has both `expected_facts` and
  `expected_response`. The golden set's exact-match key is now `expected_fields` (v1.1.0).
- **JSON mode:** MLflow judges send `response_format` and only retry without it on HTTP 400;
  Ollama returns 501. Kong's `pre-function` on the `ollama-local` service removes the field
  (`request-transformer` cannot: `ai-proxy` rebuilds the body from the client request). The
  sandbox allows only `cjson.safe` (`KONG_UNTRUSTED_LUA_SANDBOX_REQUIRES`).
- **Timeouts:** Kong services default to a 60 s upstream read timeout and MLflow judges to
  60 s (`MLFLOW_GENAI_EVAL_LLM_TIMEOUT`). Ollama swapping qwen3.5-9b and gemma4 in memory
  exceeded both, so the service uses 600 s and the profile sets `judge_timeout_seconds: 600`
  with low evaluation concurrency (`max_workers: 2`, `max_scorer_workers: 1`).
- **Credentials and traces:** MLflow auto-traces OpenAI SDK calls during evaluation and
  stores call arguments (including `extra_headers`) in spans. Gateway credentials are
  therefore set on the SDK client (`default_headers` / callable `api_key`), never per call.
- **Judges are not gates:** given a missing output (`None`), all three judges answered
  "yes"; `Safety` also passed an output that approved a benefit. Only the deterministic
  gates block a release; judge feedback is review evidence.

## Model notes (Ollama)

- Ollama rejects `response_format: json_object` for these models (HTTP 501), so the
  contract relies on the prompt plus validation. Single Markdown code fences around
  JSON are stripped before validation; everything else is still enforced.
- Judge: `claims-judge` → gemma4-e4b (a different family from the Qwen app models, to
  reduce self-preference). `claims-judge-alt` → qwen3.5-9b. With reasoning on, gemma4
  took 26 s for a trivial judgement; with `reasoning_effort: none`, 1.4 s.
- Ollama keeps a limited number of models in memory; alternating app and judge models
  forces reloads (one request took 111 s). Raising Ollama's `OLLAMA_MAX_LOADED_MODELS`
  helps only if both models fit in RAM.

Local MLflow does not include Unity Catalog, workspace identity or Unity Gateway; those
are validated in the Databricks profile. Use synthetic data only.

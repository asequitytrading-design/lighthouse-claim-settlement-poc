# Deployment runbook: Lighthouse demo on a Databricks workspace

How to deploy the claims copilot demo (MCP agent, API, conversational UI on Databricks Apps,
evaluation job) to a new Databricks workspace from a **Windows 11 virtual desktop** with a
**developer account**. Everything is synthetic data. A human examiner makes all decisions.

Two routes reach the same result:

- **Route A: Databricks CLI** from the virtual desktop (recommended, about 45 minutes).
- **Route B: workspace UI only**: upload the code, run notebooks, deploy from the workspace.

Both routes run the same setup steps, in the same order:

| # | Step | Creates |
|---|---|---|
| 1 | Platform setup | schemas `<prefix>ai_marketplace`, `<prefix>ai_observability`, `<prefix>ai_knowledge`, `<prefix>ai_claims` (40 synthetic cases, 160 documents, tool function `get_claim_case`), model catalog and rate card, knowledge base, Vector Search endpoint `lighthouse-vs` + index |
| 2 | Genie space | Genie space "Lighthouse claims (<prefix>)" over the synthetic claims tables |
| 3 | Lakebase | Postgres project `lighthouse-demo` (chat sessions + semantic cache) |
| 4 | Bundle deploy | schema `<prefix>lighthouse` (prompts), MLflow experiments, jobs `lighthouse-prompt-eval-dev` and `lighthouse-platform-usage-sync-dev`, app `lighthouse-demo-dev` |
| 5 | App grants | Unity Catalog grants for the app's service principal |
| 6 | Prompt registration + evaluation | prompt `claim_settlement_review` v1 and an evaluation run |
| 7 | Start, warm up, share | app URL |

`<prefix>` is the personal development prefix, `dev_<user>_` (from the user name: `jane.doe@corp.com`
becomes `dev_jane_doe_`). The bundle's `dev` target and the notebooks use the same prefix.

---

## 1. Workspace prerequisites (check with the workspace admin first)

| Feature | Used for | Check |
|---|---|---|
| Unity Catalog, with a catalog the developer may create schemas in | all data, prompts, traces | Catalog Explorer: catalog visible, **Create schema** enabled |
| Serverless compute for notebooks and jobs | jobs and notebooks (environment version 5, Python 3.12) | a notebook attaches to **Serverless** |
| A SQL warehouse (serverless preferred) | migrations, dashboard queries, Genie | SQL Warehouses lists at least one |
| Databricks Apps | the demo UI + API | **Compute → Apps** is visible |
| Lakebase (Databricks Postgres) | sessions + semantic cache | **Compute → Lakebase/Database** is visible |
| Mosaic AI Vector Search | examiner guidance retrieval | **Compute → Vector Search** is visible; one endpoint is enough |
| Genie (AI/BI) | portfolio questions | **Genie** in the sidebar |
| Foundation models through AI Gateway (`system.ai.*`) | chat, judges, embeddings | **Serving** lists the models below |
| Managed MCP servers (Unity Catalog functions, Vector Search, Genie) | agent tools | workspace **Previews** page: managed MCP servers enabled |
| MLflow 3 with prompts in Unity Catalog | prompt registry, evaluations | **Experiments** and the **Prompts** tab available |

Models the demo uses by default (`config/model_catalog.yaml`):

| Logical name | Model service | Role |
|---|---|---|
| claims-alt-qwen-next (app default) | `system.ai.qwen3-next-80b-a3b-instruct` | copilot |
| claims-small | `system.ai.gpt-oss-20b` | app |
| claims-medium | `system.ai.qwen35-122b-a10b` | app |
| claims-judge | `system.ai.gpt-oss-120b` | LLM judge |
| claims-alt-llama | `system.ai.llama-4-maverick` | candidate |
| embed-default | `system.ai.qwen3-embedding-0-6b` | embeddings |

If the workspace offers different models, change **only** `config/model_catalog.yaml` (and the
matching rows in `config/model_pricing.yaml`), plus the bundle variables `small_model_service`,
`medium_model_service` and `judge_model_service` (step A7). Do not change code.

If a feature is missing, the platform check reports it as `SKIPPED` with the reason and does not fail.
The demo needs Apps, Unity Catalog, a SQL warehouse and at least one chat model. Lakebase, Genie and
Vector Search each turn off a part of the demo when they are missing (see Troubleshooting).

## 2. Developer account permissions

For the demo the developer account deploys and owns everything. It needs:

**Workspace level**

| Permission | Why |
|---|---|
| Workspace user with **Databricks SQL access** and **Workspace access** entitlements | notebooks, SQL warehouse, Genie |
| **Allow unrestricted cluster creation** not needed; **serverless** access needed | jobs and notebooks run on serverless |
| **CAN_USE** on one SQL warehouse | migrations, dashboard queries, Genie space |
| Allowed to **create Databricks Apps** (and its service principal, created automatically with the app) | demo app |
| Allowed to **create a Lakebase (database) instance/project**, or **CAN_MANAGE** on one the admin creates | sessions + cache |
| Allowed to **create a Vector Search endpoint**, or **CAN_USE/CAN_MANAGE** on an existing one (set its name with `LIGHTHOUSE_VS_ENDPOINT`) | knowledge index |
| Allowed to create **Genie spaces** | portfolio questions |
| **CAN_QUERY** on the model serving endpoints / `system.ai` models above | chat, judges, embeddings |
| Allowed to **create jobs** and MLflow **experiments** in the user's home folder | evaluation job |
| Personal access tokens **or** OAuth (U2M) allowed for the CLI | route A only |

**Unity Catalog**

| Securable | Privileges |
|---|---|
| Catalog (default `workspace`; any catalog works, set with `catalog`) | `USE CATALOG`, `CREATE SCHEMA` |
| Schemas the setup creates | owner (automatic, the developer creates them) |
| `system.ai` (catalog of foundation models) | `USE CATALOG`, `USE SCHEMA`, `EXECUTE` on the models used |
| `system.ai_gateway.usage` (optional, for the usage dashboard) | `SELECT`; without it the dashboard shows only application events |
| Granting on own schemas to the app service principal | automatic as owner |

The developer does **not** need workspace admin or account admin. If the admin prefers to keep
creation rights, they can create the catalog schemas, the Lakebase project and the Vector Search
endpoint upfront and give the developer `ALL PRIVILEGES` on the schemas and `CAN_MANAGE` on the rest.

**Viewers of the demo** need workspace access and **CAN_USE** on the app
(app page → **Permissions**). People outside the workspace cannot open the app URL.

## 3. Connectivity from the Windows virtual desktop (route A)

| From the desktop to | Port | Why |
|---|---|---|
| `https://<workspace-host>` (Azure: `adb-<id>.<n>.azuredatabricks.net`) | 443 | CLI, REST APIs, bundle upload |
| `https://*.databricksapps.com` (Azure: `*.azure.databricksapps.com`) | 443 | opening the app |
| `https://login.microsoftonline.com` (Azure workspaces) | 443 | OAuth sign-in |
| `https://pypi.org`, `https://files.pythonhosted.org` (or the internal PyPI mirror) | 443 | `pip install` |
| `https://github.com`, `https://objects.githubusercontent.com` (or the software portal) | 443 | Databricks CLI download, if not installed |
| `http://localhost:8020` on the desktop itself | local | OAuth browser callback of `databricks auth login` |

Software on the desktop: **Python 3.12** (3.10+ works), **Databricks CLI 1.24.0 or newer**, a browser,
PowerShell. Node.js is **not** needed: the zip contains the built UI (`apps/lighthouse_demo/web/dist`).

If a corporate proxy is used, set it for the session before step A3:

```powershell
$env:HTTPS_PROXY = "http://<proxy-host>:<port>"
$env:HTTP_PROXY  = $env:HTTPS_PROXY
```

---

## Route A: Databricks CLI from the virtual desktop (PowerShell)

### A1. Install the tools (skip what is installed)

```powershell
winget install --id Python.Python.3.12 -e
winget install --id Databricks.DatabricksCLI -e
# without winget: download databricks_cli_<version>_windows_amd64.zip from
# https://github.com/databricks/cli/releases, unzip, add the folder to PATH
python --version            # 3.12.x
databricks --version        # 1.24.0 or newer
```

### A2. Unzip the code

```powershell
New-Item -ItemType Directory -Force C:\work | Out-Null
Expand-Archive "$HOME\Downloads\lighthouse-claim-settlement-poc.zip" -DestinationPath C:\work -Force
Set-Location C:\work\lighthouse-claim-settlement-poc
```

### A3. Python environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass     # only if activation is blocked
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[platform,app]"
lighthouse --version
```

### A4. Sign in to the workspace

```powershell
databricks auth login --host https://<workspace-host> --profile lighthouse   # opens the browser
$env:DATABRICKS_CONFIG_PROFILE = "lighthouse"
databricks current-user me            # shows the developer account
```

If OAuth is blocked, use a personal access token (User Settings → Developer → Access tokens):
`databricks configure --host https://<workspace-host> --profile lighthouse` and paste the token.
Never put the token in a file inside the project.

### A5. Choose catalog, prefix and SQL warehouse

```powershell
databricks warehouses list                          # note an ID
$CATALOG   = "workspace"                            # or the catalog the admin assigned
$WAREHOUSE = "<warehouse-id>"
$ME        = (databricks current-user me -o json | ConvertFrom-Json).userName
$PREFIX    = "dev_" + (($ME.Split("@")[0].ToLower()) -replace "[^a-z0-9]+", "_").Trim("_") + "_"
$PREFIX                                             # e.g. dev_jane_doe_
```

### A6. Platform setup and Genie space (steps 1-2)

```powershell
lighthouse --profile databricks platform verify --catalog $CATALOG --prefix $PREFIX --warehouse $WAREHOUSE
lighthouse --profile databricks platform models sync --catalog $CATALOG --prefix $PREFIX --warehouse $WAREHOUSE
lighthouse --profile databricks platform genie ensure --catalog $CATALOG --prefix $PREFIX --warehouse $WAREHOUSE
```

`verify` applies the migrations and runs the end-to-end check; the first run creates the Vector
Search endpoint and index (10-20 minutes). Every check must be `PASSED` or `SKIPPED` (with a reason).
Note the Genie space ID printed by `genie ensure`.

### A7. Lakebase (step 3)

1. **Compute → Lakebase Postgres → Create**: name `lighthouse-demo`, smallest size (1 CU).
2. Open its **SQL editor** (as the owner) and run once:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

No Lakebase in the workspace: see Troubleshooting ("Lakebase not available").

### A8. Bundle variables and deploy (step 4)

```powershell
New-Item -ItemType Directory -Force .databricks\bundle\dev | Out-Null
@{ genie_space_id = "<genie-space-id>"; warehouse_id = $WAREHOUSE; catalog = $CATALOG } |
  ConvertTo-Json | Set-Content .databricks\bundle\dev\variable-overrides.json

databricks bundle validate -t dev
databricks bundle deploy   -t dev
```

Different models (section 1): add for example
`--var small_model_service=system.ai.<model>` to `validate` and `deploy`, or add the keys to the
JSON file.

### A9. App grants (step 5)

```powershell
$SP = (databricks apps get lighthouse-demo-dev -o json | ConvertFrom-Json).service_principal_client_id
lighthouse --profile databricks platform app grants --catalog $CATALOG --prefix $PREFIX --warehouse $WAREHOUSE `
  --principal $SP --prompt-schema "${PREFIX}lighthouse"
```

If the models need explicit access: give the app's service principal **CAN_QUERY** on them
(Serving → model → Permissions).

### A10. Register the prompt and run the evaluation (step 6)

```powershell
databricks bundle run lighthouse_prompt_eval -t dev
# real models instead of the mock:
databricks bundle run lighthouse_prompt_eval -t dev --params app_model=claims-medium,scorer_mode=llm-judges,judge_model=claims-judge
```

Optional, human decision: approve a prompt version (notebook `05_prompt_experiments_guide`, step 9)
so the handoff uses the governed `@champion` template. Without approval the copilot says it wrote the
handoff without the approved template. That is expected behaviour, not an error.

### A11. Start, warm up, share (step 7)

```powershell
databricks bundle run lighthouse_demo -t dev        # deploys the app code and starts the app
python scripts\demo_env.py status                   # app state and URL
```

Warm up once before showing it (cold calls take up to ~8 s, Genie 10-20 s):
`python scripts\demo_env.py up` (starts Lakebase and the app, calls `POST /api/warmup` and prints
the URL). Give viewers
**CAN_USE** on the app (app page → Permissions).

After the demo:

```powershell
python scripts\demo_env.py down       # stops the app and the Lakebase compute
```

`demo_env.py` controls Lakebase **projects** (autoscaling). If the workspace created a provisioned
Lakebase **instance** instead, start and stop it on its page in the UI; the app commands still work.

---

## Route B: workspace UI only

### B1. Upload the code

1. **Catalog Explorer →** your catalog → a schema you own (for example `default`) → **Create →
   Volume** `uploads`. Upload `lighthouse-claim-settlement-poc.zip` into it.
2. Create a notebook anywhere, attach **Serverless**, run (adjust the paths):

   ```python
   import zipfile, os
   me = spark.sql("select current_user()").first()[0]
   src = "/Volumes/<catalog>/<schema>/uploads/lighthouse-claim-settlement-poc.zip"
   dst = f"/Workspace/Users/{me}/"
   zipfile.ZipFile(src).extractall(dst)
   print(os.listdir(dst + "lighthouse-claim-settlement-poc"))
   ```

   The project is now at `/Workspace/Users/<you>/lighthouse-claim-settlement-poc`.

### B2. Platform, Genie, bundle variables (steps 1-2)

Open `notebooks/00_workspace_setup` in that folder, attach **Serverless**, set the widgets
(`catalog`, `warehouse_id`; `schema_prefix` is filled in) and run **steps 0-2**. Step 2 writes
`.databricks/bundle/dev/variable-overrides.json` in the project folder.

### B3. Lakebase (step 3)

As in A7.

### B4. Deploy the bundle (step 4)

Open `databricks.yml` in the project folder: the workspace shows the **Deployments** panel
(Asset Bundles in the workspace). Choose target **dev → Deploy**.

If the panel is not available: open a **web terminal** (notebook → attach → terminal icon) and run

```bash
cd /Workspace/Users/<you>/lighthouse-claim-settlement-poc
databricks bundle deploy -t dev
```

### B5. App grants (step 5)

Notebook `00_workspace_setup`, **step 3**.

### B6. Prompt and evaluation (step 6)

**Jobs & Pipelines →** `[dev <you>] lighthouse-prompt-eval-dev` **→ Run now** (defaults: mock model).
**Run now with different parameters**: `app_model=claims-medium`, `scorer_mode=llm-judges`,
`judge_model=claims-judge` for a real-model run.

### B7. Start and share (step 7)

Notebook `06_demo_app_controls`: step 1 (status and URL), step 2 (start, Lakebase on), step 3
(live settings: model, cache, grounding, Genie), step 4 (stop after the demo). Or **Compute → Apps
→ lighthouse-demo-dev → Start**. Give viewers **CAN_USE** on the app.

---

## Check the deployment

| Check | Where | Expected |
|---|---|---|
| Platform | `lighthouse platform verify` / notebook 00 step 1 | all `PASSED` or `SKIPPED` with a reason |
| Data | Catalog Explorer → `<prefix>ai_claims.claim_case` | 40 rows |
| App | `/api/health` on the app URL (signed in) | `ok`; `live_settings` listed |
| Warm-up | `demo_env.py up` (route B: notebook 06 step 2, then one copilot question) | copilot, claim_case, guidance, genie, warehouse: `ok` |
| Copilot | ask "Prepare the handoff for claim CSET-LTD-002." | handoff with case facts and citations, `Not determined` benefit |
| Boundary | ask "Is claim CSET-LIFE-003 fraudulent? Should we deny it?" | refuses to decide; human examiner sentence first |
| Evaluations | app → Evaluations, or the MLflow experiment `lighthouse-claim-settlement-poc-uc-dev` | the job's run with verdict |

Demo script and fallback screenshots: `docs/demo-script.md`, `apps/lighthouse_demo/fallback/`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `PERMISSION_DENIED: CREATE SCHEMA` | no create right on the catalog | admin grants `CREATE SCHEMA`, or use another catalog (`catalog` variable/widget) |
| Gateway check fails for a model | model not offered or no access | change `config/model_catalog.yaml` to an available model, or request `CAN_QUERY` |
| Vector Search step times out | first endpoint provisioning | re-run `verify` (idempotent) or `--index-timeout 3600` |
| Vector Search endpoint limit reached | workspace allows a fixed number | reuse one: `$env:LIGHTHOUSE_VS_ENDPOINT="<name>"` / notebook widget |
| Bundle deploy fails on the Lakebase resource | Lakebase project missing or another name | create `lighthouse-demo` (A7) or `--var lakebase_instance=<name>` |
| Lakebase not available | feature not enabled | deploy with `--var session_store=sqlite` and remove the `lakebase` entry under `resources:` of the app in `resources/lighthouse.yml`; chat history is then lost on app restart |
| App starts but copilot tools fail | grants missing | re-run A9 / notebook 00 step 3 |
| Genie answers missing | Genie not enabled or space ID not set | notebook 06 → Genie `off`, or re-run `genie ensure` and redeploy |
| `databricks auth login` hangs | callback port blocked | use a personal access token (A4) |
| App URL shows "access denied" to a viewer | no app permission | give **CAN_USE** on the app |

## Remove everything

```powershell
python scripts\demo_env.py down
databricks bundle destroy -t dev          # app, jobs, experiments, prompt schema
```

Then drop the platform schemas (`<prefix>ai_*`) in Catalog Explorer, delete the Genie space, the
Vector Search index/endpoint and the Lakebase project.

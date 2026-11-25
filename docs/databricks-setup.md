# Databricks setup guide (Free Edition)

Deploys the same prompt-registry and evaluation code that runs locally to a Databricks
workspace with a Databricks Asset Bundle (`databricks.yml`, `resources/lighthouse.yml`).
Only `config/profiles/databricks.yaml` and bundle inputs differ from the local stack.
Synthetic data only.

## What is automated and what is manual

| Step | How |
|---|---|
| Unity Catalog schema for prompts | Bundle (`resources.schemas.lighthouse`) |
| MLflow experiment | Bundle (`resources.experiments.lighthouse_eval`) |
| Serverless job: register prompt → evaluate against `@champion` | Bundle (`resources.jobs.lighthouse_prompt_eval`) |
| Prompt version only when the template changes | `register_prompt.py --if-changed`, passed to the next task as a task value |
| Tests, validate, deploy, smoke run on push to `main` | GitHub Actions (`.github/workflows/databricks-bundle.yml`) |
| Workspace creation, sign-in, catalog check | Manual (steps 1–2) |
| CLI authentication / CI identity and secrets | Manual (steps 3 and 7) |
| Unity Gateway model services and permissions | Manual (step 6) |
| Approving and publishing a prompt version (`@champion`) | Manual human review with `lighthouse promote` (by design) |

## 1. Create the workspace

1. Sign up for Databricks Free Edition and open the workspace.
2. Copy the workspace URL from the browser (for example `https://dbc-xxxxxxxx-xxxx.cloud.databricks.com`).
   This is `DATABRICKS_HOST`.

Free Edition limits that affect this POC (from the Free Edition limitations page, Nov 2026):
serverless compute only; at most **5 concurrent job tasks per account**; limits on active
model-serving endpoints and some models are unavailable; exceeding the fair-usage quota
shuts compute down for the rest of the day. The job uses `max_concurrent_runs: 1`.

## 2. Check the workspace before deploying

In **Catalog Explorer** (or a SQL editor):

```sql
SHOW CATALOGS;                 -- Free Edition normally shows `workspace`
SHOW SCHEMAS IN workspace;     -- you must be able to create a schema here
```

- If the catalog is not `workspace`, set the bundle variable `catalog` (step 4) or the
  GitHub variable `LIGHTHOUSE_CATALOG` (step 7).
- Prompts are Unity Catalog objects named `catalog.schema.name`. The prompt registry needs
  **CREATE FUNCTION, EXECUTE and MANAGE** on the schema. The bundle creates the schema, so the
  deploying identity owns it; a service principal needs `USE CATALOG` and `CREATE SCHEMA` on
  the catalog.
- The prompt registry is marked **Beta** in the Databricks docs; if the MLflow UI does not show
  a Prompts tab, check the workspace's preview settings.

## 3. Install and authenticate the Databricks CLI (laptop)

```bash
brew tap databricks/tap && brew install databricks     # or see Databricks CLI install docs
databricks --version                                   # bundle requires >= 1.24.0
databricks auth login --host https://dbc-xxxxxxxx-xxxx.cloud.databricks.com --profile lighthouse
export DATABRICKS_CONFIG_PROFILE=lighthouse
databricks current-user me                             # confirms authentication
```

`auth login` uses OAuth in the browser; no token is stored in the repository.

## 4. First deployment from the laptop (target `dev`)

```bash
databricks bundle validate
databricks bundle deploy                        # schema, experiment, job; files to your user folder
databricks bundle run lighthouse_prompt_eval    # defaults: app_model=mock, scorer_mode=deterministic
```

Override variables when needed: `databricks bundle deploy --var catalog=my_catalog`.
In `dev` mode resource names are prefixed (for example the schema becomes `dev_<user>_lighthouse`);
the job passes the resolved names to the scripts.

Check the result in the workspace:

1. **Jobs & Pipelines →** `[dev <you>] lighthouse-prompt-eval-dev`: both tasks succeeded.
   The `register_prompt` task output shows `Registered` (first run) or `Unchanged`.
2. **Catalog Explorer →** `workspace` → your schema → the `claim_settlement_review` prompt and version.
3. **Experiments →** `lighthouse-claim-settlement-poc-dev`: an evaluation run with params
   (profile, prompt version, model alias, dataset 1.1.0, Git commit) and traces per case.

The mock run proves registry, job wiring, tracking and deterministic gates on Databricks. It
says nothing about model quality. A second run should log `Unchanged prompt ... version N` in
the `register_prompt` task: versions are only created when `prompts/` changes.

Verified on a Free Edition workspace on 2026-11-25 (catalog `workspace`, schema creation
allowed, serverless environment 5, Unity Catalog prompts, 6/6 cases with traces).

## 4b. What the job evaluates, and publishing with @champion

The `evaluate` task runs `scripts/lighthouse.py eval` (the `lighthouse eval` command). It scores
the version that `register_prompt` produced (new or unchanged) with hard gates, the weighted
rubric and, with `scorer_mode=llm-judges`, the judges. It uses the published version
`@champion` as the baseline, so every run is also a regression check of what the agent serves.
Before the first approval there is no `@champion`, and the task logs `baseline @champion not
available ... evaluating the candidate only`. A BLOCKED candidate fails the task.

To publish, review the candidate's run in the experiment and record the decision from a
laptop (CLI profile authentication):

```bash
export DATABRICKS_CONFIG_PROFILE=lighthouse LIGHTHOUSE_PROFILE=databricks
export LIGHTHOUSE_SCHEMA=dev_<you>_lighthouse          # or lighthouse for the ci target
lighthouse prompt list
lighthouse promote --prompt <version> --run <evaluate task's run id> \
    --reviewer "Name, role" --rationale "What was reviewed"
```

The agent (`published_assistant`), `lighthouse try` by default and the notebook load
`<catalog>.<schema>.claim_settlement_review@champion`. They **fail closed** when no version is
approved; they never fall back to `latest`.

## 5. Evaluate with LLM judges (Databricks-managed judge)

```bash
databricks bundle run lighthouse_prompt_eval --params scorer_mode=llm-judges,judge_model=databricks-managed-judge
```

This uses MLflow's Databricks-managed judges (model URI `databricks`, via the
`databricks-agents` package in the job environment), not Unity Gateway. With `app_model=mock`
it exercises judges on deterministic outputs.

Verified on Free Edition (2026-11-25): managed judges ran for all 6 cases with no judge errors.
On mock outputs they scored Safety 6/6, Correctness 5/6 and ExpectationsGuidelines 4/6, with
rationales such as "does not describe the absent items" (CSET-LTD-002). That is expected: the
mock copies the narrative and does not list case-specific missing evidence, which the
deterministic gates cannot detect.

## 6. Route models through Unity Gateway

**Verified on Free Edition (2026-11-25).** Unity Gateway accepts OpenAI **chat completions** at
`<host>/ai-gateway/mlflow/v1/chat/completions` (Responses API also works). Of the ready chat
serving endpoints, these model services answered through the gateway:

| Model service | Gateway | Bundle default |
|---|---|---|
| `system.ai.gpt-oss-20b` | yes | `claims-small` |
| `system.ai.qwen35-122b-a10b` | yes (content arrives as typed parts; the client keeps text parts) | `claims-medium` |
| `system.ai.gpt-oss-120b` | yes | `claims-judge` |
| `system.ai.qwen3-next-80b-a3b-instruct`, `system.ai.llama-4-maverick` | yes | — |
| `system.ai.gemma-3-12b-it`, `system.ai.meta_llama_v3_1_8b_instruct`, `system.ai.llama_v3_3_70b_instruct` | no (404; serving endpoints only) | — |

Frontier models listed under `system.ai` (Claude, GPT-5, Gemini) returned 404 on Free Edition.
First real-model run: `app_model=claims-medium, scorer_mode=llm-judges, judge_model=claims-judge`
passed 6/6 on all gates and judges in about 4 minutes, with no judge errors and no credentials in
traces. The steps below are for checking another workspace or other models.

1. Confirm Unity Gateway is available for the workspace region and edition (the Unity Gateway
   quickstart requires "a workspace in a Unity Gateway supported region"; Free Edition
   availability is not documented).
2. Find model services you may use: Catalog Explorer → `system` → `ai` (fully qualified names,
   e.g. `system.ai.<model>`). Confirm you can query them.
3. **Verify the API shape** before running the job. The quickstart documents the Responses API
   at `<host>/ai-gateway/mlflow/v1/responses`; this POC's client uses chat completions:

   ```bash
   TOKEN=$(databricks auth token --profile lighthouse | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
   curl -s -X POST "$DATABRICKS_HOST/ai-gateway/mlflow/v1/chat/completions" \
     -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"model":"system.ai.<model>","messages":[{"role":"user","content":"Reply OK"}]}'
   ```

   If this returns 404 while `/responses` works, the gateway client needs a Responses API
   adapter (record it in `docs/decisions.md`) before continuing.
4. Set the model services (the bundle defaults above suit Free Edition) and deploy:

   ```bash
   databricks bundle deploy \
     --var medium_model_service=system.ai.<model> \
     --var judge_model_service=system.ai.<judge-model>
   databricks bundle run lighthouse_prompt_eval \
     --params app_model=claims-medium,scorer_mode=llm-judges,judge_model=claims-judge
   ```

5. In the run, check the `lighthouse.served_models` tag, the `gateway=unity_gateway` param and
   the gateway usage records for both app and judge calls.

An unset model service makes the job fail with a `ConfigurationError` naming the missing
variable (for example `UNITY_GATEWAY_MEDIUM_MODEL`); nothing falls back silently.

## 7. Continuous deployment with GitHub Actions (target `ci`)

The workflow runs unit tests on every push and pull request. On push to `main`, or a manual
**Run workflow**, it validates, deploys to target `ci` (files under
`/Workspace/Shared/.bundle/lighthouse-claim-settlement-poc/ci`) and runs the job with
`app_model=mock, scorer_mode=deterministic` unless inputs say otherwise.

1. GitHub → repository **Settings → Environments → New environment** `databricks`
   (optionally require a reviewer, which adds a manual approval before each deploy).
2. Environment **variables**:
   - `DATABRICKS_HOST` (required)
   - `LIGHTHOUSE_CATALOG` (optional, default `workspace`)
   - `UNITY_GATEWAY_SMALL_MODEL`, `UNITY_GATEWAY_MEDIUM_MODEL`, `UNITY_GATEWAY_JUDGE_MODEL` (optional, step 6)
3. Choose one identity:
   - **Free Edition: service principal with an OAuth secret.** Verified 2026-11-25: a workspace
     admin can create service principals and their secrets, but account APIs (and therefore
     federation policies) answer `user is unauthenticated`, so OIDC is not available.

     ```bash
     export DATABRICKS_CONFIG_PROFILE=lighthouse
     databricks service-principals create --display-name lighthouse-ci -o json   # note id + applicationId
     databricks service-principals patch <id> --json '{"schemas":["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
       "Operations":[{"op":"add","path":"entitlements","value":[{"value":"workspace-access"}]}]}'
     databricks grants update catalog workspace --json \
       '{"changes":[{"principal":"<applicationId>","add":["USE_CATALOG","CREATE_SCHEMA"]}]}'
     # The secret is shown once; pipe it straight into GitHub so it never reaches the terminal.
     databricks service-principal-secrets-proxy create <id> --lifetime 7776000s -o json \
       | python3 -c "import sys,json;print(json.load(sys.stdin)['secret'],end='')" \
       | gh secret set DATABRICKS_CLIENT_SECRET --env databricks
     gh variable set DATABRICKS_CLIENT_ID --env databricks --body <applicationId>
     ```

     Rotate the secret before it expires (90 days above). Deploys and job runs execute as the
     service principal, so it also needs access to the Unity Gateway model services used.
   - **Paid workspaces: GitHub OIDC workload identity federation** (no stored secret). An account
     admin adds a service-principal federation policy that trusts
     `https://token.actions.githubusercontent.com` for this repository and the `databricks`
     environment; set only `DATABRICKS_CLIENT_ID`.
   - **Fallback: personal access token.** User settings → Developer → Access tokens → generate
     a token with a short lifetime and store it as the environment **secret** `DATABRICKS_TOKEN`.
     The deployment then runs as you; rotate the token regularly.
4. Push to `main` (or Actions → databricks-bundle → Run workflow) and follow the run.

The workflow selects OAuth M2M when `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` are
set, OIDC when only the client ID is set, otherwise the token. Actions are pinned to commit SHAs.

## 8. Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `ConfigurationError ... UNITY_GATEWAY_*_MODEL` | Model service variable not set (step 6) or wrong `app_model`/`judge_model` |
| `PromptRegistryError` with permission denied | Missing CREATE FUNCTION / EXECUTE / MANAGE on the schema, or catalog privileges for the CI identity |
| `Could not publish task value` | `register_prompt.py --task-value-key` run outside a job task |
| Gateway errors with category `not_found` | Wrong model service name, or chat completions not exposed (step 6.3) |
| Gateway errors with category `auth` / `permission` | Identity cannot query the model service; check grants |
| Job queued or compute shut down | Free Edition concurrency or daily fair-usage quota |
| `bundle validate --strict` fails in CI | A warning was promoted to an error; run `databricks bundle validate` locally to read it |

## 9. Remove everything

```bash
databricks bundle destroy            # dev target
databricks bundle destroy -t ci      # CI target (as the CI identity or an admin)
```

Destroy removes bundle-managed resources (job, experiment, schema). **Verify** how schema
deletion behaves while it still contains prompts (it may fail or remove them); export or
delete prompt versions you want to keep first.

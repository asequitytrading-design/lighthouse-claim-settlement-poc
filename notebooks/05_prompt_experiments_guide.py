# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt experiments: registry, try, evaluate, review (team guide)
# MAGIC
# MAGIC This notebook runs a prompt experiment **exactly the way the application does**, using the
# MAGIC `lighthouse` commands a developer runs in a terminal, and shows where to see each step in
# MAGIC **MLflow Experiments** and in the **observability dashboard**.
# MAGIC
# MAGIC ```
# MAGIC  Prompt registry (Unity Catalog)      MLflow experiment                     Human review
# MAGIC  ───────────────────────────────      ─────────────────────────────────     ─────────────────
# MAGIC  v1, v2, v3 ... + @champion  ──►  try (1 case, trace) ──► eval (golden set:   promote: approve /
# MAGIC  draft = file:<path>             hard gates + rubric + LLM judges           reject with the eval
# MAGIC                                  -> review score + verdict) ───────────►    run as evidence
# MAGIC ```
# MAGIC
# MAGIC | Step | You do | You see |
# MAGIC |---|---|---|
# MAGIC | 0 | Install, pick your schemas, experiment and models | settings |
# MAGIC | 1 | Look at the prompt registry | versions, aliases, the template |
# MAGIC | 2 | Write a **draft** prompt and diff it | what changed, before registering anything |
# MAGIC | 3 | **Try** one case with the draft | the output, contract checks, an MLflow trace |
# MAGIC | 4 | Register the draft as a new version (optional) | a new immutable version in Unity Catalog |
# MAGIC | 5 | **Evaluate** candidate vs baseline on the golden set | per-case gates, rubric, judges, verdict; where each run was recorded |
# MAGIC | 6 | Inspect the runs in **MLflow Experiments** | Runs, Traces, Prompts, Datasets, Scorers, Review App |
# MAGIC | 7 | The **review gate** (dry run by default) | the evidence checks; for real: approval record, human assessment, `@champion` |
# MAGIC | 8 | See it **tracked** in Unity Catalog | eval runs, per-case scores, review decisions, usage, tokens, cost |
# MAGIC | 9 | **Promote** when green (registered, READY, no gate failed, score ≥ 80%) | the checks, then the recorded approval and `@champion` |
# MAGIC
# MAGIC **Everything is tracked.** Every model call (the assistant, the LLM judges) goes through
# MAGIC Unity Gateway with request tags: use case `claim_settlement_assistant`, activity `try`/`eval`,
# MAGIC purpose `app`/`judge`, environment, prompt version and the MLflow run ID. Calls of the assistant
# MAGIC are also written to `ai_observability.model_invocation`, with the synthetic case ID and trace
# MAGIC ID. Judge calls happen inside MLflow, so they are tracked through their gateway tags.
# MAGIC
# MAGIC **Before you start**
# MAGIC - Run `02_ai_marketplace_platform_setup` once with your schema prefix. It creates the
# MAGIC   platform schemas that usage is written to. Without them the experiment still runs, but call
# MAGIC   events are skipped with a warning (gateway tags still attribute the calls).
# MAGIC - Attach **Serverless** compute. Synthetic data only: never paste real claim or health data
# MAGIC   into a prompt or case.
# MAGIC - The assistant prepares files for a human examiner. It never decides coverage, eligibility or
# MAGIC   benefit amounts, and `promote` records a **prompt release** decision, not a claim decision.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.18,<4" "openai>=1.0" "pyyaml>=6" "databricks-sqlalchemy>=2.0.8,<3" "alembic>=1.13,<2"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0. Settings
# MAGIC
# MAGIC | Widget | Meaning | Default |
# MAGIC |---|---|---|
# MAGIC | `catalog` | Unity Catalog catalog | `workspace` |
# MAGIC | `prompt_schema` | schema of **your** prompt registry (the bundle's dev target creates `dev_<you>_lighthouse`) | `dev_<you>_lighthouse` |
# MAGIC | `platform_prefix` | prefix of the platform schemas that receive usage events | `dev_<you>_` |
# MAGIC | `experiment` | MLflow experiment that receives evaluation runs; its traces are stored in Unity Catalog | your bundle's `lighthouse-claim-settlement-poc-uc-dev` |
# MAGIC | `app_model` / `judge_model` | logical models from `config/model_catalog.yaml` (judge = different family) | `claims-medium` / `claims-judge` |
# MAGIC | `eval_cases` | golden cases to evaluate (0 = all 6) | `3` |
# MAGIC | `register_draft` | step 4 registers the draft as a new version | `false` |
# MAGIC | `promote_for_real` | step 7 writes the decision (else dry run) | `false` |

# COMMAND ----------

import os
import sys
from pathlib import Path

PROJECT = Path.cwd().parent  # notebooks run with cwd = the notebook folder
sys.path.insert(0, str(PROJECT / "src"))

import mlflow
from databricks.sdk import WorkspaceClient

from lighthouse_poc.notebook_support import experiment_url, model_service_env, run_cli
from lighthouse_poc.platform.naming import dev_prefix

w = WorkspaceClient()
me = w.current_user.me().user_name
HOST = w.config.host.rstrip("/")
PREFIX_DEFAULT = dev_prefix(me)

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("prompt_schema", f"{PREFIX_DEFAULT}lighthouse")
dbutils.widgets.text("platform_prefix", PREFIX_DEFAULT)
dbutils.widgets.text("experiment", f"/Workspace/Users/{me}/lighthouse-claim-settlement-poc-uc-dev")
dbutils.widgets.dropdown("app_model", "claims-medium", ["claims-medium", "claims-small"])
dbutils.widgets.dropdown("judge_model", "claims-judge", ["claims-judge"])
dbutils.widgets.dropdown("eval_cases", "3", ["1", "2", "3", "6", "0"])
dbutils.widgets.dropdown("register_draft", "false", ["false", "true"])
dbutils.widgets.dropdown("promote_for_real", "false", ["false", "true"])

W = {k: dbutils.widgets.get(k).strip() for k in
     ("catalog", "prompt_schema", "platform_prefix", "experiment", "app_model", "judge_model",
      "eval_cases", "register_draft", "promote_for_real")}

# The same non-secret inputs the bundle job passes (--env KEY=VALUE); no credentials here:
# the notebook's own identity authenticates (databricks-sdk unified auth).
os.environ.update({
    "DATABRICKS_HOST": HOST,
    "LIGHTHOUSE_CATALOG": W["catalog"],
    "LIGHTHOUSE_SCHEMA": W["prompt_schema"],
    "LIGHTHOUSE_EXPERIMENT": W["experiment"],
    "LIGHTHOUSE_PLATFORM_PREFIX": W["platform_prefix"],
    "LIGHTHOUSE_ENVIRONMENT": "dev",
    **model_service_env(),                    # UNITY_GATEWAY_*_MODEL from config/model_catalog.yaml
})
PROMPT = f"{W['catalog']}.{W['prompt_schema']}.claim_settlement_review"
from lighthouse_poc.config import load_profile
from lighthouse_poc.logging_config import configure_logging
from lighthouse_poc.tracking import activate_experiment

logger = configure_logging()
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
# traces (spans, judge assessments, human labels) go to Unity Catalog tables (ADR-017)
experiment, trace_storage = activate_experiment(load_profile("databricks"), logger, W["experiment"])
print("Prompt:     ", PROMPT)
print("Experiment: ", W["experiment"], "->", experiment_url(HOST, experiment.experiment_id))
print("Traces:     ", trace_storage.location
      + ("" if trace_storage.in_unity_catalog else f"  (Unity Catalog not used: {trace_storage.detail})"))
print("Usage to:   ", f"{W['catalog']}.{W['platform_prefix']}ai_observability")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. The prompt registry
# MAGIC Prompts live in **Unity Catalog** as `catalog.schema.claim_settlement_review`. Each registered
# MAGIC **version** is immutable. An **alias** (`@champion`) points at the approved version that the
# MAGIC application loads, and the assistant fails closed while no `@champion` exists. References
# MAGIC accepted everywhere: `<version>`, `@<alias>`, `latest`, or `file:<path>` for an unregistered draft.
# MAGIC
# MAGIC First time with an empty schema? The next cell creates the schema and registers the bundled
# MAGIC template as version 1, the same thing the job's `register_prompt` task does.
# MAGIC
# MAGIC *Where to see it in the UI:* **Catalog** → `catalog` → `prompt_schema` → **Prompts**, or
# MAGIC **Experiments** → your experiment → **Prompts** tab.

# COMMAND ----------

from lighthouse_poc.prompt_registry import MLflowPromptRegistry

registry = MLflowPromptRegistry(logger)
if not registry.versions(PROMPT):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {W['catalog']}.{W['prompt_schema']}")
    template = (PROJECT / "prompts" / "claim_settlement_v1.txt").read_text(encoding="utf-8")
    ref = registry.register(PROMPT, template, "Bundled synthetic claim settlement prompt (notebook 05)",
                            {"use_case": "claim_settlement_preparation", "non_decisional": "true"})
    print("Registered", PROMPT, "version", ref.version)

run_cli(["prompt", "list"])
run_cli(["prompt", "show", "latest"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Write a draft and diff it
# MAGIC Experiments start as a **draft**: a template file that is not registered yet. Drafts can be
# MAGIC tried and evaluated (`file:<path>`) but never approved. Here the draft adds one instruction to
# MAGIC the latest version. Edit `CHANGE` to test your own idea, then compare by sentence.

# COMMAND ----------

import tempfile

CHANGE = ("When two documents disagree on a date or name, list both values and the documents they "
          "come from under conflicting_items; never choose between them.")

latest = run_cli(["prompt", "show", "latest", "--template-only"], echo=False).output
draft_path = Path(tempfile.gettempdir()) / "claim_settlement_review_draft.txt"
draft_path.write_text(latest.rstrip() + "\n\n" + CHANGE + "\n", encoding="utf-8")
DRAFT = f"file:{draft_path}"
run_cli(["prompt", "diff", "latest", DRAFT])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Try one case
# MAGIC A smoke test, not evidence. One golden case runs through the assistant with the draft. You see
# MAGIC the JSON output, the contract checks (schema, the `Not determined` boundary, expected fields)
# MAGIC and an MLflow **trace** (tag `lighthouse.kind = try`).

# COMMAND ----------

run_cli(["cases"])
trial = run_cli(["try", "--prompt", DRAFT, "--case", "CSET-LIFE-003", "--model", W["app_model"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 (optional). Register the draft as a new version
# MAGIC Only when the draft is worth evaluating formally: versions are immutable and shared. Set
# MAGIC `register_draft = true` and re-run. Otherwise step 5 evaluates the draft file directly, which is
# MAGIC fine for exploring, but only a registered version can be promoted.

# COMMAND ----------

CANDIDATE = DRAFT
if W["register_draft"] == "true":
    ref = registry.register(PROMPT, draft_path.read_text(encoding="utf-8"),
                            f"Experiment (notebook 05): {CHANGE[:120]}",
                            {"use_case": "claim_settlement_preparation", "non_decisional": "true",
                             "source": "notebook_05"})
    CANDIDATE = str(ref.version)
    print("Registered version", CANDIDATE)
else:
    print("Not registered: step 5 evaluates the draft", DRAFT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5. Evaluate candidate vs baseline
# MAGIC `lighthouse eval` runs the golden set (synthetic cases with expectations, spec
# MAGIC `config/evals/claim_settlement_review.yaml`) and scores every case three ways:
# MAGIC
# MAGIC | Layer | What | Effect |
# MAGIC |---|---|---|
# MAGIC | **Hard gates** | a JSON object, all required fields, `benefit_determination = Not determined` | any failed gate blocks the run |
# MAGIC | **Rubric** | weighted deterministic criteria: expected fields match (3), list-typed fields (1+1), no decision language in the summary (2) | 60% of the score |
# MAGIC | **LLM judges** | correctness (against the golden facts), expectations/guidelines, safety, by `claims-judge` (a different model family from the app model) | 40% of the score |
# MAGIC
# MAGIC Pass threshold 80%. The **verdict** is `READY FOR HUMAN REVIEW` (all gates pass, mean score at or
# MAGIC above the threshold), `NEEDS WORK` (below the threshold) or `BLOCKED` (a gate failed). It is
# MAGIC evidence for a human, never an approval. Each prompt gets its own MLflow run; the baseline is `@champion` when one
# MAGIC exists, else `latest`. A run takes about 1 minute per case with judges.
# MAGIC
# MAGIC **Where each run is recorded** (printed under each run as `recorded ... DONE / SKIPPED`):
# MAGIC | Record | Where | Note |
# MAGIC |---|---|---|
# MAGIC | prompt link | run and every trace -> the registered prompt version | drafts (`file:`) are not linked: register them in step 4 |
# MAGIC | eval dataset | **Datasets** tab; Unity Catalog table `<catalog>.<prompt_schema>.claim_settlement_review_golden` | the run is evaluated *against* the dataset when it uses all 6 cases (`eval_cases = 0`); a smaller run names it in tag `lighthouse.eval_dataset` |
# MAGIC | judge registration | **Scorers** tab: `judge_correctness`, `judge_expectations_guidelines`, `judge_safety` | registered copies name the Databricks endpoint of the judge model; the evaluation itself calls the judge through Unity Gateway (tracked) |
# MAGIC | review session | **Review App** labeling session with the candidate's traces, assigned to you | label `lighthouse_release_review`: ready / needs_work / reject, with a comment |
# MAGIC | platform records | `ai_observability.eval_run` and `eval_case_result` | queried in step 8 |
# MAGIC
# MAGIC A `SKIPPED` line means the workspace does not offer that feature (some are missing on Free
# MAGIC Edition); the evaluation still completes and the same code records it where available.

# COMMAND ----------

has_champion = any("champion" in (getattr(v, "aliases", None) or []) for v in registry.versions(PROMPT))
BASELINE = "@champion" if has_champion else "latest"
evaluation = run_cli(["eval", "--prompt", CANDIDATE, "--baseline", BASELINE, "--baseline-optional",
                      "--model", W["app_model"], "--judge-model", W["judge_model"],
                      "--limit", W["eval_cases"], "--review-session", "--review-user", me, "--json"],
                     echo=False)
result = evaluation.data
for role in ("candidate", "baseline"):
    if role in result:
        r = result[role]
        print(f"{role:<9} prompt {r['prompt_version']:<12} run {r['run_id']}  combined "
              f"{r.get('combined_mean', 0):.1%}  verdict {r.get('verdict')}")
        for item in r.get("recording", {}).values():   # done, or skipped with the reason
            print(f"          recorded  {item['name']:<20} {item['status'].upper():<8} {item['detail']}")
if "comparison" in result:
    display(spark.createDataFrame(result["comparison"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6. Inspect the runs in MLflow Experiments
# MAGIC Open the links below (they open in a new tab, so this notebook stays open). What to look at:
# MAGIC
# MAGIC | Where | What it tells you |
# MAGIC |---|---|
# MAGIC | **Runs** table | one run per prompt version; metrics `lighthouse/combined_score`, `lighthouse/hard_gate_pass_rate`, tag `lighthouse.verdict` |
# MAGIC | Run → **Overview / Parameters** | the full lineage: prompt name/version, model, judge, dataset version, spec, git commit |
# MAGIC | Run → **Metrics** | `lighthouse/rubric_score`, `lighthouse/judge_score`, `lighthouse/combined_score`, `lighthouse/judge_errors` |
# MAGIC | Run → **Traces** / **Evaluations** | one trace per case: input, output, every assessment with the **judge's rationale** |
# MAGIC | Run → **Artifacts** → `lighthouse_review.json` | the per-case review the CLI printed |
# MAGIC | Select both runs → **Compare** | candidate vs baseline side by side |
# MAGIC | **Traces** tab (filter `lighthouse.kind = try`) | the step-3 trial |
# MAGIC | **Prompts** tab, or run -> linked prompts | the prompt version each run evaluated (registered versions) |
# MAGIC | **Datasets** tab | the golden set as an evaluation dataset (Unity Catalog table), its records and the runs that used it |
# MAGIC | **Scorers** tab | the three LLM judges of the spec |
# MAGIC | **Review App** link below | the labeling session: review each case as an examiner / SME; labels appear on the traces |

# COMMAND ----------

links = [f'<p><a href="{experiment_url(HOST, experiment.experiment_id)}" target="_blank">Experiment: {W["experiment"]}</a></p>']
for role in ("candidate", "baseline"):
    if role in result:
        links.append(f'<p><a href="{experiment_url(HOST, experiment.experiment_id, result[role]["run_id"])}" '
                     f'target="_blank">{role} run ({result[role]["prompt_version"]})</a></p>')
session = result.get("candidate", {}).get("recording", {}).get("review_session", {})
if session.get("status") == "done" and "https://" in session.get("detail", ""):
    links.append(f'<p><a href="{session["detail"].split(": ", 1)[-1]}" target="_blank">Review App labeling session (candidate)</a></p>')
displayHTML("".join(links))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7. The review gate (`lighthouse promote`)
# MAGIC Publishing is a human decision. `promote` checks the evidence first. The run must be a
# MAGIC `lighthouse eval` run of **this registered version**, not BLOCKED, scored with the current spec,
# MAGIC and judges must have run. Only then does it record the reviewer, rationale and evidence, and
# MAGIC move `@champion`. Drafts cannot be promoted.
# MAGIC
# MAGIC This cell runs a **dry run** (shows the checks, writes nothing) unless `promote_for_real = true`.
# MAGIC To publish, use **step 9** at the end: it promotes only when every check is green.
# MAGIC For real, it records: the approval file and tags on the run and prompt version, the decision as
# MAGIC a **human assessment** (`lighthouse_release_review`) on every trace of the evidence run (next to
# MAGIC the judges), a row in `ai_marketplace.release_decision`, and, on approval, moves `@champion`.
# MAGIC Fill in `REVIEWER` and `RATIONALE` honestly: they are the audit record.

# COMMAND ----------

REVIEWER = f"{me} (prompt engineer)"
RATIONALE = "Reviewed per-case judge rationales and failed criteria in MLflow; no decisional language."
if CANDIDATE.startswith("file:"):
    print("The candidate is a draft: set register_draft = true and re-run steps 4-5 to promote it.")
else:
    args = ["promote", "--prompt", CANDIDATE, "--run", result["candidate"]["run_id"],
            "--reviewer", REVIEWER, "--rationale", RATIONALE]
    run_cli(args if W["promote_for_real"] == "true" else args + ["--dry-run"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8. See it tracked
# MAGIC Steps 5 and 7 are in Unity Catalog next to the usage data:
# MAGIC - **Evaluation runs**: `ai_observability.eval_run` (one row per run: prompt version, models,
# MAGIC   dataset, combined score, verdict) and `eval_case_result` (per case and criterion: gate, rubric,
# MAGIC   judge, combined). Scores and IDs only; rationales and outputs stay on the MLflow traces.
# MAGIC - **Review decisions**: `ai_marketplace.release_decision` (who decided what, on which run).
# MAGIC
# MAGIC Every call of steps 3 and 5 is in the platform tables:
# MAGIC - **Assistant calls**: `model_invocation`, written at the end of each command, with activity,
# MAGIC   prompt version, MLflow run ID, synthetic case ID and trace ID.
# MAGIC - **All calls, judges included**: `system.ai_gateway.usage`, through the request tags.
# MAGIC   Those rows arrive with a delay (often 10-40 minutes). The cell copies what has arrived into
# MAGIC   `gateway_usage`, which the views read; re-run it later to see the judge calls.
# MAGIC
# MAGIC The dashboard (`04_observability_dashboard`) shows the same data. Filter **Activity** = `try`,
# MAGIC `eval` and **Use case** = `claim_settlement_assistant`, or open **Model drill-down** and search
# MAGIC the run ID.

# COMMAND ----------

from lighthouse_poc.config import load_profile
from lighthouse_poc.platform.gateway_usage import GatewayUsageSync
from lighthouse_poc.platform.naming import PlatformNames
from lighthouse_poc.platform.runtime import warehouse_engine

names = PlatformNames(W["catalog"], W["platform_prefix"])
OBS = f"{names.catalog}.{names.observability}"
engine = warehouse_engine(load_profile("databricks"), names, logger)
try:
    print("gateway rows copied now:", GatewayUsageSync(names, logger).sync(engine))
finally:
    engine.dispose()

run_ids = ", ".join(f"'{result[r]['run_id']}'" for r in ("candidate", "baseline") if r in result)
# per run, purpose and model: calls, tokens and cost (judge rows appear once the gateway delivers them)
display(spark.sql(f"""
SELECT run_id, activity, purpose, model_key, count(*) AS calls, sum(is_error) AS errors,
       sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
       round(sum(total_cost_usd), 6) AS cost_usd, sum(CAST(app_recorded AS INT)) AS app_events,
       sum(CAST(gateway_confirmed AS INT)) AS gateway_rows
FROM {OBS}.v_model_call_cost
WHERE run_id IN ({run_ids or "''"}) OR (activity = 'try' AND event_time > current_timestamp() - INTERVAL 2 HOURS)
GROUP BY ALL ORDER BY run_id, purpose, model_key"""))
MKT = f"{names.catalog}.{names.marketplace}"
# evaluation runs and review decisions in Unity Catalog (MLflow keeps traces and rationales)
display(spark.sql(f"""
SELECT run_id, prompt_version, model_key, judge_model_key, dataset_name, cases,
       round(combined_score, 3) AS combined_score, verdict, started_at, finished_at
FROM {OBS}.eval_run WHERE run_id IN ({run_ids or "''"})"""))
display(spark.sql(f"""
SELECT case_id, kind, criterion, score, passed, run_id
FROM {OBS}.eval_case_result WHERE run_id IN ({run_ids or "''"})
ORDER BY run_id, case_id, kind, criterion"""))
display(spark.sql(f"""
SELECT decided_at, prompt_name, prompt_version, decision, reviewer, eval_run_id, review_score, verdict
FROM {MKT}.release_decision ORDER BY decided_at DESC LIMIT 20"""))
# the assistant's call events of this session (no prompt or response text is stored)
display(spark.sql(f"""
SELECT event_time, activity, purpose, model_key, prompt_version, case_id, status, latency_ms,
       input_tokens, output_tokens, run_id, trace_id
FROM {OBS}.model_invocation
WHERE activity IN ('try', 'eval') AND event_time > current_timestamp() - INTERVAL 2 HOURS
ORDER BY event_time DESC"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9. Promote when the run is green
# MAGIC Run this cell to publish the candidate. It promotes **only if** all of these hold, and prints
# MAGIC each check either way:
# MAGIC - the candidate is a **registered version** (drafts cannot be published: `register_draft = true`)
# MAGIC - verdict `READY FOR HUMAN REVIEW`
# MAGIC - **no hard gate failed** (hard gate pass rate 100%)
# MAGIC - combined score **at or above 80%**
# MAGIC
# MAGIC Then `lighthouse promote` re-checks the evidence on the MLflow run (same version, finished run,
# MAGIC unchanged spec) and records the decision: approval file and tags on the run and prompt version,
# MAGIC a human assessment on every trace, a `release_decision` row, and `@champion` moved to the version.
# MAGIC **Running this cell is your human approval**: `REVIEWER` and `RATIONALE` are the audit record, so
# MAGIC read the judge rationales (step 6) or the Review App labels first.

# COMMAND ----------

# DBTITLE 1,9. Promote the candidate if every check is green
REVIEWER = f"{me} (prompt engineer)"
RATIONALE = "Reviewed per-case judge rationales and failed criteria in MLflow; no decisional language."
MIN_SCORE = 0.80

cand = result.get("candidate", {})
gate_checks = [
    ("registered version", not CANDIDATE.startswith("file:"), f"candidate {CANDIDATE}"),
    ("verdict ready", cand.get("verdict") == "READY FOR HUMAN REVIEW", f"verdict {cand.get('verdict')}"),
    ("no hard gate failed", cand.get("hard_gate_pass_rate") == 1.0,
     f"hard gate pass rate {cand.get('hard_gate_pass_rate', 0):.0%}"),
    ("score >= 80%", (cand.get("combined_mean") or 0) >= MIN_SCORE,
     f"combined {cand.get('combined_mean', 0):.1%}"),
]
for name, ok, detail in gate_checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<20} {detail}")
if all(ok for _, ok, _ in gate_checks):
    promotion = run_cli(["promote", "--prompt", CANDIDATE, "--run", cand["run_id"],
                         "--reviewer", REVIEWER, "--rationale", RATIONALE])
    print("PROMOTED" if promotion.exit_code == 0 else "NOT PROMOTED: promote refused (see its checks above)")
else:
    print("NOT PROMOTED: fix the failing checks (prompt, cases), re-run steps 2-5, then this cell.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC | Symptom | Cause / fix |
# MAGIC |---|---|
# MAGIC | `No registered versions` / permission denied on the schema | use your own `prompt_schema` (`dev_<you>_lighthouse`) or ask for `USE SCHEMA` + `CREATE FUNCTION` on a shared one |
# MAGIC | `@champion` not found | expected before the first approval; the baseline falls back to `latest` |
# MAGIC | `usage_events_skipped ... schemas are at base` | run notebook 02 with your `platform_prefix` first |
# MAGIC | verdict `BLOCKED` | a hard gate failed: open the case's trace in MLflow and read which gate |
# MAGIC | a case shows `empty_response` | a reasoning model used its whole token budget; try again or use `claims-small` |
# MAGIC | judge rows missing in step 8 | the gateway usage table lags; re-run step 8 later |
# MAGIC | `HTTP 429` in a run | rate limited on Free Edition: evaluate fewer cases (`eval_cases`) |
# MAGIC | `recorded ... SKIPPED not available here` | the workspace does not offer that feature (Free Edition); nothing to fix, the run is complete |
# MAGIC | `recorded platform_records SKIPPED ... schemas are at` | run notebook 02 (or 04) with your `platform_prefix` to upgrade the schemas |
# MAGIC
# MAGIC Terminal equivalent of this notebook: `docs/developer-workflow.md` (`lighthouse prompt list`,
# MAGIC `try`, `eval`, `promote`).

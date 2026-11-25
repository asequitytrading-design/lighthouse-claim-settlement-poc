# Developer workflow: authoring and testing prompts

Developers work with prompts through the `lighthouse` CLI. It is installed with the package
(`uv pip install -e '.[dev]'` or `pip install -e '.[dev]'`) and runs the same code against
the local stack (Ollama via Kong, local MLflow) and Databricks (Unity Gateway, Unity Catalog
prompts). `--profile` selects which one. Synthetic cases only.

| Need | Command | Status |
|---|---|---|
| See versions, aliases, change notes | `lighthouse prompt list` | available |
| Read one version, an alias or a draft | `lighthouse prompt show <ref>` | available |
| Start a draft from an existing version | `lighthouse prompt show 3 --template-only > prompts/draft.txt` | available |
| Compare two versions or a draft | `lighthouse prompt diff <old> <new>` | available |
| Try one golden case | `lighthouse try --prompt <ref> --case <id> --model <alias>` | available |
| Golden evaluation of a draft or version, compared with a baseline | `lighthouse eval --prompt <ref> --baseline <ref>` | available |
| Approve or reject a version with evidence (moves `@champion`) | `lighthouse promote` | available |
| Register a version from the CLI | `lighthouse prompt register` | planned |

Register versions with `scripts/register_prompt.py` or by saving a version in the MLflow UI
Playground. `scripts/evaluate.py` remains the job entry point.

## Prompt references

Every command that takes a prompt accepts the same references:

| Reference | Meaning |
|---|---|
| `3` | Registered version 3 (immutable) |
| `@champion` | The version an alias points to |
| `latest` | Highest registered version |
| `file:prompts/draft.txt` | An **unregistered draft** on disk |

A draft is labelled `draft-<first 8 hex of SHA-256 of the text>`. The label appears in the
output, logs and the MLflow trace, so a result can always be tied to the exact text, and the
same text gets the same label locally and on Databricks. Iterating on a draft creates no
registry versions. Only registered versions can be approved and published.

## The loop

```bash
lighthouse prompt list                                   # what exists, what is @champion
lighthouse prompt show latest --template-only > prompts/draft.txt
# edit prompts/draft.txt
lighthouse prompt diff latest file:prompts/draft.txt     # sentence-level diff
lighthouse cases                                         # golden case IDs
lighthouse try --prompt file:prompts/draft.txt --case CSET-LTD-002 --model mock          # wiring
lighthouse try --prompt file:prompts/draft.txt --case CSET-LTD-002 --model claims-small  # real model
```

`try` prints the model output, the three deterministic gates (`output_schema_valid`,
`non_decision_boundary`, `expected_fields_match`) and the MLflow trace ID. The trace is
tagged `lighthouse.kind=try`, `lighthouse.prompt_version`, `lighthouse.prompt_reference`,
`lighthouse.model` and `lighthouse.profile`, so trials can be filtered apart from evaluation
runs. The exit code is 0 when all gates pass, 1 when a gate fails and 2 for configuration
or registry errors. `--json` prints a machine-readable result.

One case is a smoke test. A prompt change needs the full golden evaluation before review.

`prompt diff` compares sentence by sentence by default, because templates are often one long
paragraph; `--by line` compares lines.

## Golden evaluation and the review score

```bash
lighthouse eval --prompt file:prompts/draft.txt --baseline @champion --model claims-medium
lighthouse eval --spec claim_intake_triage --prompt @candidate --model claims-small
lighthouse eval --prompt latest --model claims-small --no-judges      # fast: gates + rubric only
```

`--spec` names an evaluation spec in `config/evals/` (default: the profile prompt's spec,
`claim_settlement_review`). The spec says how the prompt is run (`adapter`), which dataset
to use, and how outputs are scored:

| Layer | What | Effect |
|---|---|---|
| Hard gates | Deterministic, e.g. valid JSON, required fields, `benefit_determination = Not determined` | Any failure scores the case 0 and the run **BLOCKED** |
| Rubric | Deterministic weighted criteria, 0–1 each: expected fields (partial credit), list types, no amounts or decision words in the reply, no echo of identifiers from the input | Weighted mean = rubric score |
| LLM judges | MLflow built-ins on the profile judge model: Guidelines (spec rules), ExpectationsGuidelines (per-case rules), Safety, Correctness (expected facts) | yes = 1, no = 0; mean = judge score; judge errors are excluded and counted |

Per case: **combined = 0 if a gate failed, otherwise 0.6 × rubric + 0.4 × judges** (weights per
spec; rubric only when no judge gave a verdict). Per run, the mean combined score gives the
verdict: **BLOCKED**, **NEEDS WORK** (below `pass_threshold`, default 0.8) or **READY FOR HUMAN
REVIEW**. The verdict is evidence for a human reviewer, never an approval. With `--baseline`,
both prompts run on the same cases, model and judges, and a per-case delta table follows.

Each run is an MLflow evaluation run with:
- per-trace assessments `gate_*`, `rubric_*` and `judge_*` (value and rationale), plus
  `lighthouse_combined_score`;
- metrics `lighthouse/combined_score`, `rubric_score`, `judge_score`, `hard_gate_pass_rate`,
  `cases_below_threshold` and `judge_errors`;
- the tag `lighthouse.verdict`, the artifact `lighthouse_review.json`, and lineage params
  (prompt reference and version, model, judge, dataset, spec digest, weights, Git commit).

Exit code: 1 when BLOCKED, 0 otherwise, 2 for configuration errors. `--dataset
mlflow:<name>` evaluates an MLflow dataset instead of the spec's JSONL.

Prompts saved from the MLflow UI Playground are chat prompts (a list of messages). The
`chat_prompt` adapter renders their `{{variables}}` from each record's inputs. To iterate
on one as a file, `lighthouse prompt show <ref> --template-only > draft.json` and use
`file:draft.json`.

## Review gate: approve or reject with an approval record

```bash
lighthouse promote --prompt 2 --run <eval run id> --reviewer "A. Reviewer, claims SME" \
    --rationale "Read all judge rationales; TRI-N06 reply wording accepted." --dry-run
lighthouse promote --prompt 2 --run <eval run id> --reviewer "..." --rationale "..."   # approve
lighthouse promote --prompt 2 --run <eval run id> --reviewer "..." --rationale "..." --reject
```

Approval is refused, and nothing is written, unless every evidence check passes:

| Check | Meaning |
|---|---|
| `registered_version` | The prompt is a registered version, not a `file:` draft |
| `eval_run` / `run_finished` | The run was created by `lighthouse eval` and finished |
| `same_prompt` | The run evaluated exactly this prompt name and version |
| `verdict_ready` / `hard_gates_all_pass` | Verdict READY FOR HUMAN REVIEW, 100% hard gates |
| `spec_unchanged` | The evaluation spec is the same as when the run was scored |

A rejection only needs the first four checks. `--reviewer` and `--rationale` are required
(at least 20 characters). `--dry-run` shows the checks and the alias change without writing.

The decision is written in this order (evidence first, pointer last):
1. `approvals/<record_id>.json` on the evaluation run, with the decision, reviewer, rationale,
   UTC time, local user, prompt name and version, alias and its previous target, model, judge,
   dataset and version, spec digest, scores, verdict, evaluated Git commit, and the checks.
   Every decision adds a file, so the history is kept.
2. `lighthouse_approval_*` tags (decision, reviewer, decided_at, record_id, run_id) on the run
   and on the prompt version, holding the latest decision. Underscore keys are used because
   Unity Catalog prompt tags reject dots.
3. On approval only, the alias (default `champion`) moves to the version.

`lighthouse prompt list` shows a REVIEW column and the aliases. `prompt show` shows the latest
review. To roll back, promote the earlier version again with its own READY evaluation run.
The approval record concerns a prompt release, never a claim.

Verified 2026-11-25 on the local server and on Unity Catalog (Free Edition), with throwaway
prompts that were deleted afterwards. The test covered the refusals (wrong version, draft,
short rationale), the dry run, approvals, the alias move from 1 to 2 with the rollback hint,
the rejection, and the history of two records on one run.

## Against Databricks from a laptop

Authenticate the Databricks CLI once (`databricks auth login ... --profile lighthouse`), then:

```bash
export DATABRICKS_CONFIG_PROFILE=lighthouse LIGHTHOUSE_PROFILE=databricks
export LIGHTHOUSE_SCHEMA=dev_<you>_lighthouse
export UNITY_GATEWAY_MEDIUM_MODEL=system.ai.qwen35-122b-a10b
export LIGHTHOUSE_EXPERIMENT="/Workspace/Users/<you>/[dev <you>] lighthouse-claim-settlement-poc-dev"
lighthouse prompt list
lighthouse try --prompt file:prompts/draft.txt --case CSET-LTD-002 --model claims-medium
```

Environment variables and `--env KEY=VALUE` are interchangeable; `--experiment` overrides
the experiment for one command.

- `LIGHTHOUSE_SCHEMA`: your `dev` bundle schema is prefixed (`dev_<user>_lighthouse`). The `ci`
  schema `lighthouse` belongs to the CI service principal; reading it needs a grant
  (`PERMISSION_DENIED` otherwise).
- `--experiment`: in dev mode the bundle names the experiment
  `[dev <user>] lighthouse-claim-settlement-poc-dev`. A path that does not exist is created by
  MLflow, so copy the exact name from **Experiments**.
- Model calls use your identity, so you need access to the Unity Gateway model services.

Verified 2026-11-25: `prompt list` and `show` on Unity Catalog; the same draft
(`draft-5adc6164`) passed all three gates for CSET-LTD-002 locally on qwen3.5-4b through Kong
(about 14 s) and on Databricks on qwen35-122b-a10b through Unity Gateway (about 52 s).

## Checking a trial in the MLflow UI

1. **Experiments** → the profile's experiment (local `/lighthouse-claim-settlement-poc`) →
   **Traces** tab at the experiment level. Trials are not attached to a run, so they do not
   appear in an evaluation run's Traces tab (which shows only that run's 6 cases).
2. Filter with a quoted value: `tag.lighthouse.kind = 'try'` (unquoted `=try` matches nothing).
   Narrow further with `tag.lighthouse.prompt_version = 'draft-236b127c'`.
3. Open the trace: span tree `try_case` → `predict` with the case input and model output; the
   `lighthouse.*` tags hold prompt reference, version label, model and profile.

## Where results live

To do the same from the MLflow UI (Playground, prompt versions, judges), see
[mlflow-ui-guide.md](mlflow-ui-guide.md).

- **MLflow UI** (local `http://localhost:5000`, or the Databricks workspace): prompt versions
  and aliases, `try` traces, evaluation runs and their side-by-side comparison.
- Approval decisions (step 3) will be recorded on the prompt version and the evaluation run.
  An alias move alone is not an approval record.

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
| Golden evaluation of a draft or version, compared with a baseline | `lighthouse eval ... --baseline @champion` | planned (step 2) |
| Register, approve and publish (move `@champion`) | `lighthouse prompt register`, `lighthouse promote` | planned (steps 2–3) |

Until steps 2–3 land, register with `scripts/register_prompt.py` and run the golden evaluation
with `scripts/evaluate.py` (see [local-development.md](local-development.md)).

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

- **MLflow UI** (local `http://localhost:5000`, or the Databricks workspace): prompt versions
  and aliases, `try` traces, evaluation runs and their side-by-side comparison.
- Approval decisions (step 3) will be recorded on the prompt version and the evaluation run.
  An alias move alone is not an approval record.

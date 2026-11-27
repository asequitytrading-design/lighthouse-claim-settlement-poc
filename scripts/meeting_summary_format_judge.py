"""Code judge for the meeting-summariser output format (local experiment only).

A deterministic companion to the LLM judges on the user's local `meeting-summariser` prompt
(experiment 10, `/lighthouse-ui-sample`). It checks structure only (rules R1-R6 below); content
quality stays with the `summary_*` LLM judges. Local only: it refuses a non-local tracking URI and
is excluded from the Databricks bundle sync, because that experiment's traces never leave the machine.

Usage (from the project root, local MLflow only):
  # 1. Try the rules on traces without registering anything:
  MLFLOW_TRACKING_URI=http://localhost:5000 .venv/bin/python scripts/meeting_summary_format_judge.py --dry-run tr-xxxx tr-yyyy
  # 2. Register it in experiment 10 so it shows on the Judges tab:
  MLFLOW_TRACKING_URI=http://localhost:5000 MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS=true \
      .venv/bin/python scripts/meeting_summary_format_judge.py --register

  # 3. Run it on traces (the 3.18 UI cannot run code judges; the result still appears on the
  #    Evaluation Runs page like a UI run):
  MLFLOW_TRACKING_URI=http://localhost:5000 .venv/bin/python scripts/meeting_summary_format_judge.py --run tr-xxxx tr-yyyy

Edit the rules inside the function body (MLflow stores only the function source, so
everything the judge needs must live inside it), then re-run --dry-run and --register.
"""
import argparse
import sys

from mlflow.genai.scorers import scorer

JUDGE_NAME = "summary_json_format"
EXPERIMENT_ID = "10"


@scorer
def summary_json_format(outputs):
    """Pass only if the notes follow the meeting-summariser JSON contract."""
    import json
    import re

    from mlflow.entities import Feedback

    # ---- Rules: edit here -------------------------------------------------------------
    REQUIRED_KEYS = ["title", "summary", "people", "decisions", "action_items", "open_questions"]
    STRING_LISTS = ["summary", "decisions", "open_questions"]   # lists of non-empty strings
    ACTION_ITEM_KEYS = ["owner", "task", "due"]                  # every action item has these
    PLACEHOLDERS = {"undefined", "null", "none", "n/a", "na", "unknown", "tbd", ""}
    CHECK_PLACEHOLDERS = True    # R6: unknowns are JSON null, except where the contract says otherwise
    # The prompt contract writes "undefined" for an unstated role/organisation/reports_to and
    # pairs each with an evidence time (R7); "unassigned" is the contract's unnamed owner.
    EVIDENCE_FIELDS = {"role": "role_at", "organisation": "organisation_at", "reports_to": "reports_to_at"}
    CONTRACT_UNKNOWN = "undefined"
    TIME = re.compile(r"^\[?\d{1,2}:\d{2}(:\d{2})?\]?$")
    SPEAKER_LABEL = re.compile(r"\bSPEAKER_\d+\b")                                  # R8
    MONEY = re.compile(r"[$€£₹]\s?\d|\b\d[\d.,]*\s?(k|m|mn|million|billion|lakhs?|crores?)?\s?"
                       r"(usd|inr|eur|gbp|dollars?|rupees?|euros?|pounds?)\b|"
                       r"\b(usd|inr|eur|gbp|rs\.?)\s?\d", re.IGNORECASE)            # R9
    # -----------------------------------------------------------------------------------

    def parse(value):
        # Unwrap up to twice: a trace may hold the chat completion as a dict or as a JSON
        # string, and the completion's content is itself the (possibly fenced) notes JSON.
        for _ in range(3):
            if isinstance(value, dict) and value.get("choices"):      # Playground / gateway
                value = (value["choices"][0].get("message") or {}).get("content", "")
            if isinstance(value, str):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
                try:
                    value = json.loads(text)
                except ValueError:
                    return None
            if not (isinstance(value, dict) and value.get("choices")):
                break
        return value if isinstance(value, dict) else None

    def is_text(v):
        return isinstance(v, str) and v.strip() != ""

    data = parse(outputs)
    if data is None:
        return Feedback(value=False, rationale="R1 failed: output is not a JSON object")

    failures = []
    # R2 required keys
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        failures.append(f"R2 missing keys: {missing}")
    # R3 title and string lists
    if "title" in data and not is_text(data["title"]):
        failures.append("R3 title is empty or not a string")
    for key in STRING_LISTS:
        if key in data:
            v = data[key]
            if not isinstance(v, list):
                failures.append(f"R3 {key} is {type(v).__name__}, not a list")
            elif not all(is_text(x) for x in v):
                failures.append(f"R3 {key} has empty or non-string items")
    # R4 people
    people = data.get("people")
    if people is not None:
        if not isinstance(people, list):
            failures.append(f"R4 people is {type(people).__name__}, not a list")
        else:
            bad = [i for i, p in enumerate(people) if not (isinstance(p, dict) and is_text(p.get("name")))]
            if bad:
                failures.append(f"R4 people without a name at positions {bad}")
    # R5 action items
    items = data.get("action_items")
    if items is not None:
        if not isinstance(items, list):
            failures.append(f"R5 action_items is {type(items).__name__}, not a list")
        else:
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    failures.append(f"R5 action_items[{i}] is not an object")
                    continue
                gone = [k for k in ACTION_ITEM_KEYS if k not in item]
                if gone:
                    failures.append(f"R5 action_items[{i}] missing {gone}")
                if not is_text(item.get("owner")):
                    failures.append(f"R5 action_items[{i}] has no owner")
                if not is_text(item.get("task")):
                    failures.append(f"R5 action_items[{i}] has no task")
                due = item.get("due")
                if due is not None and not is_text(due):
                    failures.append(f"R5 action_items[{i}].due must be text or null")
    # R6 placeholder text instead of null
    if CHECK_PLACEHOLDERS:
        found = []

        def walk(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")
            elif isinstance(node, str) and node.strip().lower() in PLACEHOLDERS:
                field = path.rsplit(".", 1)[-1]
                allowed = (path.startswith("$.people[") and field in EVIDENCE_FIELDS
                           and node.strip().lower() == CONTRACT_UNKNOWN)
                if not allowed:
                    found.append(path)

        walk(data, "$")
        if found:
            shown = ", ".join(found[:5]) + (f" (+{len(found) - 5} more)" if len(found) > 5 else "")
            failures.append(f"R6 placeholder text instead of null at {len(found)} place(s): {shown}")
    # R7 evidence times: a stated field has its transcript time, an "undefined" one has null
    if isinstance(people, list):
        for i, person in enumerate(p for p in people if isinstance(p, dict)):
            for field, at in EVIDENCE_FIELDS.items():
                value, time = person.get(field), person.get(at)
                stated = is_text(value) and value.strip().lower() != CONTRACT_UNKNOWN
                if stated and not (isinstance(time, str) and TIME.match(time.strip())):
                    failures.append(f"R7 people[{i}].{field} is stated without a valid {at}")
                elif not stated and time is not None:
                    failures.append(f"R7 people[{i}].{at} must be null when {field} is not stated")
    # R8 / R9 scan every text value
    texts = []

    def collect(node):
        if isinstance(node, dict):
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)
        elif isinstance(node, str):
            texts.append(node)

    collect(data)
    if any(SPEAKER_LABEL.search(t) for t in texts):
        failures.append("R8 a SPEAKER_xx label appears in the notes")
    if any(MONEY.search(t) for t in texts):
        failures.append("R9 a money amount appears in the notes")

    if failures:
        return Feedback(value=False, rationale="; ".join(failures))
    return Feedback(value=True, rationale="All format rules R1-R9 pass")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run or register the meeting-summary format judge")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", nargs="+", metavar="TRACE_ID", help="score these traces locally")
    group.add_argument("--register", action="store_true", help=f"register in experiment {EXPERIMENT_ID}")
    group.add_argument("--run", nargs="+", metavar="TRACE_ID",
                       help="run the registered judge on these traces as a UI evaluation run")
    args = parser.parse_args()

    import mlflow

    uri = mlflow.get_tracking_uri()
    if not uri.startswith(("http://localhost", "http://127.0.0.1")):
        print(f"Refusing: tracking URI {uri!r} is not the local MLflow (these traces stay local)")
        return 2

    if args.dry_run:
        for trace_id in args.dry_run:
            trace = mlflow.get_trace(trace_id)
            fb = summary_json_format(outputs=trace.data.response)
            print(f"{trace_id}: {'PASS' if fb.value else 'FAIL'}  {fb.rationale}")
        return 0

    if args.run:
        # The MLflow 3.18 UI cannot run code judges ("Run judges" lists LLM judges only), so
        # call the same server endpoint the UI uses. The server resolves the registered judge
        # by name and version, and the result appears on the Evaluation Runs page.
        import json

        import requests

        base = uri.rstrip("/")
        listing = requests.get(f"{base}/ajax-api/3.0/mlflow/scorers/list",
                               params={"experiment_id": EXPERIMENT_ID}, timeout=30)
        listing.raise_for_status()
        versions = [s["scorer_version"] for s in listing.json().get("scorers", [])
                    if s["scorer_name"] == JUDGE_NAME]
        if not versions:
            print(f"{JUDGE_NAME} is not registered in experiment {EXPERIMENT_ID}; run --register first")
            return 1
        response = requests.post(
            f"{base}/ajax-api/3.0/mlflow/genai/evaluate/invoke",
            json={"experiment_id": EXPERIMENT_ID, "trace_ids": args.run,
                  "serialized_scorers": [json.dumps({"name": JUDGE_NAME})],
                  "scorer_versions": [max(versions)]},
            timeout=60,
        )
        if not response.ok:
            print(f"Run failed: HTTP {response.status_code} {response.text[:500]}")
            return 1
        print(f"Started evaluation run {response.json().get('run_id')} on {len(args.run)} trace(s)")
        return 0

    from mlflow.genai.scorers import delete_scorer, list_scorers

    if JUDGE_NAME in {s.name for s in list_scorers(experiment_id=EXPERIMENT_ID)}:
        delete_scorer(name=JUDGE_NAME, experiment_id=EXPERIMENT_ID, version="all")
    summary_json_format.register(name=JUDGE_NAME, experiment_id=EXPERIMENT_ID)
    print(f"Registered {JUDGE_NAME} in experiment {EXPERIMENT_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

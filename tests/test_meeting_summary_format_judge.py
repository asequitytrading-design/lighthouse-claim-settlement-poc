"""Format code judge for the local meeting-summariser prompt: rules R1-R9 on synthetic outputs."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("meeting_summary_format_judge",
                                              ROOT / "scripts" / "meeting_summary_format_judge.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
judge = module.summary_json_format

GOOD = {"title": "Weekly sync", "summary": ["Reviewed the plan."],
        "people": [{"name": "Alex", "role": None}], "decisions": ["Ship on Friday."],
        "action_items": [{"owner": "Alex", "task": "Send the deck.", "due": None}],
        "open_questions": []}


def as_chat(output):
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(output)}}]}


@pytest.mark.parametrize("outputs", [
    GOOD,
    "```json\n" + json.dumps(GOOD) + "\n```",
    as_chat(GOOD),
    json.dumps(as_chat(GOOD)),          # how Playground traces store the response
])
def test_good_output_passes_in_every_trace_shape(outputs):
    assert judge(outputs=outputs).value is True


@pytest.mark.parametrize("outputs, rule", [
    ("Here are your notes", "R1"),
    ({k: v for k, v in GOOD.items() if k != "open_questions"}, "R2"),
    (dict(GOOD, decisions="Ship on Friday."), "R3"),
    (dict(GOOD, people=[{"role": "lead"}]), "R4"),
    (dict(GOOD, action_items=[{"owner": "", "task": "Send the deck.", "due": None}]), "R5"),
    (dict(GOOD, action_items=[{"owner": "Alex", "task": "Send the deck."}]), "R5"),
    (dict(GOOD, action_items=[{"owner": "Alex", "task": "Send the deck.", "due": "null"}]), "R6"),
    (dict(GOOD, people=[{"name": "Alex", "notes": "undefined"}]), "R6"),
    (dict(GOOD, people=[{"name": "Alex", "role": "Team lead", "role_at": None}]), "R7"),
    (dict(GOOD, people=[{"name": "Alex", "role": "undefined", "role_at": "00:01:02"}]), "R7"),
    (dict(GOOD, summary=["SPEAKER_01 presented the plan."]), "R8"),
    (dict(GOOD, decisions=["Budget fixed at $40k."]), "R9"),
    (dict(GOOD, summary=["The client offered 5 lakh rupees."]), "R9"),
])
def test_each_rule_fails_with_its_label(outputs, rule):
    feedback = judge(outputs=outputs)
    assert feedback.value is False
    assert rule in feedback.rationale


def test_contract_unknowns_and_evidence_times_pass():
    person = {"name": "Alex", "role": "Team lead", "role_at": "00:03:15",
              "organisation": "undefined", "organisation_at": None,
              "reports_to": "undefined", "reports_to_at": None, "notes": "Owns the plan."}
    outputs = dict(GOOD, people=[person],
                   action_items=[{"owner": "unassigned", "task": "Book a room.", "due": None}],
                   summary=["Reviewed 3 dashboards for 2 teams on 12 March."])
    feedback = judge(outputs=outputs)
    assert feedback.value is True, feedback.rationale


def test_registered_source_is_self_contained():
    # MLflow stores only the function source, so imports must live inside the body.
    source = judge.model_dump()["call_source"]
    assert "import json" in source and "from mlflow.entities import Feedback" in source

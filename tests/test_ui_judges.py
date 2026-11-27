"""UI code judges: self-contained, accept trace output shapes, agree with the spec rubric."""
import json

import pytest

from lighthouse_poc.rubric import evaluate_criterion, load_spec
from lighthouse_poc.ui_judges import CODE_JUDGES

SPEC = load_spec("claim_intake_triage")
CRITERIA = {c.name: c for c in SPEC.hard_gates + SPEC.rubric}
GOOD = {"request_type": "new_claim", "product_line": "dental", "requires_human_review": False,
        "missing_information": [], "reply_draft": "Thank you; the claims team will review it.",
        "benefit_determination": "Not determined"}
BAD = dict(GOOD, reply_draft="Approved: you will receive $500. ID 000-000-000 noted.",
           benefit_determination="Approved")
INPUTS = {"claimant_message": "My SIN is 000-000-000, receipts attached."}
EXPECT = {"expected_fields": {"request_type": "new_claim", "product_line": "dental"}}


def as_chat(output):
    return {"id": "x", "choices": [{"message": {"role": "assistant",
                                                "content": "```json\n" + json.dumps(output) + "\n```"}}]}


def call(judge, outputs):
    signature = judge.model_dump()["call_signature"]          # e.g. "(inputs, outputs)"
    available = {"inputs": INPUTS, "outputs": outputs, "expectations": EXPECT}
    return judge(**{k: v for k, v in available.items() if k in signature})


def test_judges_are_self_contained_for_registration():
    """MLflow stores only the function body (call_source) and re-creates it on the server."""
    namespace_free = ("Feedback(", "re.", "json.")
    for judge in CODE_JUDGES:
        dump = judge.model_dump()
        source = dump["call_source"]
        assert "from mlflow.entities import Feedback" in source, judge.name
        assert "import re" in source and "import json" in source, judge.name
        assert "->" not in dump["call_signature"], f"{judge.name}: no return annotation"
        assert any(token in source for token in namespace_free)


@pytest.mark.parametrize("shape", [lambda o: o, json.dumps, as_chat])
@pytest.mark.parametrize("judge", CODE_JUDGES, ids=lambda j: j.name)
def test_ui_judges_agree_with_spec_criteria(judge, shape):
    criterion = CRITERIA.get(judge.name.split("_", 1)[1])
    for output in (GOOD, BAD):
        ui_value = float(call(judge, shape(output)).value)
        if criterion is None:
            continue
        spec_value = evaluate_criterion(criterion, INPUTS, output, EXPECT).value
        if judge.name.startswith("gate_"):
            spec_value = float(spec_value >= 1.0)
        assert ui_value == pytest.approx(spec_value), (judge.name, output["reply_draft"])


def test_unparseable_outputs_fail_closed():
    for judge in CODE_JUDGES:
        if judge.name == "rubric_expected_fields":
            continue
        assert float(call(judge, "I cannot help with that.").value) == 0.0, judge.name

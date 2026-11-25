"""Custom code judges registered in an MLflow experiment, so they appear on the UI Judges page.

MLflow stores a registered code judge as the source of one function, so each judge here is
self-contained: imports (including ``Feedback``) and helpers live inside the function body,
and there are no return annotations (MLflow re-creates the function from its source alone). They mirror the
deterministic gates and rubric of ``config/evals/claim_intake_triage.yaml`` and accept the
three output shapes found on traces:
- a dict (``lighthouse eval`` runs),
- a JSON string (possibly fenced),
- an OpenAI chat completion (Playground / AI Gateway traces: ``choices[0].message.content``).

``lighthouse eval`` stays the scoring of record (spec-driven, combined score); these copies let
reviewers run the same checks from the UI on any trace. Keep them in sync with the spec
(tests/test_ui_judges.py checks agreement on fixtures).
"""
from mlflow.genai.scorers import scorer


@scorer
def gate_non_decision_boundary(outputs):
    """Hard gate: benefit_determination is exactly "Not determined"."""
    import json
    import re

    from mlflow.entities import Feedback

    def parse(value):
        if isinstance(value, dict) and value.get("choices"):
            value = (value["choices"][0].get("message") or {}).get("content", "")
        if isinstance(value, str):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
            try:
                value = json.loads(text)
            except ValueError:
                return None
        return value if isinstance(value, dict) and "_lighthouse_error" not in value else None

    data = parse(outputs)
    if data is None:
        return Feedback(value=False, rationale="No valid JSON object in the output")
    actual = data.get("benefit_determination")
    return Feedback(value=actual == "Not determined", rationale=f"benefit_determination = {actual!r}")


@scorer
def gate_required_fields(outputs):
    """Hard gate: the triage JSON has every required key."""
    import json
    import re

    from mlflow.entities import Feedback

    def parse(value):
        if isinstance(value, dict) and value.get("choices"):
            value = (value["choices"][0].get("message") or {}).get("content", "")
        if isinstance(value, str):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
            try:
                value = json.loads(text)
            except ValueError:
                return None
        return value if isinstance(value, dict) and "_lighthouse_error" not in value else None

    required = ["request_type", "product_line", "requires_human_review", "missing_information",
                "reply_draft", "benefit_determination"]
    data = parse(outputs)
    if data is None:
        return Feedback(value=False, rationale="No valid JSON object in the output")
    missing = [k for k in required if k not in data]
    return Feedback(value=not missing,
                    rationale=f"Missing: {', '.join(missing)}" if missing else "All required fields present")


@scorer
def rubric_reply_no_amount(outputs):
    """Rubric: reply_draft contains no money amount (1 = clean)."""
    import json
    import re

    from mlflow.entities import Feedback

    def parse(value):
        if isinstance(value, dict) and value.get("choices"):
            value = (value["choices"][0].get("message") or {}).get("content", "")
        if isinstance(value, str):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
            try:
                value = json.loads(text)
            except ValueError:
                return None
        return value if isinstance(value, dict) and "_lighthouse_error" not in value else None

    data = parse(outputs)
    if data is None:
        return Feedback(value=0.0, rationale="No valid JSON object in the output")
    reply = str(data.get("reply_draft", ""))
    match = re.search(r"(\$\s?\d|\b\d[\d,.]*\s?(dollars|usd|cad)\b)", reply, re.IGNORECASE)
    return Feedback(value=0.0 if match else 1.0,
                    rationale=f"reply_draft contains {match.group(0)!r}" if match else "No amount in reply_draft")


@scorer
def rubric_reply_no_decision_language(outputs):
    """Rubric: reply_draft contains no decision or guarantee wording (1 = clean)."""
    import json
    import re

    from mlflow.entities import Feedback

    def parse(value):
        if isinstance(value, dict) and value.get("choices"):
            value = (value["choices"][0].get("message") or {}).get("content", "")
        if isinstance(value, str):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
            try:
                value = json.loads(text)
            except ValueError:
                return None
        return value if isinstance(value, dict) and "_lighthouse_error" not in value else None

    data = parse(outputs)
    if data is None:
        return Feedback(value=0.0, rationale="No valid JSON object in the output")
    reply = str(data.get("reply_draft", ""))
    pattern = (r"\b(approved|approve|denied|deny|rejected|eligible|guarantee[ds]?|"
               r"will be paid|you will receive)\b")
    match = re.search(pattern, reply, re.IGNORECASE)
    return Feedback(value=0.0 if match else 1.0,
                    rationale=f"reply_draft contains {match.group(0)!r}" if match
                    else "No decision language in reply_draft")


@scorer
def rubric_no_identifier_echo(inputs, outputs):
    """Rubric: the output does not repeat ID-like numbers from the input (1 = clean)."""
    import json
    import re

    from mlflow.entities import Feedback

    def parse(value):
        if isinstance(value, dict) and value.get("choices"):
            value = (value["choices"][0].get("message") or {}).get("content", "")
        if isinstance(value, str):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
            try:
                value = json.loads(text)
            except ValueError:
                return None
        return value if isinstance(value, dict) and "_lighthouse_error" not in value else None

    data = parse(outputs)
    if data is None:
        return Feedback(value=0.0, rationale="No valid JSON object in the output")
    pattern = re.compile(r"\b\d{3}[- ]?\d{3}[- ]?\d{3}\b")
    identifiers = set(pattern.findall(json.dumps(inputs, ensure_ascii=False)))
    if not identifiers:
        return Feedback(value=1.0, rationale="No identifiers in the input (not applicable)")
    text = json.dumps(data, ensure_ascii=False)
    echoed = sorted(i for i in identifiers if i in text)
    return Feedback(value=0.0 if echoed else 1.0,
                    rationale=f"Output repeats {len(echoed)} input identifier(s)" if echoed
                    else f"None of {len(identifiers)} input identifier(s) repeated")


@scorer
def rubric_expected_fields(outputs, expectations):
    """Rubric: share of the golden expected_fields that match (needs dataset expectations)."""
    import json
    import re

    from mlflow.entities import Feedback

    def parse(value):
        if isinstance(value, dict) and value.get("choices"):
            value = (value["choices"][0].get("message") or {}).get("content", "")
        if isinstance(value, str):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
            try:
                value = json.loads(text)
            except ValueError:
                return None
        return value if isinstance(value, dict) and "_lighthouse_error" not in value else None

    expected = (expectations or {}).get("expected_fields") or {}
    if not expected:
        return Feedback(value=1.0, rationale="No expected_fields on this trace (not applicable)")
    data = parse(outputs)
    if data is None:
        return Feedback(value=0.0, rationale="No valid JSON object in the output")
    wrong = [f"{k}={data.get(k)!r} (expected {v!r})" for k, v in expected.items() if data.get(k) != v]
    matched = len(expected) - len(wrong)
    return Feedback(value=round(matched / len(expected), 4),
                    rationale=f"{matched}/{len(expected)} expected fields match"
                    + (f"; {'; '.join(wrong)}" if wrong else ""))


CODE_JUDGES = (gate_non_decision_boundary, gate_required_fields, rubric_reply_no_amount,
               rubric_reply_no_decision_language, rubric_no_identifier_echo, rubric_expected_fields)

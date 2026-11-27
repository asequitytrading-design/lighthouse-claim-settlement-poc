"""Local meeting-summariser iteration harness: pure helpers only (no MLflow server, no model)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("meeting_summariser_iterate",
                                              ROOT / "scripts" / "meeting_summariser_iterate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_text_prompt_and_transcript_are_two_user_messages_like_the_playground():
    assert module.prompt_messages("Summarise.", "A: hi") == [
        {"role": "user", "content": "Summarise."}, {"role": "user", "content": "A: hi"}]


def test_chat_prompt_gets_the_transcript_appended():
    chat = [{"role": "system", "content": "Summarise."}]
    assert module.prompt_messages(chat, "A: hi")[-1] == {"role": "user", "content": "A: hi"}
    assert chat == [{"role": "system", "content": "Summarise."}]   # template not mutated


def test_transcript_is_the_last_user_message_of_a_playground_trace():
    request = {"n": 1, "messages": [{"role": "user", "content": "prompt"},
                                    {"role": "user", "content": "synthetic transcript"}]}
    trace = SimpleNamespace(data=SimpleNamespace(request=json.dumps(request)),
                            info=SimpleNamespace(trace_id="tr-x"))
    assert module.transcript_from_trace(trace) == "synthetic transcript"


def test_trace_without_a_transcript_message_is_rejected():
    trace = SimpleNamespace(data=SimpleNamespace(request={"messages": [{"role": "user", "content": "p"}]}),
                            info=SimpleNamespace(trace_id="tr-x"))
    with pytest.raises(SystemExit):
        module.transcript_from_trace(trace)


@pytest.mark.parametrize("uri", ["databricks", "https://dbc-123.cloud.databricks.com", "file:///tmp/m"])
def test_non_local_tracking_uri_is_refused(uri):
    with pytest.raises(SystemExit, match="Refusing"):
        module.require_local(uri)


def test_local_tracking_uri_is_accepted():
    module.require_local("http://localhost:5000")
    module.require_local("http://127.0.0.1:5000")


@pytest.mark.parametrize("value, expected", [
    (True, True), ("yes", True), ("Yes", True), (False, False), ("no", False), (None, False)])
def test_pass_values(value, expected):
    assert module._is_pass(value) is expected


def vote(name, value, at, version=3, error=None):
    return SimpleNamespace(name=name, create_time_ms=at,
                           feedback=SimpleNamespace(value=value, error=error),
                           metadata={"mlflow.assessment.scorerVersion": str(version)})


def test_majority_of_the_latest_votes_from_the_current_judge_version():
    versions = {"summary_faithful": 3}
    votes = [vote("summary_faithful", True, 1, version=2),     # older version: ignored
             vote("summary_faithful", False, 2), vote("summary_faithful", True, 3),
             vote("summary_faithful", True, 4)]
    result = module.tally(votes, versions, 3)["summary_faithful"]
    assert result["votes"] == [False, True, True] and result["verdict"] is True


def test_errors_are_not_votes_and_a_tie_fails():
    versions = {"summary_faithful": 3}
    votes = [vote("summary_faithful", True, 1), vote("summary_faithful", None, 2, error="parse"),
             vote("summary_faithful", False, 3)]
    result = module.tally(votes, versions, 3)["summary_faithful"]
    assert result == {"votes": [True, False], "errors": 1, "verdict": False}


def test_only_errors_leave_the_verdict_unresolved():
    result = module.tally([vote("summary_faithful", None, 1, error="parse")],
                          {"summary_faithful": 3}, 3)["summary_faithful"]
    assert result["verdict"] is None


def test_code_judge_needs_one_vote_llm_judge_needs_all():
    versions = {"summary_json_format": 2, "summary_faithful": 3}
    trace = SimpleNamespace(info=SimpleNamespace(assessments=[
        vote("summary_json_format", True, 1, version=2), vote("summary_faithful", True, 2)]))
    need = module.missing_votes([trace], ["summary_json_format", "summary_faithful"], versions, 3)
    assert list(need) == ["summary_faithful"]

"""lighthouse eval: specs, rubric criteria, combined review score, adapters (offline)."""
import io
import json
import logging
from pathlib import Path

import pytest

from lighthouse_poc import cli
from lighthouse_poc.config import load_profile
from lighthouse_poc.exceptions import ConfigurationError
from lighthouse_poc.gateway import Completion
from lighthouse_poc.prompt_eval import EvalRequest, build_runner, check_variables, load_records, run_eval
from lighthouse_poc.prompts import PromptResolver, ResolvedPrompt
from lighthouse_poc.review import BLOCKED, NEEDS_WORK, READY, compare, parse_value, review_case, summarize
from lighthouse_poc.rubric import Criterion, evaluate_criterion, load_spec
from lighthouse_poc.scorers import ERROR_KEY

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("test")
TRIAGE = load_spec("claim_intake_triage")
CLAIMS = load_spec("claim_settlement_review")


def triage_output(**overrides):
    base = {"request_type": "new_claim", "product_line": "dental", "requires_human_review": False,
            "missing_information": [], "reply_draft": "Thank you, we received your message.",
            "benefit_determination": "Not determined"}
    return {**base, **overrides}


EXPECT = {"expected_fields": {"request_type": "new_claim", "product_line": "dental",
                              "requires_human_review": False,
                              "benefit_determination": "Not determined"}}


def row_for(spec, outputs, inputs=None, expectations=EXPECT, judges=None):
    """Build a result_df-like row by running the real criteria (as the scorers do)."""
    row = {"trace_id": "tr-1"}
    for c in spec.hard_gates:
        row[f"gate_{c.name}/value"] = evaluate_criterion(c, inputs or {}, outputs, expectations).value >= 1
    for c in spec.rubric:
        row[f"rubric_{c.name}/value"] = evaluate_criterion(c, inputs or {}, outputs, expectations).value
    for kind, value in (judges or {}).items():
        row[f"judge_{kind}/value"] = value
    return row


# ---------------------------------------------------------------- specs
def test_shipped_specs_load_and_point_to_datasets():
    for spec in (TRIAGE, CLAIMS):
        assert spec.dataset_path.is_file() and spec.hard_gates and spec.rubric
        assert spec.scorer_version.startswith(f"spec:{spec.name}:")
    assert TRIAGE.judge_kinds() == ["guidelines", "expectations_guidelines", "safety"]
    assert CLAIMS.judge_kinds() == ["expectations_guidelines", "safety", "correctness"]


@pytest.mark.parametrize("change,message", [
    ({"adapter": "other"}, "adapter"),
    ({"hard_gates": []}, "hard gate"),
    ({"rubric": [{"name": "x", "type": "nope"}]}, "unknown criterion"),
    ({"rubric": [{"name": "x", "type": "regex_absent"}]}, "needs"),
    ({"rubric": [{"name": "x", "type": "regex_absent", "pattern": "("}]}, "invalid pattern"),
    ({"rubric": [{"name": "json_object", "type": "json_object"}]}, "unique"),
    ({"judges": {"vibes": True}}, "unknown judges"),
    ({"scoring": {"rubric_weight": 0, "judge_weight": 0}}, "weights"),
    ({"scoring": {"pass_threshold": 1.5}}, "pass_threshold"),
])
def test_invalid_specs_are_rejected(tmp_path, change, message):
    import yaml
    raw = yaml.safe_load((ROOT / "config" / "evals" / "claim_intake_triage.yaml").read_text())
    raw.update(change)
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_spec(path)


# ---------------------------------------------------------------- criteria
def test_expected_fields_gives_partial_credit():
    c = Criterion("e", "expected_fields")
    assert evaluate_criterion(c, {}, triage_output(), EXPECT).value == 1.0
    score = evaluate_criterion(c, {}, triage_output(product_line="life"), EXPECT)
    assert score.value == 0.75 and "product_line='life'" in score.rationale
    assert evaluate_criterion(c, {}, {ERROR_KEY: "X"}, EXPECT).value == 0.0
    assert evaluate_criterion(c, {}, triage_output(), {}).value == 1.0  # not applicable


def test_decision_and_amount_language_is_caught():
    by_name = {c.name: c for c in TRIAGE.rubric}
    bad = triage_output(reply_draft="Good news, your claim is approved and you will receive $5,000.")
    assert evaluate_criterion(by_name["reply_no_amount"], {}, bad, EXPECT).value == 0.0
    assert evaluate_criterion(by_name["reply_no_decision_language"], {}, bad, EXPECT).value == 0.0
    assert evaluate_criterion(by_name["reply_no_amount"], {}, triage_output(), EXPECT).value == 1.0


def test_identifier_echo_only_counts_identifiers_from_the_input():
    c = Criterion("id", "no_input_identifier_echo")
    inputs = {"claimant_message": "My SIN is 000-000-000, receipts attached."}
    assert evaluate_criterion(c, inputs, triage_output(reply_draft="Noted 000-000-000."), {}).value == 0
    assert evaluate_criterion(c, inputs, triage_output(), {}).value == 1
    assert "not applicable" in evaluate_criterion(c, {"claimant_message": "hi"}, triage_output(), {}).rationale


# ---------------------------------------------------------------- combined review score
@pytest.mark.parametrize("value,expected", [
    (True, 1.0), (False, 0.0), ("yes", 1.0), ("no", 0.0), ("0.5", 0.5), (0.25, 0.25),
    (None, None), (float("nan"), None), ("error", None), (3, 1.0)])
def test_parse_value(value, expected):
    assert parse_value(value) == expected


def test_combined_score_weights_rubric_and_judges():
    row = row_for(TRIAGE, triage_output(product_line="life"),
                  judges={"guidelines": "yes", "expectations_guidelines": "no", "safety": "yes"})
    case = review_case(row, TRIAGE, "C1")
    weights = sum(c.weight for c in TRIAGE.rubric)                    # 11
    rubric = (4 * 0.75 + (weights - 4)) / weights                      # expected_fields 3/4
    assert case.hard_pass and case.rubric_score == pytest.approx(rubric)
    assert case.judge_score == pytest.approx(2 / 3)
    assert case.combined == pytest.approx(0.6 * rubric + 0.4 * 2 / 3)
    assert any("judge no: expectations_guidelines" in n for n in case.notes)


def test_hard_gate_failure_zeroes_the_case_and_blocks_the_run():
    bad = triage_output(benefit_determination="Approved")
    case = review_case(row_for(TRIAGE, bad, judges={"safety": "yes"}), TRIAGE, "C1", ["safety"])
    assert not case.hard_pass and case.combined == 0.0
    good = review_case(row_for(TRIAGE, triage_output(), judges={"safety": "yes"}), TRIAGE, "C2",
                       ["safety"])
    assert summarize([case, good], TRIAGE).verdict == BLOCKED


def test_judge_errors_fall_back_to_rubric_and_are_counted():
    case = review_case(row_for(TRIAGE, triage_output(), judges={"safety": None}), TRIAGE, "C1",
                       ["safety"])
    assert case.judge_score is None and case.combined == case.rubric_score == 1.0
    assert summarize([case], TRIAGE).judge_errors == 1


def test_verdicts_follow_threshold():
    good = review_case(row_for(TRIAGE, triage_output()), TRIAGE, "A", [])
    weak = review_case(row_for(TRIAGE, triage_output(request_type="unclear", product_line="x",
                                                     requires_human_review=True)), TRIAGE, "B", [])
    assert summarize([good], TRIAGE).verdict == READY
    assert summarize([weak], TRIAGE).verdict == NEEDS_WORK
    review = summarize([good, weak], TRIAGE)
    assert review.metrics()["lighthouse/cases_below_threshold"] == 1.0
    assert "lighthouse/judge_score" not in review.metrics()      # no judges ran
    rows = compare(summarize([weak], TRIAGE), summarize([good], TRIAGE))
    assert rows == []                                            # different case IDs
    rows = compare(summarize([good], TRIAGE), summarize([good], TRIAGE))
    assert rows[0]["delta"] == 0.0


# ---------------------------------------------------------------- adapters and data
class Scripted:
    def __init__(self, text):
        self.text, self.messages = text, []

    def complete_messages(self, messages):
        self.messages.append(messages)
        return Completion(text=self.text, duration_ms=1, served_model="scripted")

    def complete(self, system, user):
        return self.complete_messages([{"role": "system", "content": system},
                                       {"role": "user", "content": user}])


def chat_prompt():
    return ResolvedPrompt("claim_intake_triage", "3", [
        {"role": "system", "content": "Classify."},
        {"role": "user", "content": "Claimant message:\n{{claimant_message}}"}], "registry", "3")


def test_chat_adapter_renders_variables_and_parses_fenced_json():
    model = Scripted("```json\n" + json.dumps(triage_output()) + "\n```")
    run = build_runner(TRIAGE, chat_prompt(), model, LOGGER, registry=None)
    assert run({"claimant_message": "root canal"})["request_type"] == "new_claim"
    assert model.messages[0][1]["content"] == "Claimant message:\nroot canal"


def test_chat_adapter_rejects_non_json_and_missing_variables():
    from lighthouse_poc.exceptions import OutputValidationError
    run = build_runner(TRIAGE, chat_prompt(), Scripted("I cannot help."), LOGGER, registry=None)
    with pytest.raises(OutputValidationError):
        run({"claimant_message": "x"})
    with pytest.raises(ConfigurationError, match="claimant_message"):
        check_variables(TRIAGE, chat_prompt(), [{"inputs": {"other": "x"}, "tags": {"case_id": "T1"}}])


def test_text_prompt_without_variables_sends_the_record_as_user_message():
    prompt = ResolvedPrompt("p", "1", "Instructions only.", "registry", "1")
    model = Scripted(json.dumps(triage_output()))
    build_runner(TRIAGE, prompt, model, LOGGER, registry=None)({"claimant_message": "hi"})
    system, user = model.messages[0]
    assert system["content"] == "Instructions only." and json.loads(user["content"]) == {
        "claimant_message": "hi"}


def test_load_records_from_spec_and_override(tmp_path):
    records, label = load_records(TRIAGE)
    assert len(records) == 11 and label == "golden.jsonl"
    other = tmp_path / "one.jsonl"
    other.write_text(json.dumps({"inputs": {"claimant_message": "x"}, "expectations": {}}) + "\n")
    assert len(load_records(TRIAGE, str(other))[0]) == 1


def test_qualify_prompt_name_for_unity_catalog():
    assert cli.qualify_prompt_name("workspace.s.claim_settlement_review", "claim_intake_triage") == (
        "workspace.s.claim_intake_triage")
    assert cli.qualify_prompt_name("claim_settlement_review", "claim_intake_triage") == "claim_intake_triage"
    assert cli.qualify_prompt_name("w.s.x", "c.s.explicit") == "c.s.explicit"


def test_chat_draft_files_and_template_display(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps(chat_prompt().template), encoding="utf-8")
    resolved = PromptResolver(None, "claim_intake_triage", tmp_path).resolve("file:chat.json")
    assert resolved.is_chat and resolved.variables == {"claimant_message"}
    assert json.loads(resolved.text) == chat_prompt().template
    (tmp_path / "bad.json").write_text('{"role": "user"}', encoding="utf-8")
    with pytest.raises(ConfigurationError):
        PromptResolver(None, "p", tmp_path).resolve("file:bad.json")


# ---------------------------------------------------------------- end to end (sqlite, mock)
@pytest.fixture
def tracing_on():
    import mlflow
    mlflow.tracing.enable()
    yield
    mlflow.tracing.disable()


def test_run_eval_end_to_end_with_mock_model(tracing_on, tmp_path):
    import mlflow
    mlflow.set_experiment("/lighthouse-eval-test")
    profile = load_profile("local", env={})
    records, label = load_records(CLAIMS)
    prompt = ResolvedPrompt("claim_settlement_review", "draft-test", "Summarise.", "file", "file:x")
    review, run_id = run_eval(profile, None, LOGGER, EvalRequest(
        spec=CLAIMS, prompt=prompt, model_alias="mock", records=records[:2],
        dataset_label=label, use_judges=False))
    assert review.verdict == READY and len(review.cases) == 2
    assert {c.case_id for c in review.cases} == {"CSET-STD-001", "CSET-LTD-002"}
    run = mlflow.get_run(run_id)
    assert run.data.tags["lighthouse.verdict"] == READY
    assert run.data.params["prompt_version"] == "draft-test"
    assert run.data.metrics["lighthouse/combined_score"] == 1.0


def test_eval_command_reports_and_exits_on_blocked(tracing_on, tmp_path, monkeypatch):
    import mlflow
    mlflow.set_experiment("/lighthouse-eval-cli-test")
    draft = tmp_path / "draft.txt"
    draft.write_text("Summarise.", encoding="utf-8")
    out = io.StringIO()
    ctx = cli.Context(load_profile("local", env={}), LOGGER, None, "claim_settlement_review", out)
    code = cli.main(["eval", "--prompt", f"file:{draft}", "--model", "mock", "--no-judges",
                     "--limit", "1", "--experiment", "/lighthouse-eval-cli-test"],
                    out=out, context_factory=lambda args, o: ctx)
    assert code == 0 and "READY FOR HUMAN REVIEW" in out.getvalue()
    # The mock answers triage prompts with claim JSON: required fields missing -> BLOCKED.
    out2 = io.StringIO()
    ctx2 = cli.Context(load_profile("local", env={}), LOGGER, None, "claim_settlement_review", out2)
    code = cli.main(["eval", "--spec", "claim_intake_triage", "--prompt", f"file:{draft}",
                     "--model", "mock", "--no-judges", "--limit", "1",
                     "--experiment", "/lighthouse-eval-cli-test"],
                    out=out2, context_factory=lambda args, o: ctx2)
    assert code == 1 and BLOCKED in out2.getvalue()


# ---------------------------------------------------------------- published prompt (@champion)
class AliasRegistry:
    def __init__(self, champion=None):
        self.champion = champion

    def load_alias(self, name, alias):
        from types import SimpleNamespace

        from lighthouse_poc.exceptions import PromptRegistryError
        if self.champion is None:
            raise PromptRegistryError("alias missing")
        return SimpleNamespace(version=self.champion, template="Published.", aliases=[alias],
                               commit_message="c", creation_timestamp=None, tags={})

    def load(self, name, version):
        from types import SimpleNamespace
        return SimpleNamespace(version=version, template="v", aliases=[], commit_message="c",
                               creation_timestamp=None, tags={})


def test_published_assistant_serves_champion_and_fails_closed():
    from lighthouse_poc.agent import published_assistant
    model = Scripted(json.dumps({"claim_id": "C", "product_segment": "dental", "summary": "s",
                                 "benefit_determination": "Not determined",
                                 **{k: [] for k in ("key_facts", "evidence_received",
                                                    "evidence_to_confirm", "inconsistencies",
                                                    "recommended_examiner_actions", "risk_flags")}}))
    assistant = published_assistant(AliasRegistry(champion=4), model, LOGGER, "p", "0.5.0")
    assistant.predict("C", "dental", "n", [], "c")
    assert model.messages[0][0]["content"] == "Published."
    with pytest.raises(ConfigurationError, match="lighthouse promote"):
        published_assistant(AliasRegistry(champion=None), model, LOGGER, "p", "0.5.0")


def test_try_defaults_to_the_published_version():
    args = cli.build_parser().parse_args(["try", "--case", "CSET-STD-001"])
    assert args.prompt == "@champion"


def _eval_with(registry, argv, tracing_on_fixture=None):
    import mlflow
    mlflow.set_experiment("/lighthouse-eval-baseline-test")
    out = io.StringIO()
    ctx = cli.Context(load_profile("local", env={}), LOGGER, registry, "claim_settlement_review", out)
    code = cli.main(["eval", "--model", "mock", "--no-judges", "--limit", "1",
                     "--experiment", "/lighthouse-eval-baseline-test", *argv],
                    out=out, context_factory=lambda args, o: ctx)
    return code, out.getvalue()


def test_optional_baseline_is_skipped_when_champion_is_missing(tracing_on, capsys):
    code, out = _eval_with(AliasRegistry(champion=None),
                           ["--prompt", "3", "--baseline", "@champion", "--baseline-optional"])
    assert code == 0 and "baseline" not in out.split("verdict")[-1].lower()
    assert "not available" in capsys.readouterr().err
    code, _ = _eval_with(AliasRegistry(champion=None), ["--prompt", "3", "--baseline", "@champion"])
    assert code == 2                                     # required baseline: configuration error


def test_baseline_equal_to_candidate_runs_once(tracing_on, capsys):
    code, out = _eval_with(AliasRegistry(champion=3),
                           ["--prompt", "3", "--baseline", "@champion", "--baseline-optional"])
    assert code == 0 and "evaluating it once" in capsys.readouterr().err
    assert "combined score change" not in out


def test_baseline_champion_is_compared(tracing_on):
    code, out = _eval_with(AliasRegistry(champion=2),
                           ["--prompt", "3", "--baseline", "@champion", "--scorer-mode", "deterministic"])
    assert code == 0 and "baseline @champion (2)" in out and "combined score change: +0.0" in out

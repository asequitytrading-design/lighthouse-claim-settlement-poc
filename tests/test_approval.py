"""Review gate: evidence checks, approval records and alias moves (offline, fakes)."""
import io
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lighthouse_poc import cli
from lighthouse_poc.approval import APPROVE, REJECT, ApprovalService, evidence_checks
from lighthouse_poc.config import load_profile
from lighthouse_poc.exceptions import ConfigurationError, PromptRegistryError
from lighthouse_poc.prompts import ResolvedPrompt
from lighthouse_poc.review import BLOCKED, NEEDS_WORK, READY
from lighthouse_poc.rubric import load_spec

LOGGER = logging.getLogger("test")
NAME = "claim_intake_triage"
SPEC = load_spec("claim_intake_triage")


def prompt(label="2", source="registry"):
    return ResolvedPrompt(NAME, label, "t", source, label)


def run(version="2", verdict=READY, gate_rate=1.0, status="FINISHED", kind="eval",
        scorer_version=None, run_id="run-2"):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, experiment_id="7", status=status),
        data=SimpleNamespace(
            params={"prompt_name": NAME, "prompt_version": version, "eval_spec": "claim_intake_triage",
                    "scorer_version": scorer_version or SPEC.scorer_version,
                    "model_service": "claims-small", "judge_model": "claims-judge",
                    "dataset": "golden.jsonl", "dataset_version": "1.0.0", "git_commit": "abc123"},
            tags={"lighthouse.kind": kind, "lighthouse.verdict": verdict},
            metrics={"lighthouse/hard_gate_pass_rate": gate_rate, "lighthouse/combined_score": 0.91}))


def failed(checks):
    return {c.name for c in checks if not c.passed}


# ---------------------------------------------------------------- evidence checks
def test_ready_run_of_the_same_version_passes_all_checks():
    assert failed(evidence_checks(prompt(), run(), SPEC.scorer_version, APPROVE)) == set()


@pytest.mark.parametrize("p,r,current,expected", [
    (prompt("draft-ab12cd34", "file"), run(), SPEC.scorer_version, {"registered_version", "same_prompt"}),
    (prompt(), run(version="1"), SPEC.scorer_version, {"same_prompt"}),
    (prompt(), run(verdict=NEEDS_WORK), SPEC.scorer_version, {"verdict_ready"}),
    (prompt(), run(verdict=BLOCKED, gate_rate=0.9), SPEC.scorer_version, {"verdict_ready", "hard_gates_all_pass"}),
    (prompt(), run(status="FAILED"), SPEC.scorer_version, {"run_finished"}),
    (prompt(), run(kind=None), SPEC.scorer_version, {"eval_run"}),
    (prompt(), run(scorer_version="spec:claim_intake_triage:00000000"), SPEC.scorer_version, {"spec_unchanged"}),
    (prompt(), run(), None, {"spec_unchanged"}),
])
def test_approval_is_refused_without_matching_evidence(p, r, current, expected):
    assert failed(evidence_checks(p, r, current, APPROVE)) == expected


def test_rejection_only_needs_the_run_to_match_the_version():
    assert failed(evidence_checks(prompt(), run(verdict=BLOCKED, gate_rate=0.0), None, REJECT)) == set()
    assert failed(evidence_checks(prompt(), run(version="1"), None, REJECT)) == {"same_prompt"}


# ---------------------------------------------------------------- records and writes
class FakeClient:
    def __init__(self, fail_alias=False, fail_log=False):
        self.texts, self.run_tags, self.version_tags, self.aliases = {}, {}, {}, {}
        self.fail_alias, self.fail_log = fail_alias, fail_log

    def get_run(self, run_id):
        raise RuntimeError("not found")

    def log_text(self, run_id, text, artifact_file):
        if self.fail_log:
            raise RuntimeError("store down")
        self.texts[(run_id, artifact_file)] = json.loads(text)

    def set_tag(self, run_id, key, value):
        self.run_tags[key] = value

    def set_prompt_version_tag(self, name, version, key, value):
        self.version_tags[(name, version, key)] = value

    def set_prompt_alias(self, name, alias, version):
        if self.fail_alias:
            raise RuntimeError("alias write failed")
        self.aliases[alias] = version


class FakeRegistry:
    def __init__(self, champion=None):
        self.champion = champion

    def load_alias(self, name, alias):
        if self.champion is None:
            raise PromptRegistryError("missing")
        return SimpleNamespace(version=self.champion)


def service(client, champion=None):
    return ApprovalService(LOGGER, FakeRegistry(champion), client,
                           clock=lambda: datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc),
                           user=lambda: "tester")


def test_tag_keys_are_valid_for_unity_catalog_prompt_versions():
    import re
    record = service(FakeClient()).build_record(prompt(), run(), APPROVE, "R",
                                                "A sufficiently long rationale.", "champion", [])
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", key) for key in record.summary_tags())


def test_approval_writes_record_tags_then_moves_alias():
    client = FakeClient()
    svc = service(client, champion=1)
    checks = evidence_checks(prompt(), run(), SPEC.scorer_version, APPROVE)
    record = svc.build_record(prompt(), run(), APPROVE, "A. Reviewer, claims SME",
                              "Read all judge rationales and the two weak cases.", "champion", checks)
    svc.write(record)
    stored = client.texts[("run-2", f"approvals/{record.record_id}.json")]
    assert stored["decision"] == "approved" and stored["previous_alias_version"] == "1"
    assert stored["model_service"] == "claims-small" and stored["combined_score"] == 0.91
    assert stored["decided_at"] == "2026-11-25T12:00:00+00:00" and stored["recorded_by"] == "tester"
    assert all(c["passed"] for c in stored["checks"])
    assert client.run_tags["lighthouse_approval_decision"] == "approved"
    assert client.version_tags[(NAME, "2", "lighthouse_approval_record_id")] == record.record_id
    assert client.aliases == {"champion": 2}


def test_rejection_records_without_alias_change():
    client = FakeClient()
    svc = service(client, champion=1)
    record = svc.build_record(prompt(), run(verdict=BLOCKED), REJECT, "A. Reviewer, claims SME",
                              "Blocked: decision language in two replies.", "champion", [])
    svc.write(record)
    assert record.alias is None and record.previous_alias_version is None
    assert client.aliases == {} and client.run_tags["lighthouse_approval_decision"] == "rejected"


def test_alias_failure_reports_that_the_record_exists():
    client = FakeClient(fail_alias=True)
    svc = service(client)
    record = svc.build_record(prompt(), run(), APPROVE, "R", "A sufficiently long rationale.", "champion", [])
    with pytest.raises(PromptRegistryError, match="recorded, but alias"):
        svc.write(record)
    assert client.texts  # evidence was written before the pointer


def test_record_write_failure_moves_nothing():
    client = FakeClient(fail_log=True)
    svc = service(client)
    record = svc.build_record(prompt(), run(), APPROVE, "R", "A sufficiently long rationale.", "champion", [])
    with pytest.raises(PromptRegistryError, match="approval record"):
        svc.write(record)
    assert client.aliases == {} and client.run_tags == {}


@pytest.mark.parametrize("reviewer,rationale,alias", [
    ("", "A sufficiently long rationale.", "champion"),
    ("R", "too short", "champion"),
    ("R", "A sufficiently long rationale.", "latest"),
    ("R", "A sufficiently long rationale.", "bad alias!"),
])
def test_record_inputs_are_validated(reviewer, rationale, alias):
    with pytest.raises(ConfigurationError):
        service(FakeClient()).build_record(prompt(), run(), APPROVE, reviewer, rationale, alias, [])


def test_unknown_run_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="not found"):
        service(FakeClient()).get_run("nope")


# ---------------------------------------------------------------- CLI
class CliRegistry(FakeRegistry):
    def load(self, name, version):
        return SimpleNamespace(version=version, template="t", aliases=[], commit_message="c",
                               creation_timestamp=None, tags={})


def run_promote(argv, evidence_run):
    client = FakeClient()
    client.get_run = lambda run_id: evidence_run
    out = io.StringIO()
    ctx = cli.Context(load_profile("local", env={}), LOGGER, CliRegistry(champion=1), NAME, out,
                      tracking_client=client)
    code = cli.main(["promote", *argv], out=out, context_factory=lambda args, o: ctx)
    return code, out.getvalue(), client


BASE = ["--run", "run-2", "--reviewer", "A. Reviewer, claims SME",
        "--rationale", "Reviewed judge rationales and failing cases."]


def test_cli_promote_approves_and_reports_alias_move():
    code, out, client = run_promote(["--prompt", "2", *BASE], run())
    assert code == 0 and "alias champion: 1 -> 2" in out and client.aliases == {"champion": 2}
    assert "not a claim decision" in out


def test_cli_promote_refuses_needs_work_and_writes_nothing():
    code, out, client = run_promote(["--prompt", "2", *BASE], run(verdict=NEEDS_WORK))
    assert code == 1 and "Refused" in out and "FAIL  verdict_ready" in out
    assert client.texts == {} and client.aliases == {}


def test_cli_promote_dry_run_writes_nothing():
    code, out, client = run_promote(["--prompt", "2", *BASE, "--dry-run"], run())
    assert code == 0 and "Dry run" in out and client.texts == {} and client.aliases == {}

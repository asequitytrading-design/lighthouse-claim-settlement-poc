"""Where evaluations and review decisions are recorded (ADR-016): MLflow assets, Unity Catalog."""
import io
import logging
from datetime import datetime
from types import SimpleNamespace

import pytest

from lighthouse_poc import cli
from lighthouse_poc.config import load_profile
from lighthouse_poc.experiment_assets import (DONE, SKIPPED, ExperimentAssets, dataset_name_for,
                                              registrable_judge_model)
from lighthouse_poc.platform.eval_records import eval_rows, release_rows
from lighthouse_poc.platform.migrations import SchemaMigrator
from lighthouse_poc.platform.naming import PlatformNames
from lighthouse_poc.platform.usage_recorder import PlatformWriter
from lighthouse_poc.prompt_eval import _covers
from lighthouse_poc.prompts import ResolvedPrompt
from lighthouse_poc.review import CaseReview, RunReview

LOG = logging.getLogger("test")
NAMES = PlatformNames("workspace", "dev_alex_")


class _Engine:
    def __init__(self):
        self.statements = []

    def connect(self):
        engine = self

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, statement):
                engine.statements.append(str(statement))
        return _Conn()


def _review():
    case = CaseReview("CSET-LIFE-003", "tr-1", gates={"gate_json_object": True},
                      rubric={"rubric_expected_fields": 0.5},
                      judges={"judge_safety": 1.0, "judge_correctness": None}, combined=0.7)
    return RunReview("claim_settlement_review", [case], 1.0, 0.5, 1.0, 0.7, 0.8, "NEEDS WORK", {}, 1)


MANIFEST = {"prompt_name": "workspace.dev_alex_lighthouse.claim_settlement_review",
            "prompt_version": "4", "model_service": "claims-medium", "judge_model": "claims-judge",
            "dataset": "golden_eval_dataset.jsonl", "dataset_version": "v1",
            "eval_spec": "claim_settlement_review", "scorer_version": "spec:x:1", "git_commit": "abc"}


def test_names_for_datasets_and_registrable_judges():
    assert dataset_name_for("workspace.s.claim_settlement_review", "claim_settlement_review") == \
        "workspace.s.claim_settlement_review_golden"
    assert dataset_name_for("claim_settlement_review", "spec") == "spec_golden"
    assert registrable_judge_model("openai:/system.ai.gpt-oss-120b", "system.ai.gpt-oss-120b") == \
        "databricks:/databricks-gpt-oss-120b"
    assert registrable_judge_model("databricks", None) == "databricks"
    assert registrable_judge_model("openai:/qwen3:4b", "qwen3:4b") is None     # Kong: not registrable


def test_eval_rows_cover_every_case_criterion_and_hold_no_text():
    started = datetime(2026, 11, 27, 10)
    rows = eval_rows(_review(), run_id="r1", experiment_id="e1", use_case_id="uc",
                     manifest=MANIFEST, model_service="system.ai.x", started_at=started,
                     finished_at=started)
    [run] = rows["eval_run"]
    assert run.verdict == "NEEDS WORK" and run.cases == 1 and run.judge_model_key == "claims-judge"
    kinds = {(r.kind, r.criterion): r for r in rows["eval_case_result"]}
    assert kinds[("gate", "gate_json_object")].passed is True
    assert kinds[("rubric", "rubric_expected_fields")].passed is False
    assert kinds[("judge", "judge_correctness")].value == "error" and kinds[("judge", "judge_correctness")].score is None
    assert kinds[("combined", "combined")].passed is False                 # 0.7 < 0.8


def test_release_rows_from_an_approval_record():
    record = SimpleNamespace(record_id="d1", prompt_name="p", prompt_version="4", decision="approved",
                             reviewer="A. Reviewer", rationale="Reviewed every judge rationale.",
                             run_id="r1", combined_score=0.9, verdict="READY FOR HUMAN REVIEW",
                             model_service="claims-medium", dataset_version="v1",
                             evaluated_git_commit="abc", decided_at="2026-11-27T10:00:00+00:00")
    [row] = release_rows(record, "uc", "system.ai.x")["release_decision"]
    assert row.decision_id == "d1" and row.eval_run_id == "r1" and row.decided_at.tzinfo is None


def test_platform_writer_inserts_rows_per_table(monkeypatch):
    head = SchemaMigrator(NAMES, LOG).heads()
    monkeypatch.setattr(SchemaMigrator, "current", lambda self: head)
    engine = _Engine()
    writer = PlatformWriter(load_profile("databricks", env={"DATABRICKS_HOST": "https://x"}), LOG,
                            engine_factory=lambda: engine)
    monkeypatch.setattr(writer, "_names", lambda: NAMES)
    started = datetime(2026, 11, 27, 10)
    result = writer.write(eval_rows(_review(), run_id="r1", experiment_id="e1", use_case_id="uc",
                                    manifest=MANIFEST, model_service=None, started_at=started,
                                    finished_at=started), "evaluation rows")
    assert result.status == "written" and result.written == 1 + 5
    assert "dev_alex_ai_observability.eval_run" in engine.statements[0]
    assert "eval_case_result" in engine.statements[1]
    monkeypatch.setattr(SchemaMigrator, "current", lambda self: [])
    assert writer.write({"eval_run": [object()]}, "rows").status == "skipped"


class _Client:
    def __init__(self, fail=None):
        self.fail, self.calls = fail, []

    def link_prompt_version_to_run(self, run_id, version):
        if self.fail:
            raise RuntimeError(self.fail)
        self.calls.append(("run", run_id))

    def link_prompt_versions_to_trace(self, versions, trace_id):
        self.calls.append(("trace", trace_id))


def test_prompt_link_skips_drafts_and_reports_unavailable_features(monkeypatch):
    import mlflow
    monkeypatch.setattr(mlflow.genai, "load_prompt", lambda uri, link_to_model=False: uri)
    registered = ResolvedPrompt("cat.s.p", "4", "t", "registry", "4")
    draft = ResolvedPrompt("cat.s.p", "draft-1", "t", "file", "file:x")
    client = _Client()
    done = ExperimentAssets(LOG, client).link_prompt("r1", registered, ["tr-1", "tr-2"])
    assert done.status == DONE and client.calls == [("run", "r1"), ("trace", "tr-1"), ("trace", "tr-2")]
    assert ExperimentAssets(LOG, client).link_prompt("r1", draft, ["tr-1"]).status == SKIPPED
    failed = ExperimentAssets(LOG, _Client(fail="FEATURE_DISABLED")).link_prompt("r1", registered, [])
    assert failed.status == SKIPPED and "not available here" in failed.detail   # never raises


def test_judge_registration_needs_a_databricks_model():
    result = ExperimentAssets(LOG, _Client()).register_judges("e1", [object()], None)
    assert result.status == SKIPPED and "databricks:/" in result.detail


def test_dataset_is_used_only_when_it_holds_exactly_the_records():
    import pandas as pd
    records = [{"inputs": {"case_id": "a"}}, {"inputs": {"case_id": "b"}}]
    dataset = SimpleNamespace(to_df=lambda: pd.DataFrame({"inputs": [r["inputs"] for r in records]}))
    assert _covers(dataset, records) and not _covers(dataset, records[:1]) and not _covers(None, records)


def test_cli_promote_reports_where_the_decision_was_recorded(monkeypatch):
    from lighthouse_poc import experiment_assets
    monkeypatch.setattr(experiment_assets, "run_trace_ids", lambda e, r: ["tr-1"])
    logged = []
    monkeypatch.setattr(ExperimentAssets, "human_review",
                        lambda self, traces, decision, reviewer, rationale, record_id:
                        logged.append((traces, decision)) or experiment_assets.AssetResult(
                            "human_review", DONE, f"{decision} on {len(traces)} traces"))
    from tests.test_approval import BASE, run, run_promote
    code, out, _ = run_promote(["--prompt", "2", *BASE], run())
    assert code == 0 and logged == [(["tr-1"], "approved")]
    assert "recorded  human_review" in out and "recorded  platform_records" in out
    assert "not a Databricks platform profile" in out          # local profile: no UC rows


# ---------------------------------------------------------------- traces in Unity Catalog (ADR-017)
from lighthouse_poc.tracking import activate_experiment, default_table_prefix  # noqa: E402


def test_databricks_profile_stores_traces_next_to_the_prompts():
    profile = load_profile("databricks", env={"DATABRICKS_HOST": "https://x",
                                              "LIGHTHOUSE_SCHEMA": "dev_alex_lighthouse"})
    assert profile.trace_storage.catalog == "workspace"
    assert profile.trace_storage.schema == "dev_alex_lighthouse" and profile.trace_storage.table_prefix is None
    assert load_profile("local", env={}).trace_storage is None
    assert default_table_prefix("/Workspace/Users/a@b.com/lighthouse-claim-settlement-poc-uc-dev") == \
        "lh_lighthouse_claim_settlement_poc_uc_dev"


def test_experiment_links_uc_trace_tables_or_falls_back(monkeypatch):
    import mlflow
    calls = []

    def fake_set_experiment(name, trace_location=None):
        calls.append(trace_location)
        if trace_location is not None and name.endswith("old"):
            raise RuntimeError("BAD_REQUEST: Experiment already contains traces.")
        return SimpleNamespace(experiment_id="1", name=name)

    monkeypatch.setattr(mlflow, "set_experiment", fake_set_experiment)
    profile = load_profile("databricks", env={"DATABRICKS_HOST": "https://x", "LIGHTHOUSE_SCHEMA": "s"})
    _, storage = activate_experiment(profile, LOG, "/Users/a/new")
    assert storage.in_unity_catalog and storage.location == "workspace.s.lh_new_*"
    assert calls[-1].table_prefix == "lh_new"
    _, storage = activate_experiment(profile, LOG, "/Users/a/old")
    assert not storage.in_unity_catalog and "already contains traces" in storage.detail
    assert calls[-1] is None                                      # activated without UC storage
    _, storage = activate_experiment(load_profile("local", env={}), LOG, "/x")
    assert storage.location == "mlflow experiment"

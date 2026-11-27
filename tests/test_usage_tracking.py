"""Usage attribution (ADR-013): every gateway call tagged, app calls recorded, judges tagged.
Offline: fake OpenAI clients, fake engines; no workspace."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from lighthouse_poc.config import ModelRoute, load_profile
from lighthouse_poc.exceptions import ModelInvocationError
from lighthouse_poc.gateway import REQUEST_TAGS_HEADER, GatewayChatClient, NoAuth, build_chat_model
from lighthouse_poc.notebook_support import model_service_env, run_cli
from lighthouse_poc.platform.gateway_usage import GatewayUsageSync
from lighthouse_poc.platform.migrations import SchemaMigrator
from lighthouse_poc.platform.naming import PlatformNames
from lighthouse_poc.platform.usage_recorder import (UsageRecorder, should_record, to_invocation,
                                                    tracked_run)
from lighthouse_poc.scorers import build_judge_binding
from lighthouse_poc.usage import (CallRecord, UsageContext, UsageTracker, activate, active_tracker,
                                  case_scope, context_from_profile)

LOG = logging.getLogger("test")
NAMES = PlatformNames("workspace", "dev_alex_")
ENV = {"DATABRICKS_HOST": "https://h", "UNITY_GATEWAY_SMALL_MODEL": "system.ai.s",
       "UNITY_GATEWAY_MEDIUM_MODEL": "system.ai.m", "UNITY_GATEWAY_JUDGE_MODEL": "system.ai.j"}


def _profile():
    return load_profile("databricks", env=ENV)


class _Completions:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("upstream 503")
        return SimpleNamespace(model="served-m", usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
                               choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": 1}'))])


def _client(kind="unity_gateway", fail=False, tracker=None):
    completions = _Completions(fail)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    route = ModelRoute(alias="claims-medium", upstream_model="system.ai.m")
    return GatewayChatClient("https://h/x", route, kind, NoAuth(), LOG, client=fake,
                             tracker=tracker), completions


@pytest.fixture
def databricks_auth(monkeypatch):
    """Stand-in for workspace auth (CI has no Databricks credentials)."""
    class _FakeAuth(NoAuth):
        def headers(self):
            return {"Authorization": "Bearer test-token"}
    fake = _FakeAuth()
    import lighthouse_poc.gateway as gateway
    import lighthouse_poc.scorers as scorers
    monkeypatch.setattr(gateway, "build_auth", lambda auth: fake)
    if hasattr(scorers, "build_auth"):
        monkeypatch.setattr(scorers, "build_auth", lambda auth: fake)
    return fake


def _tags(call):
    return json.loads(call["extra_headers"][REQUEST_TAGS_HEADER])


def test_profile_defaults_tag_every_call_even_without_a_run():
    client, completions = _client(tracker=UsageTracker(context_from_profile(_profile())))
    client.complete("s", "u")
    tags = _tags(completions.calls[0])
    assert tags["lh_use_case"] == "claim_settlement_assistant" and tags["lh_environment"] == "dev"
    assert tags["lh_activity"] == "app" and tags["lh_purpose"] == "app"
    assert tags["lh_model_key"] == "claims-medium" and tags["lh_event_id"]


def test_build_chat_model_carries_profile_defaults(databricks_auth):
    model = build_chat_model(_profile(), "claims-medium", LOG)
    assert model._tracker.context.use_case_id == "claim_settlement_assistant"


def test_active_run_and_case_scope_override_defaults_across_threads():
    client, completions = _client(tracker=UsageTracker(context_from_profile(_profile())))
    run = UsageTracker(context_from_profile(_profile(), activity="eval", prompt_version="3"), recording=True)
    run.update(run_id="r-1")

    def case(case_id):                      # MLflow evaluates cases in worker threads
        with case_scope(case_id=case_id):
            client.complete("s", "u")
    with activate(run):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(case, ["C1", "C2"]))
    assert active_tracker() is None
    tags = [_tags(c) for c in completions.calls]
    assert {t["lh_activity"] for t in tags} == {"eval"} and {t["lh_run_id"] for t in tags} == {"r-1"}
    records = run.drain()
    assert sorted(r.context.case_id for r in records) == ["C1", "C2"]
    assert all(r.input_tokens == 11 and r.output_tokens == 7 and r.served_model == "served-m" for r in records)
    assert {r.event_id for r in records} == {t["lh_event_id"] for t in tags}   # join key matches


def test_failed_calls_are_recorded_as_errors():
    run = UsageTracker(UsageContext(activity="eval"), recording=True)
    client, _ = _client(fail=True)
    with activate(run), pytest.raises(ModelInvocationError):
        client.complete("s", "u")
    [record] = run.drain()
    assert record.status == "error" and record.error_category


def test_callers_that_record_their_own_events_are_not_recorded_twice():
    run = UsageTracker(UsageContext(activity="demo"), recording=True)
    client, completions = _client()
    with activate(run):
        client.complete_messages([{"role": "user", "content": "x"}],
                                 tags={"lh_event_id": "own-1", "lh_purpose": "probe"})
    assert run.drain() == []
    assert _tags(completions.calls[0])["lh_event_id"] == "own-1"


def test_kong_gets_no_tag_header():
    client, completions = _client(kind="kong")
    client.complete("s", "u")
    assert REQUEST_TAGS_HEADER not in completions.calls[0]["extra_headers"]


def test_judge_calls_carry_the_run_tags(databricks_auth):
    binding = build_judge_binding(_profile(), "claims-judge")
    run = UsageTracker(context_from_profile(_profile(), activity="eval", prompt_version="4"))
    run.update(run_id="r-9")
    options = binding.scorer_options(run.judge_tags(binding.alias))
    tags = json.loads(options["extra_headers"][REQUEST_TAGS_HEADER])
    assert tags == {"lh_use_case": "claim_settlement_assistant", "lh_purpose": "judge",
                    "lh_environment": "dev", "lh_activity": "eval", "lh_prompt_version": "4",
                    "lh_run_id": "r-9", "lh_model_key": "claims-judge"}
    assert "Authorization" not in options["extra_headers"]          # no credential in headers


def test_recording_only_on_the_databricks_profile(monkeypatch):
    assert should_record(_profile())
    assert not should_record(load_profile("local"))
    monkeypatch.setenv("LIGHTHOUSE_RECORD_USAGE", "false")
    assert not should_record(_profile())


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

            def execute(self, statement, params=None):
                engine.statements.append(str(statement))
        return _Conn()


def test_recorder_writes_events_when_schemas_are_at_head(monkeypatch):
    head = SchemaMigrator(NAMES, LOG).heads()
    monkeypatch.setattr(SchemaMigrator, "current", lambda self: head)
    engine = _Engine()
    recorder = UsageRecorder(_profile(), LOG, engine_factory=lambda: engine)
    monkeypatch.setattr(recorder, "_names", lambda: NAMES)
    record = CallRecord("e1", UsageContext(use_case_id="uc", activity="eval", run_id="r", case_id="C1",
                                           purpose="app"), "claims-medium", "system.ai.m")
    result = recorder.flush([record])
    assert result.status == "written" and result.written == 1
    [insert] = engine.statements
    assert "model_invocation" in insert and "'eval'" in insert and "'C1'" in insert


def test_recorder_skips_and_explains_when_schemas_are_behind(monkeypatch):
    monkeypatch.setattr(SchemaMigrator, "current", lambda self: [])
    recorder = UsageRecorder(_profile(), LOG, engine_factory=_Engine)
    monkeypatch.setattr(recorder, "_names", lambda: NAMES)
    result = recorder.flush([CallRecord("e1", UsageContext(), "k", "s")])
    assert result.status == "skipped" and "lighthouse platform db upgrade" in result.detail


def test_recorder_failure_is_reported_not_raised(monkeypatch):
    def broken():
        raise RuntimeError("warehouse down")
    recorder = UsageRecorder(_profile(), LOG, engine_factory=broken)
    monkeypatch.setattr(recorder, "_names", lambda: NAMES)
    result = recorder.flush([CallRecord("e1", UsageContext(), "k", "s")])
    assert result.status == "error" and "warehouse down" in result.detail


def test_tracked_run_flushes_once_at_the_end():
    flushed = []
    recorder = SimpleNamespace(flush=lambda records: flushed.append(records) or SimpleNamespace(
        status="written", written=len(records), detail=""))
    client, _ = _client()
    with tracked_run(_profile(), LOG, "try", recorder=recorder, prompt_version="3") as tracker:
        client.complete("s", "u")
        client.complete("s", "u")
        assert flushed == []
    assert len(flushed) == 1 and len(flushed[0]) == 2 and tracker.last_flush.written == 2
    assert to_invocation(flushed[0][0]).activity == "try"


def test_gateway_usage_sync_is_an_incremental_idempotent_merge():
    sql = GatewayUsageSync(NAMES, LOG, lookback_hours=6).statement()
    assert sql.startswith("MERGE INTO workspace.dev_alex_ai_observability.gateway_usage t")
    assert "FROM system.ai_gateway.usage" in sql and "INTERVAL 6 HOURS" in sql
    assert "ON t.gateway_request_id = s.gateway_request_id" in sql and "WHEN MATCHED" not in sql
    for tag in ("lh_activity", "lh_run_id", "lh_model_key", "lh_event_id"):
        assert f"request_tags['{tag}']" in sql


def test_revision_0004_moves_views_to_the_copy():
    sql = SchemaMigrator(NAMES, LOG).offline_sql(start="0003_model_cost")
    assert "CREATE TABLE IF NOT EXISTS workspace.dev_alex_ai_observability.gateway_usage" in sql
    view = sql[sql.index("VIEW workspace.dev_alex_ai_observability.v_gateway_usage"):]
    assert "FROM workspace.dev_alex_ai_observability.gateway_usage" in view.split(";")[0]
    assert "coalesce(i.activity, g.activity, 'unknown') AS activity" in sql
    assert "ADD COLUMNS (activity STRING" in sql


def test_notebook_cli_helper_captures_output_and_json():
    def fake_main(argv, out):
        print(json.dumps({"argv": argv}), file=out)
        return 0
    result = run_cli(["eval", "--json"], main=fake_main, echo=False)
    assert result.command == "lighthouse --profile databricks eval --json"
    assert result.exit_code == 0 and result.data["argv"][-1] == "--json"
    assert model_service_env()["UNITY_GATEWAY_JUDGE_MODEL"] == "system.ai.gpt-oss-120b"

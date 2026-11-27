"""AI Marketplace data platform (ADR-012/013): naming, migrations, model catalog, knowledge,
gateway embeddings and the verification runner. Offline: no workspace, no model calls."""
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from lighthouse_poc.config import load_profile
from lighthouse_poc.exceptions import ConfigurationError, ModelInvocationError
from lighthouse_poc.gateway import (REQUEST_TAGS_HEADER, GatewayEmbeddingClient,
                                    build_embedding_model, request_tag_headers)
from lighthouse_poc.platform.knowledge import (Chunk, Document, KnowledgeStore,
                                               VectorIndexManager, chunk_document, load_documents)
from lighthouse_poc.platform.migrations import SchemaMigrator, render
from lighthouse_poc.platform.model_catalog import (ModelCatalogSync, approved_models,
                                                   load_model_catalog)
from lighthouse_poc.platform.naming import TABLE_SCHEMAS, PlatformNames, dev_prefix
from lighthouse_poc.platform.observability import EventWriter, ModelInvocation
from lighthouse_poc.platform.sql import float_array, literal
from lighthouse_poc.platform.verification import VIEWS, SetupVerifier

LOG = logging.getLogger("test")
NAMES = PlatformNames("workspace", "dev_alex_")


# ---------------------------------------------------------------- naming and SQL literals
def test_dev_prefix_from_user_name():
    assert dev_prefix("A.Lex-Smith@example.com") == "dev_a_lex_smith_"


@pytest.mark.parametrize("catalog, prefix", [("Workspace", ""), ("workspace", "dev"), ("ws;drop", "")])
def test_invalid_names_are_rejected(catalog, prefix):
    with pytest.raises(ConfigurationError):
        PlatformNames(catalog, prefix)


def test_tables_are_fully_qualified_in_their_schema():
    assert NAMES.table("model_catalog") == "workspace.dev_alex_ai_marketplace.model_catalog"
    assert NAMES.table("model_invocation").startswith("workspace.dev_alex_ai_observability.")
    assert NAMES.vector_index == "workspace.dev_alex_ai_knowledge.document_chunk_index"
    with pytest.raises(ConfigurationError):
        NAMES.table("nope")


def test_sql_literals_escape_quotes_and_backslashes():
    assert literal("it's a \\ test") == "'it\\'s a \\\\ test'"
    assert literal(None) == "NULL" and literal(True) == "TRUE" and literal(["a", "b"]) == "array('a', 'b')"
    assert float_array([0.5, 1]) == "CAST(array(0.5, 1.0) AS ARRAY<FLOAT>)"


# ---------------------------------------------------------------- migrations (offline)
def test_revisions_form_one_chain_with_one_head():
    migrator = SchemaMigrator(NAMES, LOG)
    revisions = migrator.revisions()
    assert revisions[0].down_revision is None
    for older, newer in zip(revisions, revisions[1:]):
        assert newer.down_revision == older.revision
    assert migrator.heads() == [revisions[-1].revision]


def test_offline_sql_creates_every_object_with_qualified_names():
    sql = SchemaMigrator(NAMES, LOG).offline_sql()
    for table, schema in TABLE_SCHEMAS.items():
        assert f"CREATE TABLE IF NOT EXISTS {NAMES.fq(schema(NAMES), table)}" in sql
    for view in VIEWS:
        assert re.search(rf"CREATE OR REPLACE VIEW workspace\.dev_alex_ai_\w+\.{view}\b", sql)
    assert "dev_alex_ai_marketplace.alembic_version" in sql      # version table location
    assert "{" not in sql.replace("{}", "")                       # every placeholder rendered
    assert "delta.columnMapping.mode" in sql and "delta.enableChangeDataFeed" in sql
    assert "ADD COLUMNS (served_model" in sql and "UPDATE workspace.dev_alex_ai_observability" in sql


def test_render_rejects_unknown_placeholders():
    assert render("SELECT * FROM {mkt}.t", NAMES) == "SELECT * FROM workspace.dev_alex_ai_marketplace.t"
    with pytest.raises(ConfigurationError):
        render("SELECT {unknown}", NAMES)


def test_online_actions_need_an_engine():
    with pytest.raises(ConfigurationError):
        SchemaMigrator(NAMES, LOG).current()


# ---------------------------------------------------------------- model catalog
def test_repo_model_catalog_is_valid_and_routes_every_role():
    entries = load_model_catalog()
    assert {e.key for e in approved_models(entries, "app")} >= {"claims-small", "claims-medium"}
    assert [e.key for e in approved_models(entries, "judge")] == ["claims-judge"]
    [embed] = approved_models(entries, "embedding")
    assert embed.embedding_dimensions == 1024 and embed.task == "embeddings"
    medium = next(e for e in entries if e.key == "claims-medium")
    judge = next(e for e in entries if e.key == "claims-judge")
    assert medium.family != judge.family                          # ADR-003


@pytest.mark.parametrize("entry, message", [
    ({"key": "x", "model_service": "system.ai.x", "task": "chat", "status": "approved", "roles": ["boss"]}, "roles"),
    ({"key": "x", "model_service": "x", "task": "chat", "status": "approved", "roles": ["app"]}, "catalog.schema.model"),
    ({"key": "x", "model_service": "system.ai.x", "task": "embeddings", "status": "approved", "roles": ["embedding"]}, "embedding_dimensions"),
    ({"key": "x", "model_service": "system.ai.x", "task": "chat", "status": "live", "roles": ["app"]}, "status"),
])
def test_invalid_catalog_entries_are_rejected(tmp_path, entry, message):
    path = tmp_path / "catalog.yaml"
    path.write_text(json.dumps({"models": [entry]}))
    with pytest.raises(ConfigurationError, match=message):
        load_model_catalog(path)


def test_catalog_sync_is_a_type_2_merge():
    entries = load_model_catalog()
    close_changed, close_removed, open_new = ModelCatalogSync(NAMES, LOG).statements(entries)
    table = NAMES.table("model_catalog")
    assert close_changed.startswith(f"MERGE INTO {table}") and "t.record_hash <> s.record_hash" in close_changed
    assert "is_current = false" in close_removed and "NOT IN ('claims-small'" in close_removed
    assert "LEFT ANTI JOIN" in open_new and "current_timestamp(), NULL, true" in open_new


def test_record_hash_changes_with_attributes_only():
    entry = load_model_catalog()[0]
    assert entry.record_hash() == load_model_catalog()[0].record_hash()
    assert entry.record_hash() != entry.__class__(**{**entry.__dict__, "status": "retired"}).record_hash()


# ---------------------------------------------------------------- knowledge
def test_repo_documents_are_synthetic_and_non_decisional():
    docs = load_documents()
    assert {d.product_line for d in docs} >= {"std", "ltd", "life", "accident", "critical_illness", "dental"}
    handoff = next(d for d in docs if d.doc_id == "kb-handoff-format")
    assert "Not determined" in handoff.text


def test_chunks_respect_the_size_limit_and_keep_all_text():
    doc = Document("d1", "t", "x", "all", "1", "\n\n".join(["word " * 60, "short", "word " * 200]))
    chunks = chunk_document(doc, max_chars=300)
    assert all(len(c.text) <= 300 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert "".join(c.text.replace("\n", "").replace(" ", "") for c in chunks) == \
        doc.text.replace("\n", "").replace(" ", "")


def test_chunk_merge_needs_one_vector_per_chunk():
    store = KnowledgeStore(NAMES, LOG)
    chunk = Chunk("d1-000", "d1", 0, "it's text")
    sql = store.chunk_merge([chunk], [[0.1, 0.2]], "embed-default", "system.ai.e")
    assert "MERGE INTO workspace.dev_alex_ai_knowledge.document_chunk" in sql
    assert "'it\\'s text'" in sql and "AS ARRAY<FLOAT>" in sql and "embedding_dimensions = 2" in sql
    with pytest.raises(Exception, match="1 chunks but 0"):
        store.chunk_merge([chunk], [], "embed-default", "system.ai.e")


class FakeVS:
    def __init__(self, endpoints):
        self.endpoints, self.created = endpoints, []

    def list_endpoints(self):
        return {"endpoints": [{"name": n} for n in self.endpoints]}

    def create_endpoint_and_wait(self, name, endpoint_type):
        self.created.append(name)


def test_free_edition_reuses_the_only_endpoint():
    assert VectorIndexManager(NAMES, LOG, FakeVS(["shared-vs"])).ensure_endpoint("lighthouse-vs") == "shared-vs"
    fake = FakeVS([])
    assert VectorIndexManager(NAMES, LOG, fake).ensure_endpoint("lighthouse-vs") == "lighthouse-vs"
    assert fake.created == ["lighthouse-vs"]


def test_index_wait_triggers_one_sync_and_waits_for_all_rows():
    states = iter([{"ready": False}, {"ready": True, "indexed_row_count": 0},
                   {"ready": True, "indexed_row_count": 5}, {"ready": True, "indexed_row_count": 13}])
    index = SimpleNamespace(syncs=0)
    index.describe = lambda: {"status": next(states)}
    index.sync = lambda: setattr(index, "syncs", index.syncs + 1)
    status = VectorIndexManager(NAMES, LOG).sync_and_wait(index, expected_rows=13, sleep=lambda s: None)
    assert status["indexed_row_count"] == 13 and index.syncs == 1


# ---------------------------------------------------------------- gateway embeddings + tags
def test_request_tags_go_only_to_unity_gateway():
    tags = {"lh_event_id": "e1", "lh_use_case": "uc", "empty": ""}
    header = request_tag_headers("unity_gateway", tags)
    assert json.loads(header[REQUEST_TAGS_HEADER]) == {"lh_event_id": "e1", "lh_use_case": "uc"}
    assert request_tag_headers("kong", tags) == {}


class FakeEmbeddings:
    def __init__(self, dims=3, count=None):
        self.dims, self.count, self.calls = dims, count, []

    def create(self, model, input, extra_headers):
        self.calls.append((model, input, extra_headers))
        n = len(input) if self.count is None else self.count
        data = [SimpleNamespace(index=i, embedding=[0.1] * self.dims) for i in reversed(range(n))]
        return SimpleNamespace(data=data, model=model, usage=SimpleNamespace(prompt_tokens=7))


def _embed_client(fake, kind="unity_gateway"):
    from lighthouse_poc.config import ModelRoute
    route = ModelRoute(alias="embed-default", upstream_model="system.ai.e", task="embeddings")
    return GatewayEmbeddingClient("http://x", route, kind, None, LOG,
                                  client=SimpleNamespace(embeddings=fake))


def test_embedding_client_orders_vectors_and_sends_tags():
    fake = FakeEmbeddings(dims=4)
    result = _embed_client(fake).embed(["a", "b"], tags={"lh_event_id": "e1"})
    assert result.dimensions == 4 and len(result.vectors) == 2 and result.prompt_tokens == 7
    assert REQUEST_TAGS_HEADER in fake.calls[0][2]


def test_embedding_count_mismatch_is_an_error():
    with pytest.raises(ModelInvocationError, match="1 embeddings for 2 inputs"):
        _embed_client(FakeEmbeddings(count=1)).embed(["a", "b"])


def test_profile_embedding_model_and_platform_section():
    profile = load_profile("databricks", env={"DATABRICKS_HOST": "https://h"})
    assert profile.platform.catalog == "workspace" and profile.platform.schema_prefix == ""
    assert profile.model("embed-default").task == "embeddings"
    with pytest.raises(ConfigurationError, match="not a gateway embeddings model"):
        build_embedding_model(profile, "claims-medium", LOG)


# ---------------------------------------------------------------- events and verification
def test_invocation_insert_has_no_text_fields_and_tags_identify_the_event():
    event = ModelInvocation(event_id="e1", status="ok", purpose="probe", model_key="k",
                            model_service="system.ai.k", use_case_id="uc", environment="dev")
    sql = EventWriter(NAMES, LOG).invocation_insert([event])
    assert sql.startswith("INSERT INTO workspace.dev_alex_ai_observability.model_invocation")
    assert "prompt_text" not in sql and "response" not in sql
    assert event.request_tags() == {"lh_event_id": "e1", "lh_purpose": "probe", "lh_model_key": "k",
                                    "lh_use_case": "uc", "lh_environment": "dev"}


def test_verifier_skips_dependent_steps_after_a_required_failure():
    profile = load_profile("databricks", env={"DATABRICKS_HOST": "https://h"})
    verifier = SetupVerifier(profile, NAMES, engine=None, logger=LOG)
    verifier.step_workspace = lambda: ("PASS", "ok")
    verifier.step_migrations = lambda migrate: ("FAIL", "boom")
    report = verifier.run(vectors=False)
    assert [c.status for c in report.checks] == ["PASS", "FAIL", "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", "SKIP"]
    assert not report.passed


def test_index_build_in_progress_means_wait_not_fail():
    states = iter([{"ready": True, "indexed_row_count": 0}, {"ready": True, "indexed_row_count": 0},
                   {"ready": True, "indexed_row_count": 13}])
    index = SimpleNamespace(attempts=0)
    index.describe = lambda: {"status": next(states)}

    def sync():
        index.attempts += 1
        if index.attempts == 1:
            raise Exception("Index is not ready to sync yet. Pipeline is in state RUNNING")
    index.sync = sync
    status = VectorIndexManager(NAMES, LOG).sync_and_wait(index, expected_rows=13, sleep=lambda s: None)
    assert status["indexed_row_count"] == 13 and index.attempts == 2


def test_new_index_needs_no_extra_sync():
    index = SimpleNamespace(describe=lambda: {"status": {"ready": True, "indexed_row_count": 13}})
    index.sync = lambda: (_ for _ in ()).throw(AssertionError("no sync for a new index"))
    VectorIndexManager(NAMES, LOG).sync_and_wait(index, expected_rows=13, sleep=lambda s: None,
                                                 require_sync=False)


# ---------------------------------------------------------------- rate card and cost (0003)
from lighthouse_poc.platform.dashboard import (GRID_WIDTH, DashboardPublisher,  # noqa: E402
                                               ObservabilityDashboard)
from lighthouse_poc.platform.pricing import RateCardSync, load_rate_card  # noqa: E402
from lighthouse_poc.platform.sample_traffic import SampleTrafficGenerator  # noqa: E402


def test_repo_rate_card_prices_every_routable_model():
    models = load_model_catalog()
    prices = load_rate_card(models=models)
    priced = {p.model_key for p in prices}
    routable = {m.key for m in models if m.status in ("approved", "candidate")}
    assert routable <= priced
    assert all(p.currency == "USD" and p.input_per_1m_tokens >= 0 for p in prices)


@pytest.mark.parametrize("rate, message", [
    ({"model_key": "nope", "effective_from": "2026-01-01", "input_per_1m_tokens": 1,
      "output_per_1m_tokens": 1, "price_source": "list"}, "not in config/model_catalog"),
    ({"model_key": "claims-small", "effective_from": "01/01/2026", "input_per_1m_tokens": 1,
      "output_per_1m_tokens": 1, "price_source": "list"}, "YYYY-MM-DD"),
    ({"model_key": "claims-small", "effective_from": "2026-01-01", "input_per_1m_tokens": -1,
      "output_per_1m_tokens": 1, "price_source": "list"}, ">= 0"),
    ({"model_key": "claims-small", "effective_from": "2026-01-01", "input_per_1m_tokens": 1,
      "output_per_1m_tokens": 1, "price_source": "guess"}, "price_source"),
])
def test_invalid_rates_are_rejected(tmp_path, rate, message):
    path = tmp_path / "pricing.yaml"
    path.write_text(json.dumps({"currency": "USD", "rates": [rate]}))
    with pytest.raises(ConfigurationError, match=message):
        load_rate_card(path, models=load_model_catalog())


def test_duplicate_rate_periods_are_rejected(tmp_path):
    rate = {"model_key": "claims-small", "effective_from": "2026-01-01", "input_per_1m_tokens": 1,
            "output_per_1m_tokens": 1, "price_source": "list"}
    path = tmp_path / "pricing.yaml"
    path.write_text(json.dumps({"currency": "USD", "rates": [rate, rate]}))
    with pytest.raises(ConfigurationError, match="Duplicate"):
        load_rate_card(path)


def test_rate_card_sync_is_one_mirroring_merge():
    sql = RateCardSync(NAMES, LOG).statement(load_rate_card())
    assert sql.startswith(f"MERGE INTO {NAMES.table('model_price')}")
    assert "t.effective_from = s.effective_from" in sql
    assert "WHEN NOT MATCHED BY SOURCE THEN DELETE" in sql
    assert "DATE '2026-01-01'" in sql and "CAST(0.07 AS DOUBLE)" in sql


def test_cost_revision_prices_calls_on_their_date():
    sql = SchemaMigrator(NAMES, LOG).offline_sql()
    assert "CREATE TABLE IF NOT EXISTS workspace.dev_alex_ai_marketplace.model_price" in sql
    view = sql[sql.index("VIEW workspace.dev_alex_ai_observability.v_model_call_cost"):]
    assert "FULL OUTER JOIN workspace.dev_alex_ai_observability.v_gateway_usage" in view
    assert "CAST(c.event_time AS DATE) >= p.effective_from" in view
    assert "input_tokens * p.input_usd_per_1m_tokens / 1e6 AS input_cost_usd" in view
    assert "prompt_text" not in view and "response" not in view.lower()


# ---------------------------------------------------------------- dashboard
def _widgets(dashboard):
    return [(page, item) for page in dashboard["pages"] for item in page["layout"]]


def test_dashboard_widgets_reference_existing_datasets_and_fields():
    built = ObservabilityDashboard(NAMES).as_dict()
    datasets = {d["name"]: "".join(d["queryLines"]) for d in built["datasets"]}
    assert set(datasets) == {"calls", "latency", "rate_card"}
    assert all("dev_alex_ai_" in sql for sql in datasets.values())
    for _, item in _widgets(built):
        widget = item["widget"]
        queries = {q["name"]: q["query"] for q in widget.get("queries", [])}
        for query in queries.values():
            assert query["datasetName"] in datasets
            for f in query["fields"]:
                column = f["expression"].split("`")[1]
                assert column in ("*", "associative_filter_predicate_group") or \
                    column in datasets[query["datasetName"]], (widget["name"], column)
        spec = widget.get("spec", {})
        names_in_queries = {f["name"] for q in queries.values() for f in q["fields"]}
        encodings = spec.get("encodings", {})
        for key, enc in encodings.items():
            for e in (enc if isinstance(enc, list) else [enc]):
                if key == "fields":
                    assert e["queryName"] in queries
                assert e["fieldName"] in names_in_queries, (widget["name"], key)


def test_dashboard_layout_fits_the_grid_without_overlaps():
    built = ObservabilityDashboard(NAMES).as_dict()
    for page in built["pages"]:
        cells = set()
        for item in page["layout"]:
            p = item["position"]
            assert p["x"] >= 0 and p["x"] + p["width"] <= GRID_WIDTH, item["widget"]["name"]
            area = {(x, y) for x in range(p["x"], p["x"] + p["width"])
                    for y in range(p["y"], p["y"] + p["height"])}
            assert not cells & area, f"overlap at {item['widget']['name']}"
            cells |= area


def test_dashboard_covers_usage_latency_tokens_and_cost_per_model():
    dashboard = ObservabilityDashboard(NAMES)
    fields = {r["fields"] for r in dashboard.widget_rows()}
    for measure in ("count(*)", "avg(latency_ms)", "sum(input_tokens)", "sum(output_tokens)",
                    "sum(input_cost_usd)", "sum(output_cost_usd)", "sum(total_cost_usd)"):
        assert any(measure in f for f in fields), measure
    pages = [p.display_name for p in dashboard.pages]
    assert pages == ["Overview", "Model drill-down"]
    assert any(r["type"] == "table" and "event_time" in r["fields"] and r["page"] == "Model drill-down"
               for r in dashboard.widget_rows())


class _FakeWorkspace:
    def __init__(self, existing_id=None):
        from databricks.sdk.errors import NotFound
        self.calls = []
        self.config = SimpleNamespace(host="https://example.cloud.databricks.com/")

        def get_status(path):
            if existing_id is None:
                raise NotFound("missing")
            return SimpleNamespace(resource_id=existing_id)
        self.workspace = SimpleNamespace(get_status=get_status,
                                         mkdirs=lambda p: self.calls.append(("mkdirs", p)))
        self.lakeview = SimpleNamespace(
            create=lambda dashboard: self.calls.append(("create", dashboard)) or SimpleNamespace(dashboard_id="new"),
            update=lambda i, dashboard: self.calls.append(("update", i)) or SimpleNamespace(dashboard_id=i),
            publish=lambda i, **kw: self.calls.append(("publish", i, kw)))


@pytest.mark.parametrize("existing, action", [(None, "create"), ("abc", "update")])
def test_dashboard_deploy_creates_once_then_updates(existing, action):
    w = _FakeWorkspace(existing)
    result = DashboardPublisher(w, LOG).deploy("Obs (dev)", "/Workspace/Users/x/lh/", "{}", "wh1")
    assert [c[0] for c in w.calls] == ["mkdirs", action, "publish"]
    assert result.created == (existing is None)
    assert result.path == "/Workspace/Users/x/lh/Obs (dev).lvdash.json"
    assert result.published_url.endswith(f"/dashboardsv3/{result.dashboard_id}/published")
    assert w.calls[-1][2] == {"embed_credentials": True, "warehouse_id": "wh1"}


# ---------------------------------------------------------------- sample traffic
class _Chat:
    def __init__(self, fail=False):
        self.fail = fail

    def complete_messages(self, messages, tags=None):
        assert tags["lh_use_case"] == "claim_settlement_demo" and tags["lh_event_id"]
        if self.fail:
            raise ModelInvocationError("busy", "rate_limit")
        return SimpleNamespace(text="Missing item named.", prompt_tokens=10, completion_tokens=5,
                               served_model="m")


class _Embed:
    def embed(self, texts, tags=None):
        return SimpleNamespace(prompt_tokens=3, dimensions=1024, served_model="e")


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


def test_sample_traffic_records_real_outcomes_including_failures():
    models = load_model_catalog()
    engine = _Engine()
    judge_fails = lambda m: _Chat(fail=m.key == "claims-judge")   # noqa: E731
    summary = SampleTrafficGenerator(models, NAMES, engine, LOG, chat_factory=judge_fails,
                                     embed_factory=lambda m: _Embed()).run(rounds=1)
    assert summary.calls == 18 and summary.errors == 6          # 6 cases x (app, judge, embed)
    [insert] = engine.statements
    assert insert.startswith(f"INSERT INTO {NAMES.table('model_invocation')}")
    assert "'rate_limit'" in insert and "Synthetic case" not in insert   # no prompt text stored


def test_sample_traffic_stops_on_configuration_errors():
    def broken(_):
        raise ConfigurationError("DATABRICKS_HOST missing")
    generator = SampleTrafficGenerator(load_model_catalog(), NAMES, _Engine(), LOG,
                                       chat_factory=broken, embed_factory=broken)
    with pytest.raises(ConfigurationError):
        generator.run(rounds=1)


def test_explorer_notebook_queries_only_platform_objects():
    text = (Path(__file__).resolve().parents[1] / "notebooks" / "03_explore_platform_data.py").read_text()
    referenced = set(re.findall(r"\{(?:MKT|OBS|KN)\}\.(\w+)", text))
    known = set(TABLE_SCHEMAS) | set(VIEWS) | {"alembic_version", "document_chunk_index"}
    assert referenced and referenced <= known, referenced - known
    assert "INSERT " not in text and "DELETE " not in text and "UPDATE " not in text


# ---------------------------------------------------------------- retrieval strategies
from lighthouse_poc.platform.retrieval import (KnowledgeCorpus, LlmRerankStrategy,  # noqa: E402
                                               SampleQuery, build_explorer, load_sample_queries,
                                               rank_matrix, recommend, route_products, summarise)

_CHUNKS = [("kb-life-evidence", 0, "life checklist death certificate"),
           ("kb-life-evidence", 1, "beneficiary identity document"),
           ("kb-dental-evidence", 0, "itemised statement procedure codes"),
           ("kb-handoff-format", 0, "benefit_determination Not determined")]


class _FakeIndex:
    """Returns chunks in a fixed order (dental first, to give strategies something to fix)."""

    def __init__(self):
        self.calls = []

    def similarity_search(self, columns, num_results, filters=None, **kw):
        self.calls.append({"filters": filters, **kw})
        rows = [c for c in [_CHUNKS[2], _CHUNKS[1], _CHUNKS[0], _CHUNKS[3]]
                if not filters or c[0] in filters["doc_id"]][:num_results]
        return {"manifest": {"columns": [{"name": n} for n in columns + ["score"]]},
                "result": {"data_array": [[f"{d}-{i:03d}", d, i, t, 0.9 - n / 10]
                                          for n, (d, i, t) in enumerate(rows)]}}


class _QEmbed:
    def __init__(self):
        self.batches = []

    def embed(self, texts, tags=None):
        self.batches.append((list(texts), tags))
        return SimpleNamespace(vectors=[[0.1] * 4 for _ in texts], prompt_tokens=len(texts),
                               served_model="e", dimensions=4)


class _Reranker:
    def __init__(self, answer):
        self.answer = answer

    def complete_messages(self, messages, tags=None):
        assert tags["lh_purpose"] == "rerank" and tags["lh_use_case"] == "knowledge_search_explorer"
        return SimpleNamespace(text=self.answer, prompt_tokens=50, completion_tokens=5, served_model="m")


def _explorer(answer='{"order": [2, 1, 3]}'):
    models = {m.key: m for m in load_model_catalog()}
    corpus = KnowledgeCorpus.from_rows(
        [{"doc_id": "kb-life-evidence", "product_line": "life", "title": "Life"},
         {"doc_id": "kb-dental-evidence", "product_line": "dental", "title": "Dental"},
         {"doc_id": "kb-handoff-format", "product_line": "all", "title": "Handoff"}],
        [{"doc_id": d, "chunk_index": i, "text": t} for d, i, t in _CHUNKS])
    embed, index = _QEmbed(), _FakeIndex()
    explorer = build_explorer(index, NAMES.vector_index, corpus, models["embed-default"], embed,
                              models["claims-small"], _Reranker(answer), LOG)
    return explorer, embed, index


def test_repo_sample_queries_name_existing_documents():
    queries = load_sample_queries()
    docs = {d.doc_id for d in load_documents()}
    assert 10 <= len(queries) <= 15
    assert {q.expected_doc_id for q in queries} == docs          # every document is asked about
    assert {q.style for q in queries} >= {"keyword", "paraphrase", "exact_term"}


def test_product_router():
    assert route_products("Treating records stop before the elimination period ends") == {"ltd"}
    assert route_products("Beneficiary without identity") == {"life"}
    assert route_products("How should the handoff be structured?") == set()


def test_strategies_share_one_embedding_call_and_score_by_document():
    explorer, embed, index = _explorer()
    query = SampleQuery("q", "beneficiary has no identity document", "kb-life-evidence")
    results = explorer.run(["dense", "hybrid", "filtered_hybrid", "parent_document", "llm_rerank"],
                           [query], k=3)
    assert len(embed.batches) == 1 and embed.batches[0][1]["lh_purpose"] == "embedding_query"
    by = {r.strategy: r for r in results}
    assert by["dense"].doc_ranking[0] == "kb-dental-evidence"           # the planted miss
    assert by["filtered_hybrid"].doc_ranking[0] == "kb-life-evidence"    # router filter fixes it
    assert index.calls[2]["filters"] == {"doc_id": ["kb-handoff-format", "kb-life-evidence"]}
    assert index.calls[1]["query_type"] == "HYBRID" and index.calls[1]["query_text"]
    assert by["parent_document"].hits[1].text == "life checklist death certificate\n\nbeneficiary identity document"
    assert by["llm_rerank"].doc_ranking[0] == "kb-life-evidence" and by["llm_rerank"].model_calls == 1
    summaries = {s.strategy: s for s in summarise(results)}
    assert summaries["dense"].mrr == 0.5 and summaries["filtered_hybrid"].mrr == 1.0
    assert recommend(list(summaries.values())).strategy == "filtered_hybrid"  # tie: no extra calls wins
    assert recommend(list(summaries.values()), allow_extra_calls=False).strategy == "filtered_hybrid"
    assert rank_matrix(results)[0]["dense"] == 2


def test_rerank_answer_parsing_and_fallback():
    assert LlmRerankStrategy.parse_order('ok {"order": [3, 1]}', 3) == [3, 1, 2]
    assert LlmRerankStrategy.parse_order('{"order": [9, "2"]}', 3) == [2, 1, 3]
    assert LlmRerankStrategy.parse_order("no json", 3) is None
    explorer, _, _ = _explorer(answer="I cannot rank these")
    [result] = explorer.run(["llm_rerank"], [SampleQuery("q", "x", "kb-life-evidence")], k=3)
    assert result.doc_ranking[0] == "kb-dental-evidence" and "hybrid order kept" in result.note


def test_explorer_events_carry_no_query_text():
    explorer, _, _ = _explorer()
    explorer.run(["dense", "llm_rerank"], [SampleQuery("q", "secret phrasing", "kb-life-evidence")], k=3)
    engine = _Engine()
    written = explorer.flush_events(NAMES, engine, LOG)
    assert written == 1 + 1 + 2                 # embedding batch + rerank call + 2 index queries
    assert all("secret phrasing" not in s for s in engine.statements)
    assert explorer.backend.queries == [] and explorer.encoder.events == []


def test_relevance_gate_uses_calibrated_dense_thresholds():
    from lighthouse_poc.platform.retrieval import RelevanceGate
    gate = RelevanceGate()
    assert gate.verdict(0.42) == "NO RELEVANT KNOWLEDGE"
    assert gate.verdict(0.53) == "BORDERLINE"
    assert gate.verdict(0.61) == "RELEVANT"
    assert gate.verdict(None) == "no results"


def test_single_query_report_checks_samples_and_free_text():
    from lighthouse_poc.platform.retrieval import run_single
    explorer, _, _ = _explorer()
    samples = [SampleQuery("q06", "beneficiary has no identity document", "kb-life-evidence")]
    report = run_single(explorer, " beneficiary has no identity document ", "filtered_hybrid", 3, samples)
    assert report.query.query_id == "q06" and report.routed_products == ["life"]
    assert report.expected_check() == "expected kb-life-evidence: rank 1"
    assert report.relevance == "RELEVANT"          # fake index scores the top hit 0.9
    free = run_single(explorer, "best pizza in Rome", "dense", 3, samples)
    assert free.query.query_id == "custom" and "free-text" in free.expected_check()
    with pytest.raises(ConfigurationError):
        run_single(explorer, "   ", "dense")


# ---------------------------------------------------------------- grounded answers
from lighthouse_poc.platform.grounding import (NO_INFORMATION, GroundedAnswerer,  # noqa: E402
                                               RetrievalConfidence, ground, verify_quotes)
from lighthouse_poc.platform.retrieval import Hit  # noqa: E402

_PASSAGES = [Hit(1, "kb-life-evidence", "kb-life-evidence-001", 1, 0.9,
                 "Each named beneficiary must provide a proof of identity document."),
             Hit(2, "kb-dental-evidence", "kb-dental-evidence-000", 0, 0.8,
                 "The itemised statement lists procedure codes and treatment dates.")]


class _Grounder:
    def __init__(self, reply, error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def complete_messages(self, messages, tags=None):
        self.calls += 1
        assert tags["lh_purpose"] == "grounded_answer" and "Passages:" in messages[1]["content"]
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.reply, prompt_tokens=80, completion_tokens=20, served_model="m")


def _answerer(reply, error=None):
    client = _Grounder(reply, error)
    return GroundedAnswerer(load_model_catalog()[0], client, LOG), client


def _dense(life=0.66, dental=0.55):
    return [Hit(1, "kb-life-evidence", "kb-life-evidence-001", 1, life, ""),
            Hit(2, "kb-dental-evidence", "kb-dental-evidence-000", 0, dental, "")]


def test_retrieval_confidence_is_zero_below_the_floor():
    c = RetrievalConfidence()
    assert c.score(0.417) == 0.0 and c.score(0.50) == 0.0 and c.score(None) == 0.0
    assert c.score(0.60) == 0.5 and c.score(0.80) == 1.0


def test_off_topic_question_is_refused_without_a_model_call():
    answerer, client = _answerer('{"answerable": true}')
    result = answerer.answer("best pizza in Rome", _PASSAGES, _dense(life=0.42, dental=0.40))
    assert result.answer == NO_INFORMATION and result.grounding_score == 0.0
    assert client.calls == 0 and not result.model_called and "similarity floor" in result.refusal_reason


def test_model_says_passages_do_not_answer():
    answerer, client = _answerer('{"answerable": false}')
    result = answerer.answer("pet insurance for a dog", _PASSAGES, _dense())
    assert not result.grounded and result.grounding_score == 0.0 and client.calls == 1
    assert "do not contain" in result.refusal_reason and len(answerer.events) == 1


def test_grounded_answer_needs_verbatim_quotes():
    good = json.dumps({"answerable": True, "answer": "Each beneficiary needs proof of identity.",
                       "citations": [1], "quotes": ["each named beneficiary must provide a proof of "
                                                    "identity  document"]})
    answerer, _ = _answerer(good)
    result = answerer.answer("beneficiary without ID", _PASSAGES, _dense())
    assert result.grounded and result.cited_chunks == ["kb-life-evidence-001"]
    assert result.grounding_score == 0.8                     # (0.66 - 0.50) / 0.20, all quotes verified
    invented = json.dumps({"answerable": True, "answer": "x", "citations": [1],
                           "quotes": ["Beneficiaries must be paid within 30 days of the claim."]})
    answerer, _ = _answerer(invented)
    result = answerer.answer("beneficiary without ID", _PASSAGES, _dense())
    assert result.answer == NO_INFORMATION and "verification" in result.refusal_reason
    assert result.unverified_quotes


def test_partly_verified_quotes_lower_the_score_and_errors_fail_closed():
    half = json.dumps({"answerable": True, "answer": "x", "citations": [1, 2],
                       "quotes": ["The itemised statement lists procedure codes", "made up support sentence"]})
    answerer, _ = _answerer(half)
    result = answerer.answer("dental codes", _PASSAGES, _dense())
    assert result.grounded and result.grounding_score == 0.4        # best cited 0.8 x 1/2
    answerer, _ = _answerer("", error=ModelInvocationError("down", "rate_limit"))
    result = answerer.answer("dental codes", _PASSAGES, _dense())
    assert result.answer == NO_INFORMATION and "rate_limit" in result.refusal_reason
    assert answerer.events[0].status == "error"
    assert verify_quotes(["short"], [1], _PASSAGES) == ([], ["short"])     # too short to count


def test_ground_uses_single_query_report_and_parent_documents():
    from lighthouse_poc.platform.retrieval import run_single
    explorer, _, _ = _explorer()
    reply = json.dumps({"answerable": True, "answer": "ID document per beneficiary.", "citations": [2],
                        "quotes": ["beneficiary identity document"]})
    answerer, _ = _answerer(reply)
    explorer.event_sources.append(answerer)
    report = run_single(explorer, "beneficiary has no identity document", "parent_document", 3)
    assert len(report.dense_hits) == 4
    result = ground(answerer, report)
    assert result.grounded and result.cited_chunks == ["kb-life-evidence (document)"]
    assert result.grounding_score == 1.0     # whole document scored by its best chunk (0.8 in the fake)
    engine = _Engine()
    assert explorer.flush_events(NAMES, engine, LOG) >= 2 and answerer.events == []

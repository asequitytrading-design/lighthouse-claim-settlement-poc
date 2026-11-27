"""AI Marketplace data platform (ADR-012/013): naming, migrations, model catalog, knowledge,
gateway embeddings and the verification runner. Offline: no workspace, no model calls."""
import json
import logging
import re
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
    assert event.request_tags() == {"lh_event_id": "e1", "lh_purpose": "probe",
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

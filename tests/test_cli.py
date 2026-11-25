"""Developer CLI: prompt references, list/show/diff and single-case try (offline)."""
import io
import json
import logging
from types import SimpleNamespace

import pytest

from lighthouse_poc import cli
from lighthouse_poc.config import load_profile
from lighthouse_poc.exceptions import ConfigurationError, PromptRegistryError
from lighthouse_poc.gateway import Completion
from lighthouse_poc.prompts import PromptRef, PromptResolver, draft_label

LOGGER = logging.getLogger("test")
NAME = "claim_settlement_review"
TEMPLATES = {1: "Summarise facts. Never decide coverage.",
             2: "Summarise facts. List missing evidence. Never decide coverage."}


def version(number, aliases=(), note=None):
    return SimpleNamespace(version=str(number), template=TEMPLATES[number],
                           aliases=list(aliases), commit_message=note or f"v{number}",
                           creation_timestamp=1764000000000 + number, tags={})


class FakeRegistry:
    def __init__(self, aliases=None):
        self.aliases = aliases or {"champion": 1}

    def versions(self, name):
        champions = {v: [a for a, n in self.aliases.items() if n == v] for v in TEMPLATES}
        return [version(n, champions[n]) for n in sorted(TEMPLATES, reverse=True)]

    def latest(self, name):
        return version(max(TEMPLATES))

    def load(self, name, number):
        if int(number) not in TEMPLATES:
            raise PromptRegistryError(f"Could not load {name} version {number}")
        return version(int(number))

    def load_alias(self, name, alias):
        if alias not in self.aliases:
            raise PromptRegistryError(f"Could not load {name}@{alias}")
        return version(self.aliases[alias], [alias])


def context(out, registry=None):
    return cli.Context(load_profile("local", env={}), LOGGER, registry or FakeRegistry(),
                       NAME, out)


def run(argv, registry=None):
    out = io.StringIO()
    code = cli.main(argv, out=out, context_factory=lambda args, o: context(o, registry))
    return code, out.getvalue()


# ---------------------------------------------------------------- references
@pytest.mark.parametrize("text,kind,value", [
    ("3", "version", "3"), ("03", "version", "3"), ("@champion", "alias", "champion"),
    ("latest", "latest", "latest"), ("file:prompts/d.txt", "file", "prompts/d.txt")])
def test_prompt_ref_parse(text, kind, value):
    assert PromptRef.parse(text) == PromptRef(kind, value)


@pytest.mark.parametrize("text", ["", "0", "-1", "v3", "@", "@1x", "file:", "champion"])
def test_prompt_ref_rejects_ambiguous_input(text):
    with pytest.raises(ConfigurationError):
        PromptRef.parse(text)


def test_resolver_labels_drafts_by_content(tmp_path):
    draft = tmp_path / "draft.txt"
    draft.write_text("Draft prompt.", encoding="utf-8")
    resolved = PromptResolver(FakeRegistry(), NAME, tmp_path).resolve("file:draft.txt")
    assert resolved.is_draft and resolved.label == draft_label("Draft prompt.")
    assert resolved.label.startswith("draft-") and resolved.format() == "Draft prompt."
    draft.write_text("Draft prompt, edited.", encoding="utf-8")
    assert PromptResolver(FakeRegistry(), NAME, tmp_path).resolve(
        "file:draft.txt").label != resolved.label


def test_resolver_registry_references():
    resolver = PromptResolver(FakeRegistry(), NAME)
    assert resolver.resolve("@champion").label == "1"
    assert resolver.resolve("@champion").reference == "@champion"
    assert resolver.resolve("latest").label == "2"
    assert resolver.resolve("2").template == TEMPLATES[2]


def test_resolver_rejects_missing_or_empty_drafts(tmp_path):
    (tmp_path / "empty.txt").write_text("  \n", encoding="utf-8")
    resolver = PromptResolver(FakeRegistry(), NAME, tmp_path)
    for ref in ("file:missing.txt", "file:empty.txt"):
        with pytest.raises(ConfigurationError):
            resolver.resolve(ref)


# ---------------------------------------------------------------- prompt commands
def test_prompt_list_marks_latest_and_aliases():
    code, out = run(["prompt", "list"])
    rows = out.splitlines()[2:]
    assert code == 0 and rows[0].split()[:1] == ["2"] and "(latest)" in rows[0]
    assert "champion" in rows[1] and "v1" in rows[1]


def test_timestamps_from_mlflow_server_and_unity_catalog():
    from google.protobuf.timestamp_pb2 import Timestamp
    assert cli._when(1764000000000) == "2025-11-24 16:00"
    assert cli._when(Timestamp(seconds=1764000000)) == "2025-11-24 16:00"
    assert cli._when(None) == cli._when(0) == cli._when(Timestamp()) == "-"


def test_prompt_list_without_versions_points_to_drafts():
    registry = FakeRegistry()
    registry.versions = lambda name: []
    code, out = run(["prompt", "list"], registry)
    assert code == 0 and "file:<path>" in out


def test_prompt_show_template_only_is_exact():
    code, out = run(["prompt", "show", "@champion", "--template-only"])
    assert code == 0 and out == TEMPLATES[1]


def test_prompt_diff_is_by_sentence():
    code, out = run(["prompt", "diff", "1", "2"])
    assert code == 0
    assert "+List missing evidence." in out.splitlines()
    assert not any(line.startswith("-") and "Never decide" in line for line in out.splitlines())


def test_prompt_diff_reports_identical(tmp_path):
    draft = tmp_path / "same.txt"
    draft.write_text(TEMPLATES[2], encoding="utf-8")
    code, out = run(["prompt", "diff", "latest", f"file:{draft}"])
    assert code == 0 and "No differences" in out


def test_unknown_alias_is_a_clean_error(capsys):
    code, _ = run(["prompt", "show", "@missing"])
    assert code == 2 and "missing" in capsys.readouterr().err


def test_common_options_work_after_the_subcommand():
    args = cli.build_parser().parse_args(["try", "--prompt", "1", "--case", "C",
                                          "--profile", "databricks", "-v"])
    assert args.profile == "databricks" and args.verbose
    args = cli.build_parser().parse_args(["--profile", "databricks", "prompt", "list"])
    assert args.profile == "databricks" and args.verbose is False


# ---------------------------------------------------------------- try
RECORD = {"inputs": {"case_id": "C-1", "product_segment": "dental", "claim_narrative": "n",
                     "submitted_documents": ["claim form"], "policy_or_benefit_context": "c"},
          "expectations": {"expected_fields": {"product_segment": "dental",
                                               "benefit_determination": "Not determined"}}}


class ScriptedModel:
    def __init__(self, payload):
        self.payload, self.system_prompts = payload, []

    def complete(self, system_prompt, user_message):
        self.system_prompts.append(system_prompt)
        return Completion(text=self.payload, duration_ms=1, served_model="scripted")


def output(**overrides):
    base = {key: [] for key in ("key_facts", "evidence_received", "evidence_to_confirm",
                                "inconsistencies", "recommended_examiner_actions", "risk_flags")}
    base.update(claim_id="C-1", product_segment="dental", summary="s",
                benefit_determination="Not determined")
    return json.dumps({**base, **overrides})


def test_try_case_uses_draft_text_and_passes_checks(tmp_path):
    draft = tmp_path / "d.txt"
    draft.write_text("Draft system prompt.", encoding="utf-8")
    ctx = context(io.StringIO())
    prompt = ctx.resolver.resolve(f"file:{draft}")
    model = ScriptedModel(output())
    result = cli.try_case(ctx, prompt, RECORD, "claims-small", lambda *a: model)
    assert model.system_prompts == ["Draft system prompt."]
    assert result["passed"] and result["prompt_version"] == prompt.label
    assert result["served_models"] == ["scripted"] and result["model"] == "claims-small"


def test_try_case_reports_contract_failures_without_raising():
    ctx = context(io.StringIO())
    prompt = ctx.resolver.resolve("1")
    result = cli.try_case(ctx, prompt, RECORD, "claims-small",
                          lambda *a: ScriptedModel(output(benefit_determination="Approved")))
    assert not result["passed"]
    assert result["checks"] == {"output_schema_valid": False, "non_decision_boundary": False,
                                "expected_fields_match": False}


def test_try_command_with_mock_model(tmp_path):
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text(json.dumps(RECORD) + "\n", encoding="utf-8")
    code, out = run(["try", "--prompt", "@champion", "--case", "C-1", "--model", "mock",
                     "--dataset", str(dataset), "--experiment", "/cli-test", "--json"])
    result = json.loads(out)
    assert code == 0 and result["passed"] and result["prompt_version"] == "1"


def test_try_unknown_case_lists_available(tmp_path, capsys):
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text(json.dumps(RECORD) + "\n", encoding="utf-8")
    code, _ = run(["try", "--prompt", "1", "--case", "NOPE", "--model", "mock",
                   "--dataset", str(dataset)])
    assert code == 2 and "C-1" in capsys.readouterr().err


def test_cases_lists_golden_ids():
    code, out = run(["cases"])
    assert code == 0 and len(out.splitlines()) == 6 and out.startswith("CSET-")

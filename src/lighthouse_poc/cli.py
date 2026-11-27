"""lighthouse: developer CLI for prompt versions, single-case trials and golden evaluations.

The same commands work locally (Ollama via Kong, local MLflow) and against Databricks
(Unity Gateway, Unity Catalog prompts) by switching --profile. Prompt references:
  3 | @champion | latest | file:prompts/draft.txt

  lighthouse prompt list
  lighthouse prompt show @champion
  lighthouse prompt show 3 --template-only > prompts/draft.txt     # start a draft
  lighthouse prompt diff 3 file:prompts/draft.txt
  lighthouse cases
  lighthouse try --prompt file:prompts/draft.txt --case CSET-LTD-002 --model claims-small
  lighthouse --profile databricks try --prompt latest --case CSET-STD-001 --model claims-medium
  lighthouse eval --prompt file:prompts/draft.txt --baseline @champion --model claims-medium
  lighthouse eval --spec claim_intake_triage --prompt @candidate --model claims-small
  lighthouse promote --prompt 2 --run <eval run id> --reviewer "Name, role" --rationale "..."
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import logging
import os
import re
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep CLI output readable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TextIO

from . import __version__
from .config import Profile, apply_env_overrides, load_dotenv, require
from .exceptions import LighthouseError
from .approval import TAG
from .prompts import REF_HELP, PromptResolver, ResolvedPrompt

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------- context
@dataclass
class Context:
    """Everything a command needs; built once by bootstrap, injectable in tests."""
    profile: Profile
    logger: logging.Logger
    registry: object
    prompt_name: str
    out: TextIO
    tracking_client: object = None  # injectable MlflowClient (tests); default: real client

    @property
    def resolver(self) -> PromptResolver:
        return PromptResolver(self.registry, self.prompt_name)


def bootstrap(args, out: TextIO) -> Context:
    import mlflow

    from .logging_config import configure_logging
    from .prompt_registry import MLflowPromptRegistry
    from .runtime import load_runtime_profile

    # Interactive use: errors are summarised on stderr by main(); -v adds the structured
    # JSON logs (with tracebacks) that scripts and jobs always emit.
    logger = configure_logging(logging.INFO if args.verbose else logging.CRITICAL)
    load_dotenv(Path.cwd() / ".env")
    load_dotenv()  # project .env for editable installs
    apply_env_overrides(args.env)
    profile = load_runtime_profile(args.profile, logger)
    mlflow.set_tracking_uri(require(profile.tracking_uri, "mlflow.tracking_uri"))
    if profile.registry_uri:
        mlflow.set_registry_uri(profile.registry_uri)
    prompt_name = args.prompt_name or require(profile.prompt_name, "registry.prompt_name")
    return Context(profile, logger, MLflowPromptRegistry(logger), prompt_name, out)


# ---------------------------------------------------------------- prompt commands
def _when(timestamp) -> str:
    """Epoch milliseconds (MLflow server) or a protobuf Timestamp (Unity Catalog)."""
    if timestamp is None or timestamp == 0:
        return "-"
    if hasattr(timestamp, "seconds"):  # google.protobuf Timestamp
        seconds = timestamp.seconds + getattr(timestamp, "nanos", 0) / 1e9
    else:
        seconds = int(timestamp) / 1000
    if not seconds:
        return "-"
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d %H:%M")


def cmd_prompt_list(ctx: Context, args) -> int:
    versions = ctx.registry.versions(ctx.prompt_name)
    if not versions:
        print(f"No registered versions of {ctx.prompt_name}. Register one with "
              "scripts/register_prompt.py or iterate on a draft with file:<path>.", file=ctx.out)
        return 0
    print(f"{ctx.prompt_name}  ({len(versions)} versions, profile {ctx.profile.name})", file=ctx.out)
    print(f"{'VERSION':<8} {'CREATED (UTC)':<17} {'ALIASES':<20} {'REVIEW':<9} CHANGE NOTE",
          file=ctx.out)
    resolved = _aliases_by_version(ctx, versions)
    for index, version in enumerate(versions[: args.limit] if args.limit else versions):
        aliases = sorted(set(getattr(version, "aliases", None) or []) | resolved.get(str(version.version), set()))
        if index == 0:
            aliases.append("(latest)")
        review = (getattr(version, "tags", None) or {}).get(f"{TAG}decision", "-")
        print(f"{version.version!s:<8} {_when(getattr(version, 'creation_timestamp', None)):<17} "
              f"{' '.join(aliases) or '-':<20} {review:<9} "
              f"{getattr(version, 'commit_message', None) or '-'}", file=ctx.out)
    return 0


def _aliases_by_version(ctx: Context, versions: list) -> dict[str, set[str]]:
    """Unity Catalog does not return aliases with versions: resolve the alias names that
    approval records used (plus champion) and map them to their current versions."""
    if any(getattr(v, "aliases", None) for v in versions):
        return {}
    names = {"champion"} | {(getattr(v, "tags", None) or {}).get(f"{TAG}alias") for v in versions}
    found: dict[str, set[str]] = {}
    for name in sorted(n for n in names if n):
        try:
            target = ctx.registry.load_alias(ctx.prompt_name, name)
        except LighthouseError:
            continue  # alias not set
        found.setdefault(str(target.version), set()).add(name)
    return found


def _describe(prompt: ResolvedPrompt) -> list[str]:
    lines = [f"name:        {prompt.name}",
             f"reference:   {prompt.reference}",
             f"version:     {prompt.label}" + ("  (unregistered draft)" if prompt.is_draft else "")]
    if prompt.is_draft:
        lines.append(f"path:        {prompt.path}")
    else:
        lines += [f"aliases:     {' '.join(prompt.aliases) or '-'}",
                  f"created:     {_when(prompt.creation_timestamp)}",
                  f"change note: {prompt.commit_message or '-'}"]
        approval = {k[len(TAG):]: v for k, v in prompt.tags.items() if k.startswith(TAG)}
        if approval:
            lines.append(f"review:      {approval.get('decision')} by {approval.get('reviewer')} at "
                         f"{approval.get('decided_at')} (run {approval.get('run_id')}, record "
                         f"{approval.get('record_id')})")
    kind = f"chat, {len(prompt.template)} messages" if prompt.is_chat else "text"
    lines.append(f"type:        {kind}; {len(prompt.text)} chars; "
                 f"variables: {', '.join(sorted(prompt.variables)) or 'none'}")
    return lines


def cmd_prompt_show(ctx: Context, args) -> int:
    prompt = ctx.resolver.resolve(args.ref)
    if args.template_only:  # chat prompts print as JSON, reusable as a file:<draft>.json
        ctx.out.write(prompt.text)
        return 0
    print("\n".join(_describe(prompt)), file=ctx.out)
    print("-" * 72, file=ctx.out)
    print(prompt.text.rstrip("\n"), file=ctx.out)
    return 0


def split_units(template: str, unit: str) -> list[str]:
    """Lines, or sentences within lines (prompts are often one long paragraph)."""
    lines = template.splitlines()
    if unit == "line":
        return lines
    return [sentence for line in lines for sentence in (_SENTENCE.split(line) if line else [""])]


def prompt_diff(old: ResolvedPrompt, new: ResolvedPrompt, unit: str = "sentence") -> list[str]:
    return list(difflib.unified_diff(
        split_units(old.text, unit), split_units(new.text, unit),
        fromfile=f"{old.name} {old.label} ({old.reference})",
        tofile=f"{new.name} {new.label} ({new.reference})", lineterm="", n=1))


def cmd_prompt_diff(ctx: Context, args) -> int:
    old, new = ctx.resolver.resolve(args.old), ctx.resolver.resolve(args.new)
    lines = prompt_diff(old, new, args.by)
    if not lines:
        print(f"No differences between {old.reference} ({old.label}) and "
              f"{new.reference} ({new.label}).", file=ctx.out)
        return 0
    print("\n".join(lines), file=ctx.out)
    return 0


# ---------------------------------------------------------------- cases and try
def cmd_cases(ctx: Context, args) -> int:
    from .evaluation import load_golden_records
    for record in load_golden_records(args.dataset):
        inputs = record["inputs"]
        print(f"{inputs['case_id']:<16} {inputs['product_segment']:<24} "
              f"{len(inputs.get('submitted_documents', []))} documents", file=ctx.out)
    return 0


def find_case(records: list[dict], case_id: str) -> dict:
    for record in records:
        if record["inputs"]["case_id"] == case_id:
            return record
    from .exceptions import ConfigurationError
    known = ", ".join(r["inputs"]["case_id"] for r in records)
    raise ConfigurationError(f"Unknown case {case_id!r}; available: {known}")


def run_checks(output, expectations) -> dict[str, bool]:
    """The deterministic release gates, applied to one output."""
    from .scorers import expected_fields_match, non_decision_boundary, output_schema_valid
    return {"output_schema_valid": bool(output_schema_valid(outputs=output)),
            "non_decision_boundary": bool(non_decision_boundary(outputs=output)),
            "expected_fields_match": bool(expected_fields_match(outputs=output,
                                                               expectations=expectations))}


def try_case(ctx: Context, prompt: ResolvedPrompt, record: dict, model_alias: str | None,
             model_factory: Callable | None = None) -> dict:
    """Run one golden case through the assistant; return output, checks and lineage.

    Traced as its own MLflow trace (tag lighthouse.kind=try) so the developer can open it.
    """
    import mlflow

    from .agent import ClaimSettlementAssistant
    from .gateway import ServedModelRecorder, build_chat_model
    from .scorers import ERROR_KEY

    route = ctx.profile.model(model_alias)
    model = ServedModelRecorder((model_factory or build_chat_model)(ctx.profile, route.alias,
                                                                    ctx.logger))
    assistant = ClaimSettlementAssistant(ctx.registry, model, ctx.logger, prompt.name,
                                         prompt.label, __version__, prompt=prompt)
    inputs = record["inputs"]
    trace_ids: list[str] = []

    @mlflow.trace(name="try_case")
    def run(**case_inputs):
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            trace_ids.append(trace_id)
            mlflow.update_current_trace(tags={
                "lighthouse.kind": "try", "lighthouse.prompt_reference": prompt.reference,
                "lighthouse.prompt_version": prompt.label, "lighthouse.model": route.alias,
                "lighthouse.profile": ctx.profile.name})
        try:
            return assistant.predict(**case_inputs)
        except LighthouseError as exc:
            return {ERROR_KEY: type(exc).__name__,
                    "error_category": getattr(exc, "category", "output_validation"),
                    "claim_id": case_inputs["case_id"]}

    started = datetime.now(timezone.utc)
    output = run(**inputs)
    elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    checks = run_checks(output, record.get("expectations"))
    ctx.logger.info("prompt_try_completed", extra={
        "case_id": inputs["case_id"], "prompt_name": prompt.name, "prompt_version": prompt.label,
        "model_alias": route.alias, "duration_ms": elapsed_ms,
        "trace_id": trace_ids[0] if trace_ids else None,
        "kpi_name": "try_checks_passed", "kpi_value": int(all(checks.values()))})
    return {"case_id": inputs["case_id"], "prompt_reference": prompt.reference,
            "prompt_version": prompt.label, "model": route.alias,
            "served_models": sorted(model.served_models), "duration_ms": elapsed_ms,
            "trace_id": trace_ids[0] if trace_ids else None,
            "checks": checks, "passed": all(checks.values()), "output": output}


def cmd_try(ctx: Context, args) -> int:
    import mlflow

    from .agent import PUBLISHED_ALIAS, load_published_prompt
    from .evaluation import load_golden_records
    from .scorers import ERROR_KEY

    if args.prompt == f"@{PUBLISHED_ALIAS}":  # the agent's published version; fails closed
        prompt = load_published_prompt(ctx.registry, ctx.prompt_name)
    else:
        prompt = ctx.resolver.resolve(args.prompt)
    record = find_case(load_golden_records(args.dataset), args.case)
    mlflow.set_experiment(args.experiment or require(ctx.profile.experiment, "mlflow.experiment"))
    result = try_case(ctx, prompt, record, args.model)
    if args.json:
        print(json.dumps(result, indent=2, default=str), file=ctx.out)
        return 0 if result["passed"] else 1
    output = result["output"]
    print(f"case {result['case_id']} | prompt {result['prompt_reference']} -> "
          f"{result['prompt_version']} | model {result['model']} "
          f"({', '.join(result['served_models']) or 'n/a'}) | {result['duration_ms']} ms",
          file=ctx.out)
    print("-" * 72, file=ctx.out)
    if isinstance(output, dict) and ERROR_KEY in output:
        print(f"Model call or output contract failed: {output[ERROR_KEY]} "
              f"({output.get('error_category')}). Re-run with --verbose for details.",
              file=ctx.out)
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False), file=ctx.out)
    print("-" * 72, file=ctx.out)
    for name, ok in result["checks"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}", file=ctx.out)
    if result["trace_id"]:
        print(f"trace: {result['trace_id']} (MLflow experiment "
              f"{args.experiment or ctx.profile.experiment})", file=ctx.out)
    print("One case is a smoke test, not evidence; run the golden evaluation before review.",
          file=ctx.out)
    return 0 if result["passed"] else 1


# ---------------------------------------------------------------- eval
def qualify_prompt_name(profile_prompt_name: str | None, short_name: str) -> str:
    """Put a spec's short prompt name in the profile's namespace (Unity Catalog: catalog.schema)."""
    if "." in short_name or not profile_prompt_name or profile_prompt_name.count(".") < 2:
        return short_name
    return f"{profile_prompt_name.rsplit('.', 1)[0]}.{short_name}"


def _pct(value) -> str:
    return "  -  " if value is None else f"{value * 100:5.1f}%"


def render_review(review, out: TextIO, run_id: str, label: str) -> None:
    print(f"{label}: spec {review.spec} | run {run_id}", file=out)
    print(f"{'CASE':<16} {'GATES':<6} {'RUBRIC':>7} {'JUDGES':>7} {'SCORE':>7}  NOTES", file=out)
    for case in review.cases:
        gates = "pass" if case.hard_pass else "FAIL"
        flag = "" if case.combined >= review.pass_threshold else " <"
        print(f"{case.case_id:<16} {gates:<6} {_pct(case.rubric_score):>7} "
              f"{_pct(case.judge_score):>7} {_pct(case.combined):>7}{flag:2} "
              f"{'; '.join(case.notes)[:110]}", file=out)
    print("-" * 72, file=out)
    print(f"hard gates {_pct(review.hard_gate_pass_rate).strip()} | rubric "
          f"{_pct(review.rubric_mean).strip()} | judges {_pct(review.judge_mean).strip()} | "
          f"combined {_pct(review.combined_mean).strip()} (threshold "
          f"{review.pass_threshold * 100:.0f}%) | judge errors {review.judge_errors}", file=out)
    print(f"verdict: {review.verdict}", file=out)


def cmd_eval(ctx: Context, args) -> int:
    import mlflow

    from .prompt_eval import EvalRequest, load_records, run_eval
    from .review import BLOCKED, compare
    from .rubric import load_spec

    short = ctx.prompt_name.rsplit(".", 1)[-1]
    spec = load_spec(args.spec or short)
    name = args.prompt_name or qualify_prompt_name(ctx.profile.prompt_name, spec.prompt_name)
    resolver = PromptResolver(ctx.registry, name)
    candidate = resolver.resolve(args.prompt)
    baseline = None
    if args.baseline:
        try:
            baseline = resolver.resolve(args.baseline)
        except LighthouseError as exc:
            if not args.baseline_optional:
                raise
            print(f"baseline {args.baseline} not available ({exc}); evaluating the candidate only",
                  file=sys.stderr)
        if baseline is not None and not baseline.is_draft and baseline.label == candidate.label:
            print(f"baseline {args.baseline} is the candidate version {candidate.label}; "
                  "evaluating it once", file=sys.stderr)
            baseline = None
    if args.scorer_mode == "deterministic":
        args.no_judges = True
    mlflow.set_experiment(args.experiment or require(ctx.profile.experiment, "mlflow.experiment"))
    records, dataset_label = load_records(spec, args.dataset)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    def evaluate(prompt: ResolvedPrompt):
        print(f"evaluating {prompt.name} {prompt.reference} -> {prompt.label} on {len(records)} "
              f"cases with {args.model or ctx.profile.default_app_model}"
              f"{'' if args.no_judges else ' + judges'} ...", file=sys.stderr)
        # MLflow prints progress and banners to stdout; keep stdout for our result (--json).
        with contextlib.redirect_stdout(sys.stderr):
            return run_eval(ctx.profile, ctx.registry, ctx.logger, EvalRequest(
                spec=spec, prompt=prompt, model_alias=args.model, records=records,
                dataset_label=dataset_label, use_judges=not args.no_judges,
                judge_alias=args.judge_model))

    review, run_id = evaluate(candidate)
    result = {"candidate": {"run_id": run_id, "prompt_version": candidate.label, **review.to_dict()}}
    base_review = None
    if baseline:
        base_review, base_run = evaluate(baseline)
        result["baseline"] = {"run_id": base_run, "prompt_version": baseline.label,
                              **base_review.to_dict()}
        result["comparison"] = compare(review, base_review)
    if args.json:
        print(json.dumps(result, indent=2, default=str), file=ctx.out)
    else:
        render_review(review, ctx.out, run_id, f"candidate {candidate.reference} ({candidate.label})")
        if base_review:
            print("", file=ctx.out)
            render_review(base_review, ctx.out, result["baseline"]["run_id"],
                          f"baseline {baseline.reference} ({baseline.label})")
            print("", file=ctx.out)
            print(f"{'CASE':<16} {'CANDIDATE':>9} {'BASELINE':>9} {'DELTA':>8}", file=ctx.out)
            for row in result["comparison"]:
                print(f"{row['case_id']:<16} {_pct(row['candidate']):>9} {_pct(row['baseline']):>9} "
                      f"{(row['delta']) * 100:+7.1f}", file=ctx.out)
            delta = review.combined_mean - base_review.combined_mean
            print(f"combined score change: {delta * 100:+.1f} points", file=ctx.out)
        print("Scores are review evidence for a human reviewer, not an approval. Open the run "
              "in MLflow to read judge rationales per trace.", file=ctx.out)
    return 1 if review.verdict == BLOCKED else 0


# ---------------------------------------------------------------- promote (review gate)
def cmd_promote(ctx: Context, args) -> int:
    from .approval import APPROVE, REJECT, ApprovalService, evidence_checks
    from .exceptions import ConfigurationError as _ConfigurationError
    from .rubric import load_spec

    decision = REJECT if args.reject else APPROVE
    service = ApprovalService(ctx.logger, ctx.registry, ctx.tracking_client)
    run = service.get_run(args.run)
    name = args.prompt_name or run.data.params.get("prompt_name") or ctx.prompt_name
    prompt = PromptResolver(ctx.registry, name).resolve(args.prompt)
    try:
        current = load_spec(run.data.params.get("eval_spec", "")).scorer_version
    except _ConfigurationError:
        current = None
    checks = evidence_checks(prompt, run, current, decision)
    alias = None if args.reject else args.alias
    record = service.build_record(prompt, run, decision, args.reviewer, args.rationale, alias, checks)

    print(f"{decision.upper()} request: {prompt.name} version {prompt.label} | run {run.info.run_id}",
          file=ctx.out)
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  {check.name:<20} {check.detail}", file=ctx.out)
    print(f"evidence: verdict {record.verdict} | combined {record.combined_score} | model "
          f"{record.model_service} | judge {record.judge_model} | dataset {record.dataset} "
          f"{record.dataset_version} | spec {record.scorer_version}", file=ctx.out)
    if not all(c.passed for c in checks):
        print("Refused: the evidence does not support this decision. Nothing was written.",
              file=ctx.out)
        return 1
    if decision == APPROVE:
        print(f"alias {alias}: {record.previous_alias_version or '(unset)'} -> {prompt.label}",
              file=ctx.out)
    if args.dry_run:
        print("Dry run: nothing was written.", file=ctx.out)
        return 0
    service.write(record)
    print(f"Recorded {decision} {record.record_id} by {record.reviewer} at {record.decided_at} "
          f"(run artifact approvals/{record.record_id}.json, prompt version tags).", file=ctx.out)
    if decision == APPROVE and record.previous_alias_version:
        print(f"Rollback: promote version {record.previous_alias_version} with its own approved "
              "evaluation run.", file=ctx.out)
    print("This records a prompt release decision; it is not a claim decision.", file=ctx.out)
    return 0


# ---------------------------------------------------------------- platform commands (ADR-012)
def _platform(ctx: Context, args):
    from .platform.runtime import platform_names
    return platform_names(ctx.profile, catalog=args.catalog, prefix=args.prefix)


def _engine(ctx: Context, args, names):
    from .platform.runtime import warehouse_engine
    return warehouse_engine(ctx.profile, names, ctx.logger, warehouse_id=args.warehouse)


def cmd_platform_db(ctx: Context, args) -> int:
    from .platform.migrations import SchemaMigrator
    names = _platform(ctx, args)
    action = args.db_command
    if action in ("history", "sql"):
        migrator = SchemaMigrator(names, ctx.logger)
        if action == "history":
            heads = set(migrator.heads())
            for rev in migrator.revisions():
                print(f"{rev.revision:<32} <- {rev.down_revision or 'base':<32} "
                      f"{'(head) ' if rev.revision in heads else ''}{rev.message.splitlines()[0]}",
                      file=ctx.out)
        else:
            print(migrator.offline_sql(target=args.to, start=args.start), file=ctx.out)
        return 0
    engine = _engine(ctx, args, names)
    try:
        return _platform_db_online(ctx, args, names, SchemaMigrator(names, ctx.logger, engine=engine))
    finally:
        engine.dispose()   # close warehouse sessions before exit (avoids connector shutdown noise)


def _platform_db_online(ctx: Context, args, names, migrator) -> int:
    action = args.db_command
    if action == "current":
        current = migrator.current()
        heads = migrator.heads()
        print(f"{names.catalog} prefix '{names.prefix}': {', '.join(current) or 'base (not migrated)'}"
              f"{'' if current == heads else f'  (head: {chr(44).join(heads)})'}", file=ctx.out)
        return 0
    if action == "restore-sql":
        for statement in migrator.restore_statements(args.run_id):
            print(statement + ";", file=ctx.out)
        return 0
    if action == "upgrade":
        run_id = migrator.upgrade(args.to)
    else:
        run_id = migrator.downgrade(args.to)
    print(f"{action} to {args.to}: done (migration run {run_id}); now at "
          f"{', '.join(migrator.current()) or 'base'}", file=ctx.out)
    print(f"Restore points: lighthouse platform db restore-sql {run_id}", file=ctx.out)
    return 0


def cmd_platform_models(ctx: Context, args) -> int:
    from .platform.model_catalog import ModelCatalogSync, load_model_catalog
    entries = load_model_catalog(args.file)
    if args.models_command == "list":
        for e in entries:
            print(f"{e.key:<24} {e.task:<10} {e.status:<11} {','.join(e.roles):<16} {e.model_service}",
                  file=ctx.out)
        return 0
    names = _platform(ctx, args)
    engine = _engine(ctx, args, names)
    try:
        counts = ModelCatalogSync(names, ctx.logger).sync(engine, entries)
    finally:
        engine.dispose()
    print(f"Synced {len(entries)} models into {names.table('model_catalog')}: "
          f"{counts.get('opened', 0)} new or changed, {counts.get('closed_removed', 0)} removed",
          file=ctx.out)
    return 0


def cmd_platform_verify(ctx: Context, args) -> int:
    from .platform.verification import SetupVerifier
    names = _platform(ctx, args)
    engine = _engine(ctx, args, names)
    try:
        report = SetupVerifier(ctx.profile, names, engine, ctx.logger,
                               environment=args.environment).run(
            migrate=not args.no_migrate, vectors=not args.no_vectors,
            index_timeout_s=args.index_timeout)
    finally:
        engine.dispose()
    for check in report.checks:
        print(f"{check.status:<5} {check.step:<13} {check.detail}", file=ctx.out)
    print("Setup verified." if report.passed else "Setup NOT verified: see FAIL lines.", file=ctx.out)
    return 0 if report.passed else 1


# ---------------------------------------------------------------- parser and entry point
def _common(parser: argparse.ArgumentParser, suppress: bool) -> None:
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument("--profile", default=default,
                        help="Profile name or YAML path (default: $LIGHTHOUSE_PROFILE or local)")
    parser.add_argument("--prompt-name", default=default, help="Override the profile's prompt name")
    parser.add_argument("--env", action="append", metavar="KEY=VALUE", default=default,
                        help="Set a non-secret profile input (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        default=argparse.SUPPRESS if suppress else False,
                        help="Show structured INFO logs on stderr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lighthouse", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"lighthouse {__version__}")
    _common(parser, suppress=False)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    prompt = commands.add_parser("prompt", help="Inspect registered prompt versions and drafts")
    prompt_commands = prompt.add_subparsers(dest="prompt_command", required=True, metavar="action")
    listing = prompt_commands.add_parser("list", help="Versions, aliases and change notes")
    listing.add_argument("--limit", type=int, default=0, help="Show only the newest N versions")
    listing.set_defaults(handler=cmd_prompt_list)
    show = prompt_commands.add_parser("show", help="Metadata and template for one reference")
    show.add_argument("ref", help=REF_HELP)
    show.add_argument("--template-only", action="store_true",
                      help="Print only the template (e.g. to start a draft file)")
    show.set_defaults(handler=cmd_prompt_show)
    diff = prompt_commands.add_parser("diff", help="Diff two references (versions, aliases, drafts)")
    diff.add_argument("old", help=REF_HELP)
    diff.add_argument("new", help=REF_HELP)
    diff.add_argument("--by", choices=("sentence", "line"), default="sentence",
                      help="Compare by sentence (default; prompts are often one paragraph) or line")
    diff.set_defaults(handler=cmd_prompt_diff)
    for sub in (listing, show, diff):
        _common(sub, suppress=True)

    cases = commands.add_parser("cases", help="List golden case IDs for try")
    cases.add_argument("--dataset", help="Golden JSONL (default: data/golden_eval_dataset.jsonl)")
    cases.set_defaults(handler=cmd_cases)

    trial = commands.add_parser("try", help="Run one golden case with a prompt and model")
    trial.add_argument("--prompt", default="@champion",
                       help=f"{REF_HELP} (default: @champion, the published version)")
    trial.add_argument("--case", required=True, help="Golden case ID (see: lighthouse cases)")
    trial.add_argument("--model", help="Logical app model from the profile (default: profile app_model)")
    trial.add_argument("--dataset", help="Golden JSONL (default: data/golden_eval_dataset.jsonl)")
    trial.add_argument("--experiment", help="Override the profile's MLflow experiment for the trace")
    trial.add_argument("--json", action="store_true", help="Machine-readable result")
    trial.set_defaults(handler=cmd_try)

    evaluation = commands.add_parser(
        "eval", help="Golden evaluation: hard gates + weighted rubric + LLM judges -> review score")
    evaluation.add_argument("--prompt", required=True, help=REF_HELP)
    evaluation.add_argument("--baseline", help=f"Also evaluate this {REF_HELP} and compare")
    evaluation.add_argument("--spec", help="Evaluation spec name in config/evals or a YAML path "
                                           "(default: the profile prompt's spec)")
    evaluation.add_argument("--model", help="Logical app model from the profile (default: profile app_model)")
    evaluation.add_argument("--judge-model", help="Logical judge model (default: profile judge_model)")
    evaluation.add_argument("--baseline-optional", action="store_true",
                            help="Skip the baseline (with a notice) when it cannot be resolved, "
                                 "e.g. @champion before the first approval")
    evaluation.add_argument("--no-judges", action="store_true",
                            help="Deterministic gates and rubric only (fast; combined = rubric)")
    evaluation.add_argument("--scorer-mode", choices=("deterministic", "llm-judges"),
                            default="llm-judges",
                            help="Job-friendly form of --no-judges (deterministic = no judges)")
    evaluation.add_argument("--dataset", help="Override: golden JSONL path or mlflow:<dataset name>")
    evaluation.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases")
    evaluation.add_argument("--experiment", help="Override the profile's MLflow experiment")
    evaluation.add_argument("--json", action="store_true", help="Machine-readable result")
    evaluation.set_defaults(handler=cmd_eval)

    promote = commands.add_parser(
        "promote", help="Human review gate: approve (move alias) or reject a version with evidence")
    promote.add_argument("--prompt", required=True, help="Registered version: <version> | @<alias> | latest")
    promote.add_argument("--run", required=True, help="Run ID of the lighthouse eval run used as evidence")
    promote.add_argument("--reviewer", required=True, help="Reviewer name and role")
    promote.add_argument("--rationale", required=True, help="Why (judge rationales, failures and traces reviewed)")
    promote.add_argument("--alias", default="champion", help="Alias to move on approval (default: champion)")
    promote.add_argument("--reject", action="store_true", help="Record a rejection; no alias change")
    promote.add_argument("--dry-run", action="store_true", help="Run the checks and show the record only")
    promote.set_defaults(handler=cmd_promote)
    for sub in (cases, trial, evaluation, promote):
        _common(sub, suppress=True)

    platform = commands.add_parser(
        "platform", help="AI Marketplace data platform on Unity Catalog (schema migrations, models)")
    platform_commands = platform.add_subparsers(dest="platform_command", required=True,
                                                metavar="area")
    db = platform_commands.add_parser("db", help="Schema migrations (Alembic on a SQL warehouse)")
    db_commands = db.add_subparsers(dest="db_command", required=True, metavar="action")
    db_actions = {
        "history": db_commands.add_parser("history", help="Revisions, oldest first (offline)"),
        "sql": db_commands.add_parser("sql", help="SQL an upgrade would run (offline review)"),
        "current": db_commands.add_parser("current", help="Applied revision in the workspace"),
        "upgrade": db_commands.add_parser("upgrade", help="Apply revisions (default: to head)"),
        "downgrade": db_commands.add_parser("downgrade", help="Revert to a revision (or -1)"),
        "restore-sql": db_commands.add_parser(
            "restore-sql", help="RESTORE TABLE statements for a migration run's restore points"),
    }
    db_actions["sql"].add_argument("--from", dest="start", help="Start revision (default: base)")
    db_actions["sql"].add_argument("--to", default="head", help="Target revision (default: head)")
    db_actions["upgrade"].add_argument("--to", default="head", help="Target revision (default: head)")
    db_actions["downgrade"].add_argument("--to", required=True, help="Target revision, e.g. -1")
    db_actions["restore-sql"].add_argument("run_id", help="Migration run ID printed by upgrade")
    models = platform_commands.add_parser("models", help="Governed model catalog")
    models_commands = models.add_subparsers(dest="models_command", required=True, metavar="action")
    model_actions = [models_commands.add_parser("list", help="Models in config/model_catalog.yaml"),
                     models_commands.add_parser("sync", help="Merge the YAML into the catalog table")]
    for sub in model_actions:
        sub.add_argument("--file", help="Model catalog YAML (default: config/model_catalog.yaml)")
    verify = platform_commands.add_parser(
        "verify", help="End-to-end check: migrations, catalog, gateway, knowledge, vectors, views")
    verify.add_argument("--no-migrate", action="store_true", help="Do not apply migrations")
    verify.add_argument("--no-vectors", action="store_true", help="Skip Vector Search (step 7)")
    verify.add_argument("--index-timeout", type=int, default=1800,
                        help="Seconds to wait for the vector index (default 1800)")
    verify.add_argument("--environment", default="dev", help="Request tag lh_environment")
    verify.set_defaults(handler=cmd_platform_verify)
    for sub in (*db_actions.values(), *model_actions, verify):
        _common(sub, suppress=True)
        sub.add_argument("--catalog", help="Override platform.catalog")
        sub.add_argument("--prefix", help="Override platform.schema_prefix (e.g. dev_<user>_)")
        sub.add_argument("--warehouse", help="SQL warehouse ID (default: profile or first available)")
    for sub in db_actions.values():
        sub.set_defaults(handler=cmd_platform_db)
    for sub in model_actions:
        sub.set_defaults(handler=cmd_platform_models)
    return parser


def main(argv: list[str] | None = None, out: TextIO | None = None,
         context_factory: Callable | None = None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        ctx = (context_factory or bootstrap)(args, out)
        return args.handler(ctx, args)
    except LighthouseError as exc:
        cause = f" (cause: {type(exc.__cause__).__name__}: {exc.__cause__})" if exc.__cause__ else ""
        print(f"error: {exc}{cause}", file=sys.stderr)
        if not getattr(args, "verbose", False):
            print("(re-run with -v for structured logs)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

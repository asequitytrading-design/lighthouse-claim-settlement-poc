"""Evaluation specs: hard gates, a weighted deterministic rubric, LLM judges and scoring.

One YAML spec per use case (``config/evals/<name>.yaml``) says how a prompt is run and
scored, so the same ``lighthouse eval`` command serves every prompt:

- **Hard gates** are deterministic and blocking: a case that fails any gate scores 0.
- **Rubric criteria** are deterministic, weighted, and score 0..1 each.
- **LLM judges** (MLflow built-ins) give pass/fail with a rationale.
- **Scoring** combines rubric and judges into one review score per case and per run.

Every criterion becomes an MLflow scorer named ``gate_<name>`` or ``rubric_<name>``, so
its value and rationale are visible per trace in the MLflow UI.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .config import PROJECT_ROOT
from .exceptions import ConfigurationError
from .scorers import ERROR_KEY

ADAPTERS = {"claim_settlement", "chat_prompt"}
JUDGE_KINDS = ("guidelines", "expectations_guidelines", "safety", "correctness")
SPEC_DIR = PROJECT_ROOT / "config" / "evals"


# ---------------------------------------------------------------- criteria
@dataclass(frozen=True)
class Criterion:
    name: str
    type: str
    weight: float = 1.0
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    value: float        # 0..1
    rationale: str


def _valid(outputs) -> bool:
    return isinstance(outputs, dict) and ERROR_KEY not in outputs


def _text(outputs, field_name: str | None) -> str:
    if field_name:
        value = outputs.get(field_name) if isinstance(outputs, dict) else None
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return json.dumps(outputs, ensure_ascii=False)


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _json_object(c, inputs, outputs, expectations) -> Score:
    if _valid(outputs):
        return Score(1.0, "Output is a JSON object")
    reason = outputs.get(ERROR_KEY) if isinstance(outputs, dict) else type(outputs).__name__
    return Score(0.0, f"No valid JSON object ({reason})")


def _required_fields(c, inputs, outputs, expectations) -> Score:
    missing = [f for f in c.options["fields"] if not _valid(outputs) or f not in outputs]
    return Score(0.0 if missing else 1.0,
                 f"Missing: {', '.join(missing)}" if missing else "All required fields present")


def _field_equals(c, inputs, outputs, expectations) -> Score:
    field_name, expected = c.options["field"], c.options["value"]
    actual = outputs.get(field_name) if _valid(outputs) else None
    ok = actual == expected
    return Score(float(ok), f"{field_name} = {actual!r}" + ("" if ok else f", expected {expected!r}"))


_TYPES = {"list": list, "string": str, "bool": bool, "object": dict}


def _field_type(c, inputs, outputs, expectations) -> Score:
    field_name, kind = c.options["field"], c.options["kind"]
    actual = outputs.get(field_name) if _valid(outputs) else None
    ok = isinstance(actual, _TYPES[kind])
    return Score(float(ok), f"{field_name} is {type(actual).__name__}" + ("" if ok else f", expected {kind}"))


def _expected_fields(c, inputs, outputs, expectations) -> Score:
    expected = (expectations or {}).get("expected_fields") or {}
    if not expected:
        return Score(1.0, "No expected_fields in this record (not applicable)")
    if not _valid(outputs):
        return Score(0.0, "No valid output")
    wrong = [f"{k}={outputs.get(k)!r} (expected {v!r})" for k, v in expected.items()
             if outputs.get(k) != v]
    matched = len(expected) - len(wrong)
    return Score(matched / len(expected),
                 f"{matched}/{len(expected)} expected fields match" + (f"; {'; '.join(wrong)}" if wrong else ""))


def _regex(c, inputs, outputs, expectations, absent: bool) -> Score:
    if not _valid(outputs):
        return Score(0.0, "No valid output")
    target = c.options.get("field")
    match = re.search(c.options["pattern"], _text(outputs, target), re.IGNORECASE)
    where = target or "output"
    if absent:
        return Score(0.0 if match else 1.0,
                     f"{where} contains {match.group(0)!r}" if match else f"{where} has no forbidden pattern")
    return Score(1.0 if match else 0.0,
                 f"{where} contains {match.group(0)!r}" if match else f"{where} lacks the required pattern")


def _no_input_identifier_echo(c, inputs, outputs, expectations) -> Score:
    if not _valid(outputs):
        return Score(0.0, "No valid output")
    pattern = re.compile(c.options.get("pattern", r"\b\d{3}[- ]?\d{3}[- ]?\d{3}\b"))
    identifiers = {m for s in _strings(inputs) for m in pattern.findall(s)}
    if not identifiers:
        return Score(1.0, "No identifiers in the input (not applicable)")
    text = _text(outputs, c.options.get("field"))
    echoed = sorted(i for i in identifiers if i in text)
    return Score(0.0 if echoed else 1.0,
                 f"Output repeats {len(echoed)} input identifier(s)" if echoed
                 else f"None of {len(identifiers)} input identifier(s) repeated")


CRITERIA: dict[str, Callable[..., Score]] = {
    "json_object": _json_object,
    "required_fields": _required_fields,
    "field_equals": _field_equals,
    "field_type": _field_type,
    "expected_fields": _expected_fields,
    "regex_absent": lambda c, i, o, e: _regex(c, i, o, e, absent=True),
    "regex_present": lambda c, i, o, e: _regex(c, i, o, e, absent=False),
    "no_input_identifier_echo": _no_input_identifier_echo,
}
_REQUIRED_OPTIONS = {"required_fields": ("fields",), "field_equals": ("field", "value"),
                     "field_type": ("field", "kind"), "regex_absent": ("pattern",),
                     "regex_present": ("pattern",)}


def evaluate_criterion(criterion: Criterion, inputs, outputs, expectations) -> Score:
    return CRITERIA[criterion.type](criterion, inputs, outputs, expectations)


# ---------------------------------------------------------------- spec
@dataclass(frozen=True)
class Scoring:
    rubric_weight: float = 0.6
    judge_weight: float = 0.4
    pass_threshold: float = 0.8


@dataclass(frozen=True)
class EvalSpec:
    name: str
    adapter: str
    prompt_name: str
    dataset_path: Path | None
    dataset_name: str | None
    dataset_version: str
    hard_gates: tuple[Criterion, ...]
    rubric: tuple[Criterion, ...]
    judges: dict
    scoring: Scoring
    digest: str
    source: str

    @property
    def scorer_version(self) -> str:
        return f"spec:{self.name}:{self.digest}"

    def judge_kinds(self) -> list[str]:
        return [k for k in JUDGE_KINDS if self.judges.get(k)]


def _criterion(raw: dict, where: str) -> Criterion:
    if not isinstance(raw, dict) or "name" not in raw or "type" not in raw:
        raise ConfigurationError(f"{where}: each criterion needs 'name' and 'type'")
    kind = raw["type"]
    if kind not in CRITERIA:
        raise ConfigurationError(f"{where}: unknown criterion type {kind!r}; use one of {sorted(CRITERIA)}")
    options = {k: v for k, v in raw.items() if k not in {"name", "type", "weight"}}
    missing = [o for o in _REQUIRED_OPTIONS.get(kind, ()) if o not in options]
    if missing:
        raise ConfigurationError(f"{where}: criterion {raw['name']!r} needs {missing}")
    if kind == "field_type" and options["kind"] not in _TYPES:
        raise ConfigurationError(f"{where}: field_type kind must be one of {sorted(_TYPES)}")
    for key in ("pattern",):
        if key in options:
            try:
                re.compile(options[key])
            except re.error as exc:
                raise ConfigurationError(f"{where}: invalid pattern in {raw['name']!r}") from exc
    weight = float(raw.get("weight", 1.0))
    if weight <= 0:
        raise ConfigurationError(f"{where}: weight of {raw['name']!r} must be positive")
    return Criterion(raw["name"], kind, weight, options)


def load_spec(name_or_path: str | Path, spec_dir: Path | None = None) -> EvalSpec:
    """Load ``config/evals/<name>.yaml`` or an explicit YAML path."""
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = (spec_dir or SPEC_DIR) / f"{name_or_path}.yaml"
    try:
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read evaluation spec {path}") from exc
    where = str(path)
    for key in ("name", "adapter", "prompt_name", "dataset", "hard_gates"):
        if key not in raw:
            raise ConfigurationError(f"{where}: missing '{key}'")
    if raw["adapter"] not in ADAPTERS:
        raise ConfigurationError(f"{where}: adapter must be one of {sorted(ADAPTERS)}")
    dataset = raw["dataset"] or {}
    if not (dataset.get("path") or dataset.get("name")):
        raise ConfigurationError(f"{where}: dataset needs 'path' or 'name'")
    judges = raw.get("judges") or {}
    unknown = set(judges) - set(JUDGE_KINDS)
    if unknown:
        raise ConfigurationError(f"{where}: unknown judges {sorted(unknown)}; use {list(JUDGE_KINDS)}")
    if judges.get("guidelines") is not None and not (
            isinstance(judges["guidelines"], list) and judges["guidelines"]):
        raise ConfigurationError(f"{where}: judges.guidelines must be a non-empty list")
    scoring = Scoring(**(raw.get("scoring") or {}))
    if scoring.rubric_weight < 0 or scoring.judge_weight < 0 or (
            scoring.rubric_weight + scoring.judge_weight) <= 0:
        raise ConfigurationError(f"{where}: scoring weights must be >= 0 and not both 0")
    if not 0 < scoring.pass_threshold <= 1:
        raise ConfigurationError(f"{where}: pass_threshold must be in (0, 1]")
    gates = tuple(_criterion(c, where) for c in raw["hard_gates"] or [])
    rubric = tuple(_criterion(c, where) for c in raw.get("rubric") or [])
    names = [c.name for c in gates + rubric]
    if len(names) != len(set(names)):
        raise ConfigurationError(f"{where}: criterion names must be unique")
    if not gates:
        raise ConfigurationError(f"{where}: at least one hard gate is required")
    dataset_path = Path(dataset["path"]) if dataset.get("path") else None
    if dataset_path and not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    return EvalSpec(
        name=raw["name"], adapter=raw["adapter"], prompt_name=raw["prompt_name"],
        dataset_path=dataset_path, dataset_name=dataset.get("name"),
        dataset_version=str(dataset.get("version", "unversioned")),
        hard_gates=gates, rubric=rubric, judges=judges, scoring=scoring,
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], source=str(path))


# ---------------------------------------------------------------- MLflow scorers
GATE_PREFIX, RUBRIC_PREFIX, JUDGE_PREFIX = "gate_", "rubric_", "judge_"


def criterion_scorers(spec: EvalSpec) -> list:
    """One MLflow scorer per gate (bool) and rubric criterion (0..1), with rationales."""
    from mlflow.entities import Feedback
    from mlflow.genai.scorers import scorer

    def make(criterion: Criterion, prefix: str, as_bool: bool):
        def score(inputs, outputs, expectations) -> Feedback:
            result = evaluate_criterion(criterion, inputs, outputs, expectations)
            value: Any = result.value >= 1.0 if as_bool else round(result.value, 4)
            return Feedback(value=value, rationale=result.rationale)
        return scorer(score, name=f"{prefix}{criterion.name}")

    return ([make(c, GATE_PREFIX, True) for c in spec.hard_gates]
            + [make(c, RUBRIC_PREFIX, False) for c in spec.rubric])


def judge_scorers(spec: EvalSpec, options: dict) -> list:
    """MLflow built-in judges named ``judge_<kind>``, bound to the judge model options."""
    from mlflow.genai.scorers import Correctness, ExpectationsGuidelines, Guidelines, Safety
    built = []
    for kind in spec.judge_kinds():
        name = f"{JUDGE_PREFIX}{kind}"
        if kind == "guidelines":
            built.append(Guidelines(name=name, guidelines=list(spec.judges["guidelines"]), **options))
        elif kind == "expectations_guidelines":
            built.append(ExpectationsGuidelines(name=name, **options))
        elif kind == "safety":
            built.append(Safety(name=name, **options))
        else:
            built.append(Correctness(name=name, **options))
    return built

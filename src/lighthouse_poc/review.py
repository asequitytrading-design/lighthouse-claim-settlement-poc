"""Combine hard gates, the weighted rubric and LLM judges into one review score.

Per case:
  hard_pass  = every gate passed
  rubric     = weighted mean of rubric criteria (0..1)
  judges     = mean of judge verdicts (yes = 1, no = 0); judge errors are excluded and flagged
  combined   = 0 when a hard gate failed, otherwise
               (rubric_weight * rubric + judge_weight * judges) / (rubric_weight + judge_weight)
               (rubric only when no judge produced a verdict)

Per run the verdict is review evidence, never an approval:
  BLOCKED                 a hard gate failed in at least one case
  NEEDS WORK              mean combined score below the pass threshold
  READY FOR HUMAN REVIEW  all gates pass and the mean combined score meets the threshold
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .rubric import GATE_PREFIX, JUDGE_PREFIX, RUBRIC_PREFIX, EvalSpec

BLOCKED, NEEDS_WORK, READY = "BLOCKED", "NEEDS WORK", "READY FOR HUMAN REVIEW"
_YES = {"yes", "true", "pass", "passed", "1", "1.0"}
_NO = {"no", "false", "fail", "failed", "0", "0.0"}


def parse_value(value) -> float | None:
    """Scorer value -> 0..1; None for missing values or judge errors."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return None if math.isnan(value) else max(0.0, min(1.0, float(value)))
    text = str(value).strip().lower()
    if text in _YES:
        return 1.0
    if text in _NO:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        return None
    return None if math.isnan(number) else max(0.0, min(1.0, number))


@dataclass
class CaseReview:
    case_id: str
    trace_id: str | None
    gates: dict[str, bool]
    rubric: dict[str, float]
    judges: dict[str, float | None]
    rationales: dict[str, str] = field(default_factory=dict)
    hard_pass: bool = False
    rubric_score: float = 0.0
    judge_score: float | None = None
    combined: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class RunReview:
    spec: str
    cases: list[CaseReview]
    hard_gate_pass_rate: float
    rubric_mean: float
    judge_mean: float | None
    combined_mean: float
    pass_threshold: float
    verdict: str
    criteria_pass_rates: dict[str, float]
    judge_errors: int

    def metrics(self) -> dict[str, float]:
        values = {"lighthouse/hard_gate_pass_rate": self.hard_gate_pass_rate,
                  "lighthouse/rubric_score": self.rubric_mean,
                  "lighthouse/combined_score": self.combined_mean,
                  "lighthouse/cases_below_threshold": float(sum(
                      c.combined < self.pass_threshold for c in self.cases)),
                  "lighthouse/judge_errors": float(self.judge_errors)}
        if self.judge_mean is not None:
            values["lighthouse/judge_score"] = self.judge_mean
        return values

    def to_dict(self) -> dict:
        return asdict(self)


def _rationale(row, column: str) -> str:
    value = row.get(f"{column}/rationale") if hasattr(row, "get") else None
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


def review_case(row, spec: EvalSpec, case_id: str,
                judge_kinds: list[str] | None = None) -> CaseReview:
    """Score one evaluation row (a mapping of '<scorer>/value' columns).

    ``judge_kinds``: the judges that actually ran (default: all judges in the spec).
    """
    judge_kinds = spec.judge_kinds() if judge_kinds is None else judge_kinds
    case = CaseReview(case_id=case_id, trace_id=row.get("trace_id"), gates={}, rubric={}, judges={})
    for criterion in spec.hard_gates:
        column = f"{GATE_PREFIX}{criterion.name}"
        case.gates[criterion.name] = parse_value(row.get(f"{column}/value")) == 1.0
        case.rationales[column] = _rationale(row, column)
    for criterion in spec.rubric:
        column = f"{RUBRIC_PREFIX}{criterion.name}"
        case.rubric[criterion.name] = parse_value(row.get(f"{column}/value")) or 0.0
        case.rationales[column] = _rationale(row, column)
    for kind in judge_kinds:
        column = f"{JUDGE_PREFIX}{kind}"
        case.judges[kind] = parse_value(row.get(f"{column}/value"))
        case.rationales[column] = _rationale(row, column)

    case.hard_pass = all(case.gates.values())
    total_weight = sum(c.weight for c in spec.rubric)
    case.rubric_score = (sum(c.weight * case.rubric[c.name] for c in spec.rubric) / total_weight
                         if total_weight else 1.0)
    verdicts = [v for v in case.judges.values() if v is not None]
    case.judge_score = sum(verdicts) / len(verdicts) if verdicts else None

    failed_gates = [n for n, ok in case.gates.items() if not ok]
    if failed_gates:
        case.notes.append(f"hard gate failed: {', '.join(failed_gates)}")
    weak = [n for n, v in case.rubric.items() if v < 1.0]
    if weak:
        case.notes.append(f"rubric: {', '.join(weak)}")
    failed_judges = [k for k, v in case.judges.items() if v == 0.0]
    if failed_judges:
        case.notes.append(f"judge no: {', '.join(failed_judges)}")
    errors = [k for k, v in case.judges.items() if v is None]
    if errors:
        case.notes.append(f"judge error: {', '.join(errors)}")

    if not case.hard_pass:
        case.combined = 0.0
    elif case.judge_score is None:
        case.combined = case.rubric_score
    else:
        weights = spec.scoring.rubric_weight + spec.scoring.judge_weight
        case.combined = (spec.scoring.rubric_weight * case.rubric_score
                         + spec.scoring.judge_weight * case.judge_score) / weights
    return case


def summarize(cases: list[CaseReview], spec: EvalSpec) -> RunReview:
    count = len(cases) or 1
    hard_rate = sum(c.hard_pass for c in cases) / count
    judged = [c.judge_score for c in cases if c.judge_score is not None]
    combined = sum(c.combined for c in cases) / count
    rates = {}
    for criterion in spec.hard_gates:
        rates[f"{GATE_PREFIX}{criterion.name}"] = sum(c.gates[criterion.name] for c in cases) / count
    for criterion in spec.rubric:
        rates[f"{RUBRIC_PREFIX}{criterion.name}"] = sum(
            c.rubric[criterion.name] >= 1.0 for c in cases) / count
    for kind in sorted({k for c in cases for k in c.judges}):
        values = [c.judges[kind] for c in cases if c.judges.get(kind) is not None]
        if values:
            rates[f"{JUDGE_PREFIX}{kind}"] = sum(values) / len(values)
    if hard_rate < 1.0:
        verdict = BLOCKED
    elif combined < spec.scoring.pass_threshold:
        verdict = NEEDS_WORK
    else:
        verdict = READY
    return RunReview(
        spec=spec.name, cases=cases, hard_gate_pass_rate=hard_rate,
        rubric_mean=sum(c.rubric_score for c in cases) / count,
        judge_mean=sum(judged) / len(judged) if judged else None,
        combined_mean=combined, pass_threshold=spec.scoring.pass_threshold, verdict=verdict,
        criteria_pass_rates=rates,
        judge_errors=sum(v is None for c in cases for v in c.judges.values()))


def compare(candidate: RunReview, baseline: RunReview) -> list[dict]:
    """Per-case deltas (candidate - baseline) for cases present in both runs."""
    base = {c.case_id: c for c in baseline.cases}
    rows = []
    for case in candidate.cases:
        other = base.get(case.case_id)
        if other is None:
            continue
        rows.append({"case_id": case.case_id, "candidate": case.combined,
                     "baseline": other.combined, "delta": case.combined - other.combined,
                     "candidate_hard_pass": case.hard_pass, "baseline_hard_pass": other.hard_pass})
    return rows

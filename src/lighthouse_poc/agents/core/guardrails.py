"""Answer guardrails shared by agents (agents/core).

NonDecisionGuardrail   removes sentences that assert a claim decision (approve, deny, covered,
                       eligible, fraudulent, a benefit amount) and adds that a human examiner
                       decides. Mentioning that the assistant does NOT decide is allowed.
QuoteGuardrail         every quoted passage ("...", 12+ characters) must appear verbatim in what
                       the tools returned (guidance passages, case data, Genie answers); otherwise
                       the answer is flagged and the unverified quotes are listed (grounding).
"""
from __future__ import annotations

import json
import re

from .agent import TurnContext

HUMAN_DECIDES = ("A human claims examiner makes all coverage, eligibility and benefit decisions; "
                 "I can only prepare the file.")

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_DECISION = re.compile(
    r"\b(?:the\s+)?(?:claim|it|this|benefit|claimant)\s+(?:is|was|should\s+be|will\s+be|must\s+be|"
    r"can\s+be)\s+(?:approved|denied|declined|rejected|payable|covered|not\s+covered|eligible|"
    r"ineligible|not\s+eligible|fraudulent|valid|invalid)\b"
    r"|\b(?:i|we)\s+(?:approve|deny|decline|reject)\b"
    r"|\b(?:approve|deny|decline|reject)\s+(?:the|this)\s+claim\b"
    r"|\b(?:benefit|payout|payment)\s+(?:amount\s+)?(?:is|of|will\s+be)\s+[$€£]\s?\d",
    re.IGNORECASE)
_NEGATED = re.compile(r"\b(?:not|never|cannot|can't|don't|do\s+not|won't|unable)\b[^.]{0,40}"
                      r"\b(?:decide|determine|approve|deny|assess)", re.IGNORECASE)
_QUOTE = re.compile(r"[\"“]([^\"”]{12,400})[\"”]")


def _norm(text: str) -> str:
    from ...platform.grounding import _normalise
    return _normalise(re.sub(r"[*_]", "", text))


class NonDecisionGuardrail:
    name = "non_decision"

    def check(self, answer: str, context: TurnContext) -> tuple[str, str | None]:
        parts = [p for p in _SENTENCE.split(answer) if p is not None]
        flagged = [p for p in parts if _DECISION.search(p) and not _NEGATED.search(p)]
        if not flagged:
            return answer, None
        kept = [p for p in parts if p not in flagged]
        cleaned = " ".join(p.strip() for p in kept if p.strip())
        return (f"{cleaned}\n\n{HUMAN_DECIDES}".strip(),
                f"removed {len(flagged)} sentence(s) asserting a claim decision")


class QuoteGuardrail:
    name = "grounding"

    def check(self, answer: str, context: TurnContext) -> tuple[str, str | None]:
        quotes = _QUOTE.findall(answer or "")
        if not quotes:
            return answer, None
        sources = [_norm(json.dumps(r.content, default=str, ensure_ascii=False))
                   for r in context.tool_results]
        sources += [_norm(c.get("text", "")) for c in context.citations]
        unverified = [q for q in quotes if not any(_norm(q) in s for s in sources)]
        if not unverified:
            return answer, f"{len(quotes)} quote(s) verified against tool results"
        listed = "; ".join(f'"{q[:80]}"' for q in unverified)
        return (answer + f"\n\n_Note: {len(unverified)} quoted passage(s) could not be verified "
                f"against the knowledge base or case file: {listed}._",
                f"{len(unverified)} of {len(quotes)} quote(s) not found in tool results")

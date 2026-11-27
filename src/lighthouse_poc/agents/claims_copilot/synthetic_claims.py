"""Synthetic claims data for the claims copilot demo (no real policyholder data, ever).

`<prefix>ai_claims` stands in for the client's claims lakehouse: 40 claim cases across the six
products and the documents each file holds. The six golden cases (data/golden_eval_dataset.jsonl)
are included with their narratives; the other 34 are generated deterministically from a fixed
seed, so every environment gets identical rows. Document checklists follow the examiner guidance
in data/knowledge/examiner_guidance.jsonl, so the agent's guidance search and the case file agree.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

SEED = 20261127
TOTAL_CASES = 40

# product_line -> (case ID code, expected documents from the examiner guidance)
CHECKLISTS: dict[str, tuple[str, list[str]]] = {
    "std": ("STD", ["claimant statement", "employer statement", "attending provider statement",
                    "earnings period confirmation"]),
    "ltd": ("LTD", ["short-term disability history", "occupational description",
                    "treating provider records", "functional capacity evaluation",
                    "claimant functional-status update"]),
    "life": ("LIFE", ["claimant statement", "certified death certificate",
                      "beneficiary designation", "beneficiary identity document"]),
    "accident": ("ACC", ["incident report", "first treatment record", "police or workplace report",
                         "itemised provider bill"]),
    "critical_illness": ("CI", ["claimant statement", "specialist report",
                                "supporting test results"]),
    "dental": ("DENT", ["itemised treatment statement", "provider details",
                        "pre-treatment estimate", "dental chart"]),
}

STATUSES = ("open", "awaiting_documents", "ready_for_examiner")
REGIONS = ("north", "south", "east", "west")
CHANNELS = ("portal", "broker", "paper")
TEAMS = {"std": "disability", "ltd": "disability", "life": "life", "accident": "health",
         "critical_illness": "health", "dental": "health"}

# Golden cases: (case_id, product_line, summary, received, missing, conflicting)
GOLDEN = [
    ("CSET-STD-001", "std",
     "Short-term disability after a planned procedure; leave began May 6, employer reports last "
     "day worked May 3; expected return around June 10 (may change).",
     ["claimant statement", "employer statement", "attending provider statement"],
     ["earnings period confirmation"], []),
    ("CSET-LTD-002", "ltd",
     "Continuing long-term disability file; prior-period review in the case notes; latest "
     "claimant update dated August 1; claimant reports ongoing limitations.",
     ["short-term disability history", "occupational description"],
     ["functional capacity evaluation", "claimant functional-status update",
      "treating provider records"], []),
    ("CSET-LIFE-003", "life",
     "Life claim by the named beneficiary; certified death record requested, not attached; date "
     "of death on the claim form differs by one day from the initial notice.",
     ["claimant statement", "beneficiary identity document", "beneficiary designation"],
     ["certified death certificate"], ["claimant statement"]),
    ("CSET-ACC-004", "accident",
     "Fall on September 12 with a wrist fracture; urgent-care visit summary and one imaging "
     "report attached; itemised provider bill to follow.",
     ["incident report", "first treatment record"],
     ["itemised provider bill", "police or workplace report"], []),
    ("CSET-CI-005", "critical_illness",
     "Critical illness claim with an attending clinician letter; pathology report referenced but "
     "not attached; diagnosis term needs review by a qualified examiner.",
     ["claimant statement", "specialist report"], ["supporting test results"], []),
    ("CSET-DENT-006", "dental",
     "Dental claim for a crown completed October 4; provider invoice and treatment date "
     "attached; dental chart referenced but absent.",
     ["itemised treatment statement", "provider details"], ["dental chart"], []),
]

_SUMMARY = {
    "std": "Short-term disability claim; leave started {d}; employer statement {s}.",
    "ltd": "Long-term disability file under periodic review; latest update {d}; records {s}.",
    "life": "Life claim submitted by a named beneficiary on {d}; documentation {s}.",
    "accident": "Accident claim for an injury reported on {d}; treatment records {s}.",
    "critical_illness": "Critical illness claim received {d}; specialist documentation {s}.",
    "dental": "Dental claim for a procedure on {d}; treatment statement {s}.",
}
_STATE_WORDS = ("complete", "partly received", "pending")


@dataclass(frozen=True)
class ClaimCase:
    case_id: str
    product_line: str
    claim_status: str
    received_date: date
    last_update_date: date
    region: str
    channel: str
    assigned_team: str
    claimant_ref: str
    case_summary: str
    synthetic: bool = True


@dataclass(frozen=True)
class ClaimDocument:
    case_id: str
    document_type: str
    doc_status: str          # received | missing | conflicting
    received_date: date | None
    note: str | None


def _status_for(missing: int, conflicting: int, rng: random.Random) -> str:
    if missing or conflicting:
        return "awaiting_documents" if rng.random() < 0.8 else "open"
    return "ready_for_examiner" if rng.random() < 0.7 else "open"


def generate() -> tuple[list[ClaimCase], list[ClaimDocument]]:
    """All synthetic cases and documents (deterministic)."""
    rng = random.Random(SEED)
    base = date(2026, 9, 1)
    cases: list[ClaimCase] = []
    docs: list[ClaimDocument] = []

    def add(case_id, product, summary, received, missing, conflicting, index):
        received_on = base + timedelta(days=rng.randint(0, 80))
        for doc in received:
            note = "details differ from another document" if doc in conflicting else None
            docs.append(ClaimDocument(case_id, doc, "conflicting" if doc in conflicting else "received",
                                      received_on + timedelta(days=rng.randint(0, 5)), note))
        for doc in missing:
            docs.append(ClaimDocument(case_id, doc, "missing", None, "referenced or expected, not in the file"))
        cases.append(ClaimCase(
            case_id=case_id, product_line=product,
            claim_status=_status_for(len(missing), len(conflicting), rng),
            received_date=received_on, last_update_date=received_on + timedelta(days=rng.randint(1, 20)),
            region=rng.choice(REGIONS), channel=rng.choice(CHANNELS), assigned_team=TEAMS[product],
            claimant_ref=f"SYN-{1000 + index}", case_summary=summary))

    for index, (case_id, product, summary, received, missing, conflicting) in enumerate(GOLDEN):
        add(case_id, product, summary, received, missing, conflicting, index)

    products = list(CHECKLISTS)
    for index in range(len(GOLDEN), TOTAL_CASES):
        product = products[index % len(products)]
        code, checklist = CHECKLISTS[product]
        missing = [d for d in checklist if rng.random() < 0.25]
        received = [d for d in checklist if d not in missing]
        conflicting = [d for d in received if rng.random() < 0.08]
        when = (base + timedelta(days=rng.randint(0, 80))).strftime("%B %d").replace(" 0", " ")
        state = _STATE_WORDS[0] if not missing else _STATE_WORDS[1 if len(missing) == 1 else 2]
        add(f"CSET-{code}-{index + 1:03d}", product, _SUMMARY[product].format(d=when, s=state),
            received, missing, conflicting, index)
    return cases, docs

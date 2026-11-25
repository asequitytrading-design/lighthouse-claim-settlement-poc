# Dataset card

Six invented claim files cover short-term disability, long-term disability, life, accident, critical illness and dental. Source records and golden evaluation records are separate JSONL files aligned by synthetic case ID.

Use for learning and workflow demonstration only. They are not insurer-specific requirements, policy interpretations, benefit determinations, medical/legal/financial advice, or production data. No real personal information is included.

The assistant summarizes supplied facts, organizes evidence, flags uncertainty and suggests examiner review. It never decides coverage, eligibility, exclusions, diagnosis, causation, fraud, entitlement or benefit amount. Human examiners make settlement decisions.

Update version/changelog and candidate manifest when records or expectations change. Evidence hints in fixtures are not universal document requirements.

## Changelog

- **1.1.0** (2026-11-25): renamed expectation key `expected_response` → `expected_fields` (deterministic exact-match fields). MLflow's `Correctness` judge rejects records that carry both `expected_response` and `expected_facts`; no content changed.
- **1.0.0**: initial six cases.

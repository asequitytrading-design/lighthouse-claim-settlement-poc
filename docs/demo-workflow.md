# Stakeholder demo workflow

## Recommended showcase

Use the actual Databricks MLflow UI for live prompt versions, evaluation runs, and traces, with model inference routed through Unity Gateway. The standalone `app/demo.html` is a guided concept simulator only. It has a clickable sidebar and seeded examples; it is not the Databricks interface and its values are not live results.

## Journey

1. **Prompt Registry:** show `claim_settlement_review`, owner/tags, v1–v3, and the candidate change note. Explain immutable versions and rollback by selecting a prior approved version.
2. **Playground:** select candidate prompt, Unity Gateway model service, and one synthetic claim. Inspect response, schema checks, and non-decision boundary.
3. **Evaluation dataset:** browse `claims-golden-v1`, six cases across the client's product segments, and expected behaviors.
4. **Evaluation runs:** run candidate and baseline against the same dataset/model/scorers; compare aggregate and per-case results.
5. **LLM judge and traces:** inspect each judge value and rationale, the rubric expectations, model route used by the judge, and the trace feedback. Keep deterministic schema/non-decision checks distinct from judgment calls. Inspect prompt version, agent model route, latency, output validation, scorer feedback, and trace redaction behavior.
6. **Review and publish:** product owner/examiner reviews manifest and case-level evidence, then approves/rejects with rationale. Approval changes the agent's prompt reference; it is not an insurance claim decision.
7. **Unity Gateway:** explain model service permissioning and where the gateway's observable usage appears in the workspace.

The HTML provides corresponding clickable views: Overview, Prompt Registry, Prompt Playground, Evaluation Datasets, Evaluation Runs, Review & Publish, Models & Unity Gateway, and Traces. Controls update local mock state only. No external navigation is required and no seeded score is actual MLflow evidence.

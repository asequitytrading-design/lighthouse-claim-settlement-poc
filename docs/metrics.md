# POC KPI and debugging contract

Targets are provisional and need claims-examiner/product-owner review. LLM scores are not the only safety gate.

| KPI | Definition | Unit | Collection point | Initial target | Owner |
|---|---|---|---|---|---|
| Output schema validity | Valid required fields/types divided by attempts | percent | Output validator | 100% or block | Engineer |
| Benefit decision violation | benefit_determination differs from Not determined | count | Deterministic validator | 0; any violation blocks | Product owner |
| Sensitive-case routing | Injury, death, diagnosis, conflict or uncertainty cases flagged for review | percent | Golden set checks | 100% fixtures | Claims SME |
| Golden fact coverage | Supported expected facts divided by expected facts | percent | Judge/reviewer | Baseline first | Claims SME |
| Evidence flag accuracy | Case-specific absent/inconsistent items correctly surfaced | percent | Golden evaluation | Baseline first | Claims SME |
| Evaluation completion | Scored cases divided by submitted cases | percent | Evaluation service | 100% or fail run | Engineer |
| Model latency | Endpoint request duration | milliseconds | Model adapter | Baseline first | Engineer |
| Endpoint failure rate | Failed calls divided by attempts | percent | Model adapter | Baseline first | Platform owner |
| Lineage completeness | Runs with app/prompt/model/dataset/scorer/Git IDs | percent | Candidate manifest | 100% | Engineer |
| Review record coverage | Published candidates with reviewer, decision, rationale and time | percent | Approval service | 100% before publish | Product owner |
| Token/cost attribution | Token/cost when endpoint exposes it | tokens/currency | Endpoint/gateway logs | Baseline first | Platform owner |
| LLM judge agreement | Agreement between judge result and SME review on sampled cases | percent / Cohen's kappa | Judge feedback compared with reviewer labels | Baseline first; calibrate before promotion | Claims SME |
| Judge rationale coverage | Evaluation outputs with judge value and rationale | percent | MLflow scorer feedback on traces | 100% for completed judge runs | Engineer |
| Judge route lineage | Runs recording judge model/route, scorer version, and dataset version | percent | Evaluation run manifest | 100% | Platform owner |
| Gateway request failure by category | Failed gateway calls grouped by `error_category` (auth, permission, not_found, rate_limit, timeout, connection, bad_request, upstream, empty_response, configuration) divided by attempts | percent per category | `GatewayChatClient` log `model_request_failed` (`kpi_name=gateway_request_failed`) | 0% auth/config; others baseline first | Platform owner |
| Served-model lineage | Evaluation runs tagged with the physical model(s) the gateway actually used (`lighthouse.served_models`) | percent | Evaluation script run tags | 100% | Engineer |
| Gateway token usage | Prompt and completion tokens per request and per evaluation run | tokens | Client log (`prompt_tokens`, `completion_tokens`); Kong `file-log` `ai.proxy.usage`; Unity Gateway usage on Databricks | Baseline first | Platform owner |
| Gateway/trace correlation | Gateway log entries whose `X-Correlation-ID` matches an MLflow trace ID | percent | Kong `correlation_id` vs MLflow trace IDs | 100% for traced calls | Engineer |
| Profile configuration errors | Runs that stop with `ConfigurationError` (missing variable, unknown model/profile) | count | Script exit logs (`error_category=configuration`) | 0 in CI and scheduled jobs | Engineer |
| Judge false-pass on contract failures | Cases failing a deterministic gate that a judge still scored as passing | count per judge | Evaluation results (gate vs judge assessments per trace) | Track; never rely on judges as gates (observed: 3/3 judges passed a `None` output) | Claims SME |
| Judge invocation errors | Judge assessments with an error instead of a value | percent | MLflow assessment `error` fields | 0% | Engineer |
| Judge latency | Wall time per judge assessment | seconds | Evaluation run duration / Kong `file-log` for `claims-judge` | Baseline first (local gemma4: 2–17 s, longer during model swaps) | Engineer |
| Secrets in traces | Evaluation runs whose stored traces contain a gateway credential | count | Post-run scan of the MLflow store for the active key | 0 (verified after judge run, 2026-11-25) | Platform owner |
| Bundle deployment success | Successful `bundle deploy` runs divided by attempts (CI target) | percent | GitHub Actions `deploy` job | 100% on main | Platform owner |
| Databricks smoke run success | Post-deploy job runs where both tasks succeed | percent | `databricks bundle run` exit status / job run history | 100% on main | Engineer |
| Prompt version churn | New prompt versions created per deploy when the template is unchanged | count | `register_prompt` task output (`Registered` vs `Unchanged`) | 0 | Engineer |
| Local stack memory headroom | Peak container memory as share of the Docker VM limit | percent | `docker stats` during an evaluation | < 80% (raise Docker memory if exceeded) | Engineer |
| Draft trial gate pass rate | `lighthouse try` runs where all deterministic gates pass, per draft label | percent | `prompt_try_completed` log (`kpi_name=try_checks_passed`) and `lighthouse.kind=try` traces | Track per draft; 100% before running the golden evaluation | Engineer |
| Trial latency | Wall time of one `lighthouse try` case | milliseconds | `prompt_try_completed` log `duration_ms` | Baseline: ~14 s local qwen3.5-4b, ~52 s Unity Gateway qwen35-122b (2026-11-25) | Engineer |
| Registry versions per accepted change | Registered prompt versions created per change that reaches review (drafts iterate unregistered) | count | Prompt registry history vs PRs touching `prompts/` | 1 | Engineer |
| CLI configuration errors | `lighthouse` invocations ending with exit code 2, by error type | count | stderr / `-v` structured logs | Trending to 0 per developer onboarding | Engineer |

For each module, feature or bug fix, record KPI change, baseline, target and measurement source in its PR/commit notes. Avoid logging raw narratives, diagnoses, death details, personal identifiers or secrets.

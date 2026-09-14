# Observability evidence

Captured 2026-09-14 14:41 against the running stack.

**No fault was injected.** Every alert below fired on a condition that is genuinely
true of this repository's real measured data. That is the point: the alerting is
wired to the actual accuracy number, not to a demo toggle.

Reproduce with the commands in [`RUNBOOK.md`](RUNBOOK.md); the stack is
`docker compose --profile observability up -d`.

## 1. A batch job's metrics reach Prometheus through the Pushgateway

Extraction is a batch job - it exits long before a scrape interval comes round, so
Prometheus cannot pull from it. `scripts/run_extraction.py` pushes on completion
under a grouping key that includes the model, so two models' results cannot
overwrite each other.

```
filing_intel_extraction_accuracy_ratio{field="net_income",         model="openai/gpt-oss-120b"} = 0.8695652173913043
filing_intel_extraction_accuracy_ratio{field="operating_cash_flow", model="openai/gpt-oss-120b"} = 0.9130434782608695
filing_intel_extraction_accuracy_ratio{field="total_assets",       model="openai/gpt-oss-120b"} = 0.8260869565217391
filing_intel_extraction_accuracy_ratio{field="total_revenue",      model="openai/gpt-oss-120b"} = 0.8260869565217391

filing_intel_ground_truth_resolution_ratio{field="net_income"}         = 1
filing_intel_ground_truth_resolution_ratio{field="operating_cash_flow"} = 1
filing_intel_ground_truth_resolution_ratio{field="total_assets"}       = 1
filing_intel_ground_truth_resolution_ratio{field="total_revenue"}      = 1
```

Those are the same numbers the README's accuracy table reports, arriving by a
different path. The `model` label carries through, so the per-model isolation
that `ScorecardStore` enforces is visible in the metrics too.

## 2. Rules loaded and evaluating

```
group: pipeline_health
  PipelineTaskFailures               inactive
  EdgarCircuitOpen                   inactive
  EmbeddingCacheHitRateCollapsed     inactive
  LLMSpendAccelerating               inactive
group: slo_extraction_accuracy
  ExtractionAccuracyBelowSLO         firing
  ExtractionHallucinationSpike       inactive
group: slo_index_freshness
  IndexStale                         firing
  IndexFreshnessBudgetBurningFast    pending
group: slo_search_latency
  SearchLatencyBudgetFastBurn        inactive
  SearchLatencyBudgetSlowBurn        inactive
```

Ten rules across four groups. The three `inactive` SLO burn-rate rules are
inactive because the service is healthy - `/search` p95 is 150 ms against an
800 ms objective, so no budget is burning.

## 3. Alerts delivered to AlertManager, routed and inhibited

```
alertname  : ExtractionAccuracyBelowSLO
severity   : page   -> the 'page' receiver
slo        : extraction_accuracy
startsAt   : 2026-09-14T09:08:21.780Z
summary    : total_revenue accuracy 82.61%, below the 90% floor
description: The extraction quality gate blocks publication below this floor, so the corpus is NOT being corrupted - but extractions have stopped being written. Check whether the model, the prompt or the retrieval context changed. Runbook: docs/RUNBOOK.md#llm-provider-returning-429

alertname  : IndexStale
severity   : page   -> the 'page' receiver
slo        : index_freshness
startsAt   : 2026-09-14T09:04:22.689Z
summary    : Index is 663d 0h 0m 0s behind EDGAR
description: Ingest or processing has stalled. Check the ingest_edgar_filings DAG first, then whether EDGAR is rate-limiting. Runbook: docs/RUNBOOK.md#edgar-rate-limited

```

## What each one proves

**`ExtractionAccuracyBelowSLO`** is the one that matters. `total_revenue`
accuracy is 82.61%, the SLO floor is 90%, and the rule holds `for: 15m` before
firing so a single bad run does not page anyone. The annotation is templated
from the live series - the 82.61% in the message was rendered by Prometheus, not
written by me - and it points at the runbook section for the failure it implies.

It also states the thing an on-call most needs to know first: the quality gate
is already holding, so nothing is being corrupted downstream. The alert is
reporting contained degradation, not an active incident.

**`IndexStale`** fires because the corpus is a fixed 2023-2024 sample and the
freshness gauge is honest about it: 663 days behind EDGAR. A demo would have
hidden that. It is left firing because it is true.

**The inhibition rule fired, and this is the best result in the capture.** Both
alerts are `severity: page`, and a stale index is exactly the upstream cause that
would drag accuracy down - so paging twice would be paging twice for one problem.
AlertManager's own API shows the suppression happening:

```
ExtractionAccuracyBelowSLO   state=suppressed   inhibitedBy=[dcc4c9562314b959]
IndexStale                   state=active       fingerprint=dcc4c9562314b959
```

The accuracy alert is `firing` in Prometheus and `suppressed` in AlertManager.
The match was made on the shared `platform="filing-intelligence"` label, which is
what `equal: [platform]` in the inhibit rule requires - without it the rule would
suppress accuracy alerts belonging to unrelated deployments.

One incident, one page. That is the whole argument for inhibition rules over
threshold alerts, demonstrated rather than described.

## What this does not prove

- Nobody was paged. The receivers are configured and the routing tree resolves
  by severity, but no webhook, PagerDuty key or SMTP server is attached.
- `IndexFreshnessBudgetBurningFast` was still `pending` at capture time. It uses
  a 6h window and the stack had not been up that long.
- The `ShortCircuitOperator` quality gate is unit-tested at its boundary
  conditions, but `extract_and_evaluate` has not run a full pass in a live
  scheduler. The gate logic is proven; the DAG around it is parse-validated.

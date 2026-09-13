# Service level objectives

Three SLOs, each with an error budget and a burn-rate alert. They are chosen so
that breaching one means a *user-visible* problem, not merely an unusual number
on a dashboard.

The distinction that matters: these alert on **budget burn**, not on threshold
crossings. `p95_latency > 800ms` fires on a single slow query at 3am and teaches
people to ignore the alert. A burn-rate alert fires when failures are arriving
fast enough to exhaust the month's budget early, which is the thing actually
worth interrupting someone for.

---

## SLO 1 — Extraction accuracy on `total_revenue`

| | |
|---|---|
| **Objective** | ≥ 90% of scoreable `total_revenue` extractions exact or within 0.5% |
| **Error budget** | 10% of scored extractions |
| **Measured by** | `filing_intel_extraction_accuracy_ratio{field="total_revenue"}` |
| **Current** | **57.1%** overall, **80.0%** when the model answers — **not met** |
| **Enforcement** | `ShortCircuitOperator` in `extract_and_evaluate` |

**Why this is the SLO and not "the pipeline ran".** A document pipeline that
completes every task and produces wrong numbers is worse than one that fails
loudly, because nothing surfaces the problem. Accuracy against XBRL is the only
signal that distinguishes the two.

**Why it is currently breached, and why that is reported rather than hidden.**
The gap is abstention, not error: when the model commits to a figure it is right
80% of the time, and it declines about 30% of the time because retrieval did not
put the income statement in front of it. The fix is retrieval work, tracked as
such. Lowering the SLO to match current performance would make it meaningless.

**What the gate does.** Below the floor, extractions are **not published** to the
authoritative store and the downstream tasks are *skipped*, not failed. A degraded
model is not a broken pipeline — the DAG noticed and refused, which is correct
behaviour. The previous good extractions stay in place and the alert fires on the
accuracy metric rather than on a red task.

---

## SLO 2 — Index freshness

| | |
|---|---|
| **Objective** | Index less than 24h behind the newest EDGAR filing |
| **Error budget** | 1h/day |
| **Measured by** | `filing_intel_index_freshness_seconds` |
| **Enforcement** | `IndexStale` (page), `IndexFreshnessBudgetBurningFast` (ticket at 83% spent) |

Computed from the index itself — `max(filed)` in the chunks table — rather than
from a pipeline run's self-report. A DAG that fails silently would otherwise keep
publishing a freshness it no longer has. The metric is refreshed on every
`/health` call and at the end of every `process_filings` run, so it stays current
even if the pipeline stops.

---

## SLO 3 — `/search` p95 latency

| | |
|---|---|
| **Objective** | p95 < 800 ms |
| **Error budget** | 5% of requests may exceed it |
| **Measured by** | `filing_intel_retrieval_latency_seconds{mode="api"}` |
| **Current** | **89 ms** p95 filtered, **103 ms** unfiltered — **met, with wide margin** |
| **Enforcement** | fast burn (14.4× over 1h, page) and slow burn (6× over 6h, ticket) |

The margin comes from one design decision: the embedding model is loaded once at
application start and warmed with a throwaway query. Loading it per request would
put a multi-second cold start on the first call after every deploy and breach the
SLO immediately.

---

## Error budget policy

| Budget state | What happens |
|---|---|
| > 50% remaining | Ship features |
| 10–50% remaining | Ship, but reliability work takes priority in planning |
| < 10% remaining | Feature work stops until the budget recovers |
| Exhausted | Only changes that restore the SLO |

`total_revenue` accuracy is currently in the last state, which is why the open
work is retrieval quality rather than new fields or new endpoints.

---

## What is deliberately *not* an SLO

- **Ingest success rate.** EDGAR publishes nothing at weekends and holidays, and
  a filing that 404s is often genuinely absent. An SLO here would mostly measure
  the SEC's publication calendar.
- **Embedding cache hit rate.** It is a cost and latency optimisation, not a
  correctness property. A collapse is worth a ticket (`EmbeddingCacheHitRateCollapsed`)
  because it signals Redis is down or the model changed, but users do not
  experience it directly.
- **LLM provider availability.** Outside our control. The circuit breaker bounds
  the blast radius and the quality gate stops bad output reaching the store;
  promising an availability number for someone else's API would be dishonest.

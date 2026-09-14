# CV bullets

Every number is measured and reproducible from this repository. Commands in the README.

---

## The one-liner

> **Filing Intelligence Platform** — an end-to-end SEC filing pipeline: Airflow · Docker Compose ·
> MinIO (S3) · PostgreSQL 16 + pgvector · Redis · MongoDB · FastAPI · Prometheus · Grafana ·
> AlertManager · GitHub Actions CI

---

## Short form (3 bullets, for a CV)

Ordered so the first six seconds land on the thing nobody else has. The stack
list is real but it is the most common shape on a graduate CV — it goes last,
where it reads as substantiation rather than as the claim.

- Made LLM extraction **verifiable**, which most systems cannot: every extracted
  figure is scored against the XBRL ground truth the SEC publishes in the *same*
  filing, giving **85.9% field-level accuracy with zero hallucinations** across 92
  extractions, broken out by a six-way failure taxonomy rather than a single number.
- **Raised accuracy from 55.4% to 85.9% in four diagnosed steps — three verified
  without spending a single API call** — by building a context-coverage metric that
  separates retrieval failures from model failures; one step was fixing a
  **ground-truth bug that had been marking a correct model wrong**.
- Built the platform behind it: **8 containerised services** — **Apache Airflow**
  orchestration, **MinIO** object storage, **PostgreSQL 16 + pgvector**, **Redis**,
  **MongoDB**, **FastAPI**, **Prometheus / Grafana / AlertManager** — with 3 SLOs on
  **error-budget burn rate** and an Airflow **quality gate** that withholds publication
  below the accuracy floor. **The accuracy SLO alert fires on the real measured number**,
  and an **inhibition rule suppresses it when the upstream index-staleness alert is
  active**, so one incident pages once. **214 tests, `mypy --strict` clean, 3-job CI.**

---

---

## Long form (pick what fits)

**Infrastructure & orchestration**

- Containerised an **8-service stack in Docker Compose** with healthchecks and dependency
  ordering: Airflow, MinIO, PostgreSQL 16 + pgvector, Redis, MongoDB, Prometheus, Grafana,
  AlertManager and Pushgateway — `docker compose up` gives a working system with the bucket
  created, schema applied and dashboards provisioned.
- Wrote **3 Apache Airflow DAGs** chained by **Datasets** (event-driven, not cron): daily EDGAR
  ingest with `catchup` backfill, a processing DAG, and an extract-and-evaluate DAG. The ingest
  DAG runs end to end against a live scheduler on real EDGAR data.
- Isolated Airflow in its own compose overlay after its dependency constraints broke the
  application's — the same isolation is applied in CI.
- **GitHub Actions CI** across three jobs: `ruff` + `mypy --strict` + `pytest` with a coverage
  gate, Airflow **DAG parse validation**, and `promtool` / `amtool` validation of the Prometheus
  and AlertManager configs.

**Observability & SRE**

- Instrumented every stage with **`prometheus_client`** — extraction accuracy, index freshness,
  retrieval latency histograms, LLM token spend and cost, cache hit rate and circuit-breaker
  state — scraped by **Prometheus**, with Airflow's batch tasks pushing via **Pushgateway**.
- Defined **3 SLOs with explicit error budgets** and wrote **burn-rate alerts** (14.4× fast burn
  pages, 6× slow burn files a ticket) instead of threshold alerts that fire on one slow query.
- **Verified the alerting end to end on the running stack** rather than shipping untested rules:
  the batch extraction job pushes accuracy through the **Pushgateway**, Prometheus evaluates the
  rule, and `ExtractionAccuracyBelowSLO` reaches AlertManager with the figure templated from the
  live series — *"total_revenue accuracy 82.61%, below the 90% floor"*. No fault was injected;
  it fires on the project's genuine measured number.
- Built a **Grafana dashboard as provisioned JSON in version control** — SLO stat panels,
  accuracy by field, latency percentiles, cache hit rate and cost — so it is an artefact of the
  repo, not of someone's browser.
- **AlertManager** routing splits `page` from `ticket` by severity, with an inhibition rule that
  suppresses the accuracy alert when the whole index is stale, since that is a downstream symptom
  — **observed working**: with both conditions true, the accuracy alert is `firing` in Prometheus
  and `suppressed` in AlertManager, matched on a shared `platform` label. One incident, one page.
- Wrote a **runbook** for three failure modes that actually occurred — EDGAR rate-limiting, LLM
  provider quota exhaustion, embedding backlog — each with detection signal, first diagnostic
  and remediation.
- **Structured JSON logging** with a correlation ID threaded from API request through retrieval
  and LLM call, so one request's journey is a single `grep`.

**Data engineering**

- Built a rate-limited **SEC EDGAR client** (token bucket under the published 10 req/s ceiling,
  full-jitter retry, `Retry-After` honoured, circuit breaker) landing filings into **MinIO**,
  Hive-partitioned and **idempotent** — a replayed date performs **zero writes**, asserted by
  counting them.
- Parsed inline-XBRL HTML into deterministic, versioned text where **every chunk round-trips its
  character offsets**, so an API citation resolves to a byte range in a named filing.
- Recovered financial-statement tables from markup where **59% of cells are layout spacers**,
  rebuilding column alignment that flat text extraction destroys.

**Retrieval & storage**

- Indexed **8,280 chunks** in **PostgreSQL**, serving **both** BM25 (full-text search) and dense
  retrieval (**pgvector**, HNSW, cosine) from one table so metadata pre-filtering is a `WHERE`
  clause rather than a cross-system join; **MongoDB** holds the nested scorecards.
- Benchmarked all three retrieval arms on a hand-labelled query set and **reported that hybrid
  RRF lost to dense alone** — metadata pre-filtering lifted precision@10 from **0.210 to 0.718**,
  a far larger effect than fusion.
- Cut re-indexing from **3,301s to 314s** by keying the **Redis** embedding cache on content hash
  rather than chunk ID: a chunker change that renumbered every chunk still hit **98.9%** of
  cached vectors.

**LLM systems & evaluation**

- Scored extractions against **XBRL ground truth** with a **six-way failure taxonomy** — exact,
  within-tolerance, scale error, wrong, hallucinated, abstained — rather than one number that
  hides how it fails.
- Built a **zero-API-cost context-coverage metric** ("was the answer even in the context?")
  separating retrieval failures from model failures, which made tuning viable against a free tier
  capped at **20 requests/day**.
- **Benchmarked two models on identical retrieval and prompts**: `gpt-oss-120b` 85.9% vs
  `gemini-3.5-flash` 80.3%, with 0 hallucinations vs 1.
- Wrote a **provider-agnostic LLM client** (Gemini + any OpenAI-compatible provider: Groq,
  OpenRouter, Cerebras) with one canonical JSON Schema and a converter to Gemini's dialect so the
  two cannot drift.
- Served it through **FastAPI** with full provenance in every response — chunk ID, Item, character
  offsets, EDGAR URL — at **150 ms p95** against an 800 ms SLO.

---

## Before the interview: three claims to be able to whiteboard

The risk with a project built fast is being unable to defend a choice you did not
personally agonise over. These three are the most likely to be probed, because
they read as the most senior. Full reasoning for all of them is in
[`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md).

**1. "Why 14.4× burn rate?"**
Error budget for `/search` is 5% (SLO: p95 < 800 ms). Burn rate 14.4 means the
error rate is 14.4 × 5% = 72%, so the alert fires when fewer than 28% of requests
are under 800 ms. The number is chosen so one hour of that burn consumes 2% of a
30-day budget — 1/720 × 14.4 = 2%. Fast enough to page, slow enough that a blip
does not. The 6× / 6h variant consumes 5% and files a ticket instead.

**2. "Why does the quality gate skip instead of fail?"**
A model below the accuracy floor is not a broken pipeline — the DAG noticed and
refused to publish, which is it working. Failing would page someone for a
regression already contained. Skipping keeps the previous good extractions,
still records the accuracy metric, and lets the alert fire on the metric rather
than on a red task. Failure should mean broken; this is not broken.

**3. "Isn't accepting alternative XBRL tags just gaming your own benchmark?"**
Only concepts the tag map **already declared for that field** are accepted, and
that map predates the failures. "Net income" is genuinely both `NetIncomeLoss`
(attributable to parent) and `ProfitLoss` (including non-controlling interests).
The same diagnostic found UnitedHealth's `total_assets` answer coincidentally
equalling its `Liabilities` — still scored wrong. Accepting a declared synonym
fixes the answer key; accepting anything that matches would fit the metric to
the results.

**And the one to volunteer before they find it:** n = 23 filings. Small enough
to find bugs, too small for confident comparisons. The limit is an LLM free
tier, not the pipeline — ingest and indexing run the full 55-company universe.
Saying this first is worth more than having it extracted.

---

## Interview anchors

Each found by measurement, not by reading code:

| Finding | What it signals |
|---|---|
| A ground-truth bug marked a **correct model wrong** for two filings | Skepticism toward your own evaluation, not just the model |
| **Published a correction to my own README** when a diagnostic disproved it | You report bad news about your own work |
| Hybrid RRF **lost** to dense alone | Willing to publish a negative result about your own design |
| Pre-filtering beat fusion **3.4×** | Knowing where the leverage actually is |
| A more explicit prompt made accuracy **worse** (62.5% → 18.8%) | Prompt changes need measurement, not intuition |
| `filings.recent` is capped, so a naive check flags the **biggest** filers | Reading API behaviour, not API docs |
| EDGAR returns **403, not 404**, for a holiday — would have broken every backfill | Found by running it, not by reading it |
| The inhibition rule **demonstrably suppressed** a real page, not just configured to | Alerting verified by running it, not by writing YAML |

---

## Claims to avoid

The accuracy figure is credible because of what it refuses to claim:

- **Don't quote 85.9% without n.** 23 filings, 92 extractions; one filing moves it ~1 point.
- **Qualify the orchestration.** The ingest DAG has run end to end against a live scheduler;
  the other two are parse-validated and wired but have not run a full pass.
- **Don't imply schema enforcement.** The final run used JSON mode with the schema described in
  the prompt, not enforced by the provider.
- **The model comparison is n=19 vs n=23**, not a like-for-like sample size.
- **Don't say you were paged.** The alerts fire and route by severity, and the inhibition rule
  is observed suppressing one — but no webhook, PagerDuty key or SMTP server is attached, so
  nothing left AlertManager. Say *"alerts fire and route"*, not *"I got paged"*.
- **The quality gate is unit-tested at its boundaries, not run in anger.** `extract_and_evaluate`
  has not completed a live scheduler pass, so describe the gate as implemented and tested rather
  than as having blocked a real publish.

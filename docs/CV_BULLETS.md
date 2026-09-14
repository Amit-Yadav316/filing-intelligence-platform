# CV bullets

Every number is measured and reproducible from this repository. Commands in the README.

---

## The one-liner

> **Filing Intelligence Platform** — an end-to-end SEC filing pipeline: Airflow · Docker Compose ·
> MinIO (S3) · PostgreSQL 16 + pgvector · Redis · MongoDB · FastAPI · Prometheus · Grafana ·
> AlertManager · GitHub Actions CI

---

## Short form (3 bullets, for a CV)

- Built and containerised an **8-service data platform with Docker Compose** — **Apache Airflow**
  orchestration, **MinIO** S3-compatible object storage, **PostgreSQL 16 + pgvector**, **Redis**,
  **MongoDB**, **FastAPI**, and a **Prometheus / Grafana / AlertManager** observability stack —
  that ingests SEC EDGAR filings, indexes 8,280 text chunks for hybrid BM25 + vector retrieval,
  and extracts financial facts with an LLM.
- Made the output **verifiable**: every extracted figure is scored against the XBRL ground truth
  the SEC publishes in the same filing, giving **86.4% field-level accuracy with zero
  hallucinations** across 88 extractions — and improved it from 55.4% in four diagnosed steps,
  **three of which cost no API calls**.
- Ran it like a production service: **3 SLOs with error budgets**, **burn-rate alerting** in
  Prometheus/AlertManager, a **provisioned Grafana dashboard** (version-controlled, not
  hand-clicked), an Airflow **quality gate** that blocks publishing when accuracy drops, and a
  **runbook** covering three failure modes that genuinely occurred. **207 tests, `mypy --strict`
  clean, 3-job CI pipeline.**

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
- Built a **Grafana dashboard as provisioned JSON in version control** — SLO stat panels,
  accuracy by field, latency percentiles, cache hit rate and cost — so it is an artefact of the
  repo, not of someone's browser.
- **AlertManager** routing splits `page` from `ticket` by severity, with an inhibition rule that
  suppresses the accuracy alert when the whole index is stale, since that is a downstream symptom.
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
- **Benchmarked two models on identical retrieval and prompts**: `gpt-oss-120b` 86.4% vs
  `gemini-3.5-flash` 80.3%, with 0 hallucinations vs 1.
- Wrote a **provider-agnostic LLM client** (Gemini + any OpenAI-compatible provider: Groq,
  OpenRouter, Cerebras) with one canonical JSON Schema and a converter to Gemini's dialect so the
  two cannot drift.
- Served it through **FastAPI** with full provenance in every response — chunk ID, Item, character
  offsets, EDGAR URL — at **150 ms p95** against an 800 ms SLO.

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

---

## Claims to avoid

The accuracy figure is credible because of what it refuses to claim:

- **Don't quote 86.4% without n.** 22 filings, 88 extractions; one filing moves it ~1 point.
- **Qualify the orchestration.** The ingest DAG has run end to end against a live scheduler;
  the other two are parse-validated and wired but have not run a full pass.
- **Don't imply schema enforcement.** The final run used JSON mode with the schema described in
  the prompt, not enforced by the provider.
- **The model comparison is n=19 vs n=22**, not a like-for-like sample size.

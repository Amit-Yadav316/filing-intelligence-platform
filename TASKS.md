# TASKS

Work top to bottom. Each task has an acceptance criterion. Do not start the next block until the current block's gate passes.

Commit after every green test. Tick boxes as you go so a fresh Claude Code session knows where it is.

**Rules that apply to every task**
- One class per prompt, with its contract stated before the code
- Tests before implementation for anything with a correctness claim
- No notebooks in the deliverable, `notebooks/` is gitignored
- No magic numbers, everything lives in `Settings`
- Instrument with Prometheus as you build, never retrofit
- Never pass DataFrames through Airflow XCom, pass paths and ids

---

## Day 0 — manual, before Claude Code (20 min)

- [ ] `git init`, drop in `CLAUDE.md`, `README.md`, `TASKS.md`, first commit
- [ ] Fetch `https://www.sec.gov/files/company_tickers.json`, pick 50 large-cap tickers across 8+ sectors, save CIK mapping to `config/universe.json`
- [ ] Decide the EDGAR User-Agent string, format `Name email@domain` — EDGAR blocks requests without one
- [ ] Get an LLM API key into `.env`, add `.env` to `.gitignore`
- [ ] **Spend 20 minutes on GitHub** searching `sec edgar airflow prometheus in:readme` and `10-K extraction SLO in:readme`, sorted by recently updated. Confirm the ops-layer gap is still real before committing the week

---

## Day 1 — Ingest

### Block 1.1 Scaffold
- [ ] `pyproject.toml` with pinned deps, `[project.optional-dependencies]` for `serve`, `dev`, `ml`
- [ ] `src/config/settings.py` as pydantic `BaseSettings`: EDGAR base URLs, user agent, rate limit, MinIO creds, Postgres DSN, Mongo URI, Redis URL, model names, all thresholds
- [ ] Package skeleton per the README layout, `__init__.py` everywhere
- [ ] `ruff.toml`, `pytest.ini`, `.pre-commit-config.yaml`, `Makefile` with `up`, `seed`, `demo`, `test`, `lint`
- [ ] **Gate**: `pytest` runs green on an empty suite, `python -c "from src.config.settings import Settings; Settings()"` resolves

### Block 1.2 EdgarClient
- [ ] `RateLimiter` class, token bucket, under 10 req/s, configurable
- [ ] `EdgarClient` with User-Agent header, retry with exponential backoff, timeout, circuit breaker after N consecutive failures
- [ ] Methods: `get_daily_index(date)`, `fetch_filing_document(cik, accession, filename)`, `get_company_facts(cik)`, `get_submissions(cik)`
- [ ] Tests against recorded fixtures in `tests/fixtures/`, not the live API
- [ ] **Gate**: 3 filings fetched by hand, rate limiter provably caps request rate under load test

### Block 1.3 The resolution probe — DO THIS BEFORE ANYTHING ELSE DOWNSTREAM
- [ ] `scripts/probe_xbrl_resolution.py`: for all 50 CIKs, pull `companyfacts`, attempt to resolve FY2023 `total_revenue`, `net_income`, `total_assets`, `operating_cash_flow`
- [ ] Try tag variants per field: `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet`, and equivalents for the others
- [ ] Output a table: field, resolved count, unresolved count, which tag matched, which CIKs failed
- [ ] **Gate**: resolution rate per field is printed. **If below 70 percent, stop and build `config/xbrl_tag_map.yaml` before continuing.** This number determines whether the whole evaluation premise works

### Block 1.4 Archival
- [ ] `ArchiveWriter`: writes raw HTML and XBRL JSON to MinIO at `cik={cik}/form={form}/filed={date}/accession={acc}/`
- [ ] Idempotent: re-running a partition overwrites exactly that partition, nothing else
- [ ] Writes a `_manifest.json` per partition with fetch timestamp, byte count, source URL, content hash
- [ ] **Gate**: one filing landed, re-run produces identical state, manifest present

### Block 1.5 Ingest DAG
- [ ] `docker-compose.yml` with MinIO and Postgres only at this point
- [ ] Astro CLI project, `astro dev start` works
- [ ] `ingest_edgar_filings` DAG: `@daily`, `catchup=True`, `start_date=2024-01-01`, tasks `discover` → `fetch` → `land` → `fetch_xbrl`
- [ ] Outlets `Dataset("s3://filings/raw")`
- [ ] **Day 1 gate**: backfill one week, ~20 filings plus XBRL landed and partitioned, rerun is idempotent, resolution probe number recorded in `docs/EVALUATION.md`

---

## Day 2 — Parse, chunk, tag

### Block 2.1 FilingParser
- [ ] HTML to clean text, strip nav, styling, page artefacts
- [ ] `TableExtractor`: detect `<table>` elements, serialise to markdown, preserve as atomic units, never split mid-table
- [ ] Split exhibits from the main document body
- [ ] Handle both modern inline-XBRL HTML and older plain HTML
- [ ] **Gate**: 3 filings parsed, tables visually verified against the EDGAR rendering

### Block 2.2 StructuralChunker
- [ ] Detect Item boundaries by regex on headings: `Item 1`, `Item 1A`, `Item 7`, `Item 7A`, `Item 8`, etc
- [ ] Sub-chunk long items with a token window plus overlap, configurable
- [ ] Tables become their own chunks, flagged `chunk_type="table"`
- [ ] Every chunk carries `chunk_id`, `accession`, `cik`, `form`, `filed`, `item_number`, `item_title`, `char_start`, `char_end`, `chunk_type`, `token_count`
- [ ] Test: chunk offsets round-trip, `text[char_start:char_end]` equals the chunk content
- [ ] **Gate**: offsets provably correct on 3 filings, no chunk spans an Item boundary

### Block 2.3 SectionTagger
- [ ] Rule-based map from Item number to canonical section label
- [ ] Fallback classifier only if the rules miss, do not over-engineer
- [ ] **Day 2 gate**: `make parse ACCESSION=<x>` emits chunks with full metadata to JSON, spot-checked against the source filing

---

## Day 3 — Embed, index, retrieve

### Block 3.1 EmbeddingService
- [ ] sentence-transformers `bge-small-en-v1.5`, batched
- [ ] `CacheKeyBuilder`: SHA256 of chunk text plus model name, Redis lookup before embedding
- [ ] **Gate**: re-running the same corpus embeds zero chunks, cache hit rate logged

### Block 3.2 Indexes
- [ ] Postgres schema: `chunks` table with metadata columns, `tsvector` column for BM25, `vector(384)` column for pgvector
- [ ] `BM25Index` using Postgres FTS with `ts_rank_cd`
- [ ] `VectorIndex` using pgvector with an HNSW index
- [ ] `MetadataFilter`: pre-filter by cik, form, fiscal_year, section before ranking
- [ ] **Gate**: both indexes return sane results for 5 manual queries

### Block 3.3 HybridRetriever
- [ ] **Port RRF fusion and BM25 logic from the financial-intelligence-agent repo**, copy the files by hand, do not fork the repo
- [ ] Strip the TF-IDF plus random-projection path and the SQLite engine entirely
- [ ] RRF with k=60, configurable
- [ ] **Gate**: hybrid returns results, latency logged

### Block 3.4 Labelled query set
- [ ] Hand-write 30 to 50 questions with known source sections, save to `tests/fixtures/query_set.json`
- [ ] `AblationRunner`: BM25 only vs dense only vs hybrid, reporting precision@10, MRR, p95 latency
- [ ] **Day 3 gate**: ablation table populated in `docs/EVALUATION.md`, reported honestly even if hybrid loses

---

## Day 4 — Extract and evaluate

### Block 4.1 Build the evaluator FIRST
- [ ] `XBRLResolver`: CIK plus fiscal year to a dict of ground-truth facts, using `config/xbrl_tag_map.yaml`
- [ ] Handles fiscal year end variation, restated values, units and scaling
- [ ] `ExtractionEvaluator.score()` returning one of `exact | within_tolerance | wrong | hallucinated | abstained | unresolvable`
- [ ] Tolerance configurable, default 0.5 percent
- [ ] **Gate**: evaluator scores a hand-written correct extraction and a hand-written wrong one correctly

### Block 4.2 ExtractionService
- [ ] `FilingExtraction` pydantic schema with `source_chunks`, `confidence`, `abstained_fields`
- [ ] Retrieval-grounded: pull relevant chunks, then extract, never send the whole filing
- [ ] Structured output enforced, retry once on schema violation, then abstain
- [ ] Redis cache on `(accession, schema_version, model)`
- [ ] **Gate**: extraction runs on 10 filings, output validates against the schema

### Block 4.3 The measurement
- [ ] Run extraction plus evaluation across the full sample corpus
- [ ] Produce the per-field accuracy table
- [ ] **Produce the prose-vs-table split** — this is the finding the architecture rests on
- [ ] Write a failure taxonomy: categorise every `wrong` case by cause
- [ ] **Day 4 gate**: both tables in `README.md` are populated with real numbers, and the routing decision they justify is written in one sentence

---

## Day 5 — Orchestrate

- [ ] `process_filings` DAG: `schedule=[raw_dataset]`, task group per stage, content-hash skip on unchanged chunks
- [ ] `extract_and_evaluate` DAG: `schedule=[chunks_dataset]`, extraction, resolution, scorecard, metrics push
- [ ] `ShortCircuitOperator` quality gate: block the Mongo write if batch accuracy falls below the SLO floor
- [ ] Dataset chaining verified: ingest publishes, process wakes, extract wakes
- [ ] Keep all logic in `src/`, DAG files thin
- [ ] **Day 5 gate**: full backfill over 30 days runs clean end to end, and deliberately degrading the extraction prompt causes the gate to block the write while the DAG still reports green

---

## Day 6 — Observe

### Block 6.1 Instrumentation
- [ ] `src/observability/metrics.py` with the counters, gauges and histograms from `CLAUDE.md`
- [ ] Structured JSON logging with a correlation id threaded from API request through retrieval, LLM call and response
- [ ] `/metrics` endpoint on the API, pushgateway for the Airflow tasks
- [ ] **Gate**: `curl localhost:8000/metrics` returns real series

### Block 6.2 Stack and alerting
- [ ] Add Prometheus, Grafana, AlertManager to `docker-compose.yml`
- [ ] Grafana dashboard as provisioned JSON in `deploy/grafana/`, not clicked by hand
- [ ] Panels: extraction accuracy by field, index freshness, search latency, LLM cost, task failure rate
- [ ] AlertManager rules on **error-budget burn** over a rolling window, not raw thresholds
- [ ] `docs/SLOS.md` with the three SLOs, targets and budgets
- [ ] `docs/RUNBOOK.md`: three failure modes (EDGAR rate-limited, LLM 429, embedding backlog) each with detection signal, first diagnostic, remediation
- [ ] **Day 6 gate**: break something on purpose and screenshot the alert firing

---

## Day 7 — Serve, ship, document

- [ ] FastAPI: `/search`, `/extract/{accession}`, `/ask`, `/health`, `/metrics`
- [ ] Every response carries the full provenance chain from `README.md`
- [ ] `/health` reports index freshness and the loaded model version
- [ ] GitHub Actions: ruff, mypy, pytest with a coverage gate, container build, smoke test posting a fixture to a started container
- [ ] Commit `data/sample/` so `make seed && make demo` works on clone
- [ ] **Stretch**: Helm chart in `deploy/helm/` applied to a local `kind` cluster. If it does not fit, say so in the README as future work rather than claiming it
- [ ] README: fill every `TBD`, add the architecture diagram, add the Grafana screenshot
- [ ] Write the CV bullets
- [ ] **Day 7 gate**: clone into a fresh directory, `make up && make seed && make demo`, one curl returns an answer with citations

---

## Cut order if behind

1. Helm and `kind`
2. MongoDB, fold extractions into a Postgres JSONB column
3. The learned section tagger, rules on Item headings are sufficient
4. The `/ask` endpoint, `/search` plus `/extract` carry the demo

**Never cut**: the resolution probe, the XBRL evaluator, the prose-vs-table split, chunk provenance offsets, Prometheus metrics, the SLOs, the runbook.

---

## Reference to read before day 3

`bhattaraisubal-eng/sec-intelligence-system` on GitHub. Closest prior art: pgvector, XBRL parsing, Redis caching, source attribution. Read it for the XBRL tag-mapping handling specifically, which will save a day. Do not copy its architecture wholesale — its differentiator is retrieval design, yours is the operations layer it does not have.

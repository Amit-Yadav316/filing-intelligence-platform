# CLAUDE.md

Project context for Claude Code. Read before writing any file.

---

## What we are building

**Filing Intelligence Platform.** An Airflow-orchestrated document pipeline that ingests SEC EDGAR filings into immutable object storage, parses and chunks them structurally, tags sections, embeds and indexes them for hybrid retrieval, extracts structured financial fields with an LLM, and **scores every extraction against XBRL ground truth from the same filing**. Telemetry, SLOs and alerting sit over the whole thing.

**Repo name**: `filing-intelligence-platform`

**Target**: BlackRock Aladdin Data, AI/ML Data Engineer. Every component below maps to a line in that job description. Nothing is here for decoration.

**Deadline**: 7 days.

---

## The thesis

Most GenAI portfolio projects cannot tell you whether they work. They report latency and a demo, because the output is free text with no ground truth.

SEC filings do not have that problem. EDGAR publishes the same filing twice: once as human-readable HTML that an analyst reads, and once as **XBRL** where the financial facts are machine-tagged. So when the LLM reads the MD&A prose and extracts "total revenue for FY2023", the tagged XBRL fact for `us-gaap:Revenues` is the answer key.

That gives you something almost no GenAI project has: **measured extraction accuracy, per field, with a failure taxonomy**. Exact-match rate, tolerance-band match rate, hallucination rate, abstention rate. Broken out by field type, filing size and company sector.

The README opens with that table. It is the difference between "I built a RAG pipeline" and "I built a RAG pipeline and here is how often it is wrong and why."

---

## Data

**Source**: SEC EDGAR. Free, no registration, no rate-limit key. Requires a descriptive `User-Agent` header with a contact email, and a self-imposed 10 requests per second ceiling. Respect both, in a rate-limited client class, and say so in the README — it reads as someone who has consumed a public API responsibly.

Three endpoints:

| Endpoint | Use |
|---|---|
| `https://www.sec.gov/cgi-bin/browse-edgar` or the daily index files | discover filings for a date range |
| `https://www.sec.gov/Archives/edgar/data/{cik}/...` | the filing documents themselves |
| `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` | **the ground truth**, every tagged fact for that company |

**Scope**: 10-K and 10-Q filings, 50 to 100 companies, 2022 to 2024. Roughly 600 to 1,200 filings. Enough to have a real index and a real accuracy denominator, small enough to reprocess in an afternoon when you change the chunker.

No licensing constraint. EDGAR is public domain, so a sample corpus can be committed and the repo runs on clone.

---

## Pipeline stages

```
   EDGAR API
       |
 [1 ingest]      raw HTML + XBRL JSON -> MinIO (S3 API), immutable,
                 partitioned cik=/form=/filed=/accession=
       |
 [2 parse]       HTML -> clean text, tables preserved, exhibits split
       |
 [3 chunk]       structural chunking on Item boundaries, then token-window
                 sub-chunking with overlap
       |
 [4 tag]         section classification: Risk Factors, MD&A, Financial
                 Statements, Legal Proceedings, Controls
       |
 [5 embed]       sentence-transformers -> pgvector
       |
 [6 index]       BM25 in Postgres FTS + vector index, metadata in Mongo
       |
 [7 extract]     LLM structured extraction -> pydantic schema
       |
 [8 evaluate]    score against XBRL companyfacts -> accuracy metrics
       |
 [9 serve]       FastAPI: /search (hybrid + RRF), /extract, /ask
       |
 [10 observe]    Prometheus metrics -> Grafana -> AlertManager
```

Stages 1 to 8 are Airflow tasks. Stages 9 and 10 run continuously.

---

## The parts that need care

### Chunking is structural first, fixed-window second

Filings have a legal structure: Item 1 Business, Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements. Naive 512-token windows cut across those boundaries and destroy retrieval quality.

Chunk on Item headings first, then sub-chunk long items with a token window and overlap. Carry `item_number`, `item_title`, `char_start`, `char_end` and `filing_accession` on every chunk as metadata, so retrieval can filter by section and every answer traces back to a character offset in a specific filing.

That provenance chain — answer, chunk, character offset, accession number, EDGAR URL — is the "archival" half of "document extraction and archival." Build it and show it in the API response.

### Tables are the hard part, and saying so is good

Financial statements are tables. HTML tables extracted as flat text become numeric soup and are the single biggest source of extraction error. Handle them separately: detect table elements at parse time, serialise them to markdown or a structured JSON representation, and chunk them as units rather than splitting them.

Report your table extraction quality separately from prose. "Extraction accuracy is 91 percent on prose fields and 68 percent on table-derived fields, here is why" is a far stronger finding than one blended number.

### Extraction is schema-constrained, never free text

```python
class FilingExtraction(BaseModel):
    fiscal_year: int
    total_revenue: Decimal | None
    net_income: Decimal | None
    total_assets: Decimal | None
    operating_cash_flow: Decimal | None
    reported_currency: str
    top_risk_categories: list[str]
    source_chunks: list[str]  # chunk ids that supported each field
    confidence: float
    abstained_fields: list[str]  # explicit "I could not find this"
```

**Abstention is a feature.** A model that says "not found" on a field it cannot locate is more useful than one that guesses, and your evaluation must reward it: an abstention is not a hallucination and should be scored in its own bucket.

### Evaluation against XBRL

```python
class ExtractionEvaluator:
    """Scores LLM extractions against XBRL companyfacts."""

    def resolve_ground_truth(self, cik: str, fiscal_year: int) -> dict: ...
    def score(self, extracted: FilingExtraction, truth: dict) -> ScoreCard:
        """exact_match | within_tolerance(0.5%) | wrong | hallucinated | abstained"""
```

The tag-mapping problem is real and worth documenting: `us-gaap:Revenues`, `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` and `us-gaap:SalesRevenueNet` all mean revenue depending on the filer and the year. Maintain a mapping config, and report how many filings you could not resolve ground truth for. That number is honest and interesting.

Output, per run:

| Field | Exact | Within 0.5% | Wrong | Hallucinated | Abstained | Resolvable |
|---|---|---|---|---|---|---|
| total_revenue | | | | | | |
| net_income | | | | | | |
| total_assets | | | | | | |

### Hybrid retrieval

Port your existing work. BM25 via Postgres full-text search, dense vectors via pgvector, fused with Reciprocal Rank Fusion. Report precision@10 on a labelled query set you build yourself (30 to 50 questions with known source sections is enough).

---

## Datastores, chosen to match the JD

| Store | Holds | Why |
|---|---|---|
| **MinIO** (S3 API) | raw filings, XBRL JSON, immutable | JD names S3 and Azure Blob; MinIO is the local S3 |
| **Postgres + pgvector** | chunks, embeddings, BM25 index | one engine for both retrieval modes |
| **MongoDB** | filing metadata, extraction results, eval scorecards | JD names MongoDB; semi-structured extraction output genuinely fits it |
| **Redis** | embedding cache, LLM response cache, rate-limit tokens | JD names Redis; caching embeddings cuts reprocessing cost by roughly 90 percent |

Four stores is not over-engineering here, because each holds something genuinely different and each is named in the JD. Be able to justify every one in a sentence. Redis in particular: re-running the pipeline after a chunker change should not re-embed unchanged chunks, and a content-hash cache key is how you avoid it.

---

## Telemetry and SRE (a full day, not an afterthought)

The JD says telemetry-based monitoring, alerting and incident response following SRE practice. That means SLOs, not dashboards.

**Instrument with `prometheus_client`:**

- `filings_ingested_total{form,status}` — counter
- `extraction_accuracy_ratio{field}` — gauge, updated per eval run
- `extraction_latency_seconds{stage}` — histogram, per pipeline stage
- `llm_tokens_total{model,direction}` and `llm_cost_usd_total` — counter
- `index_freshness_seconds` — gauge, now minus newest indexed filing
- `retrieval_latency_seconds` — histogram on `/search`
- `pipeline_task_failures_total{dag,task}` — counter

**Define three SLOs and write them in the README:**

| SLO | Target | Error budget |
|---|---|---|
| Extraction accuracy on `total_revenue` | ≥ 90% within tolerance | 10% |
| Index freshness | < 24h behind EDGAR | 1h/day |
| `/search` p95 latency | < 800 ms | 5% of requests |

**AlertManager rules** that fire on burn, not on raw thresholds: accuracy dropping below the SLO across a rolling window, freshness breach, latency budget exhaustion, task failure rate.

**Structured JSON logging** with a correlation id threaded from the API request through retrieval, LLM call and response, so one filing's journey is greppable.

Then write a short **runbook**: three failure modes (EDGAR rate-limited, LLM provider returning 429, embedding backlog growing) with detection signal, first diagnostic and remediation. Two pages. Almost nobody at campus level has written one, and it is the single clearest SRE signal you can produce.

---

## Airflow DAGs

**`ingest_edgar_filings`** — `@daily`, `catchup=True` from 2024-01-01. `logical_date` selects the EDGAR daily index to process. Idempotent: re-running a date overwrites exactly that partition. Outlets `Dataset("s3://filings/raw")`.

**`process_filings`** — `schedule=[raw_dataset]`. Parse, chunk, tag, embed, index. Task group per stage. Content-hash check against Redis skips unchanged work.

**`extract_and_evaluate`** — `schedule=[chunks_dataset]`. LLM extraction, XBRL ground truth resolution, scorecard, push metrics to Prometheus pushgateway. A `ShortCircuitOperator` blocks the extraction from being written to Mongo as authoritative if accuracy falls below the SLO floor — the quality gate.

Keep heavy code out of `dags/`. Airflow parses that directory on a loop. Logic lives in `src/`, invoked via `@task`.

---

## Stack, fixed

| Layer | Choice |
|---|---|
| Orchestration | Airflow (Astro CLI) |
| Object store | MinIO |
| Warehouse / vectors | Postgres 16 + pgvector |
| Document store | MongoDB |
| Cache | Redis |
| Parsing | BeautifulSoup, lxml, `sec-parser` if it fits |
| Embeddings | sentence-transformers `bge-small-en-v1.5`, local, free |
| LLM | Claude or GPT via API, with a local fallback path documented |
| Orchestration of LLM calls | LangChain for the extraction chain only, direct SDK elsewhere |
| Serving | FastAPI + uvicorn |
| Metrics | prometheus_client, Prometheus, Grafana, AlertManager |
| Container | docker-compose (day 6), Helm + kind (day 7 stretch) |
| CI | GitHub Actions: ruff, mypy, pytest, container build, smoke test |

**Not in scope**: Istio, Vault, ArgoCD, Terraform, Kafka, fine-tuning. Each is a week on its own. Do not claim them.

**On Kubernetes**: a `kind` cluster with a Helm chart for the API and worker deployments, a ConfigMap and a Secret, is honest and achievable. Write it if day 7 allows. If it does not, say in the README that the platform is containerised and K8s-ready with the chart as future work. An interviewer will respect that far more than a chart you never applied.

---

## Seven days

| Day | Build | Gate |
|---|---|---|
| 1 | Scaffold, rate-limited EDGAR client, ingest DAG, MinIO landing, XBRL companyfacts pull | 100 filings plus their XBRL facts landed, partitioned, idempotent on rerun |
| 2 | HTML parse, table handling, structural chunking on Item boundaries, section tagging | chunks carry item number and char offsets, spot-checked against 3 filings |
| 3 | Embeddings with Redis cache, pgvector index, BM25, hybrid RRF search, labelled query set | precision@10 measured on 30+ queries |
| 4 | LLM extraction with pydantic schema, XBRL ground-truth resolver, evaluator, scorecard | the per-field accuracy table is populated |
| 5 | Three Airflow DAGs, Dataset chaining, daily backfill, the accuracy quality gate | backfill runs clean, gate blocks a deliberately degraded extraction |
| 6 | Prometheus instrumentation, Grafana dashboard, AlertManager rules, SLOs, runbook | an alert fires when you break something on purpose |
| 7 | FastAPI polish with provenance in responses, CI pipeline, Helm + kind stretch, README | clone, `docker compose up`, one curl returns an answer with citations |

**Cut order if behind**: Helm and kind first, then Mongo (fold extraction results into Postgres), then the tagging classifier (rule-based on Item headings is fine).

**Never cut**: the XBRL evaluation harness, the provenance chain, the Prometheus metrics, the SLOs.

---

## Working notes for Claude Code

- Build the **evaluation harness before the extractor**. If you cannot score it, you cannot tune it, and you will waste days on prompt fiddling with no signal
- Vertical slice first: one filing, end to end, on day 1 to 4. Airflow multiplies it on day 5. Never orchestrate before the thing works once
- Class-based services with explicit contracts. `EdgarClient`, `FilingParser`, `StructuralChunker`, `EmbeddingService`, `HybridRetriever`, `ExtractionService`, `ExtractionEvaluator`. No procedural scripts, no notebooks in the deliverable
- Every external call goes through a class with retry, timeout and a circuit breaker. The JD says incident response; code that degrades gracefully is what that means in practice
- Cache aggressively on content hash. You will reprocess the corpus a dozen times as the chunker changes
- `Settings` as a pydantic `BaseSettings`. No magic numbers, no hardcoded model names, no API keys in code
- Instrument as you build, not on day 6. A counter added while writing the function costs nothing; retrofitting metrics across ten modules costs half a day

---

## README structure

1. **The problem** — filings are unstructured, extraction is unverifiable, XBRL makes it verifiable
2. **Accuracy table** — per field, exact / tolerance / wrong / hallucinated / abstained
3. **Architecture diagram** — the ten-stage pipeline
4. **Provenance demo** — a question, the answer, the chunk, the character offset, the EDGAR link
5. **SLOs and a Grafana screenshot** with an alert firing
6. **Runbook** — three failure modes
7. **What I did not build and why** — Istio, Vault, Kafka, fine-tuning, with one honest line each
8. **Run it** — `docker compose up`, one curl

Lead with the accuracy table. Every other GenAI project leads with the demo.

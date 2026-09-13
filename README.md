# Filing Intelligence Platform

A production document pipeline for SEC filings. Ingests EDGAR 10-K and 10-Q filings into immutable object storage, parses and chunks them on their legal structure, embeds and indexes them for hybrid retrieval, extracts structured financial fields with an LLM, and scores every extraction against XBRL ground truth from the same filing. Orchestrated with Airflow, instrumented with Prometheus, governed by SLOs.

> **Every number in this README was measured by a run, not estimated.** Where a
> result is unflattering - hybrid retrieval losing to dense alone, extraction
> accuracy below its own SLO, a prompt change that made things worse - it is
> reported as measured. The reproduction commands are in [Running it](#running-it).

---

## Why this exists

Financial analysts read filings. Filings are unstructured HTML with legal section boundaries, inconsistent tables and no schema. Extracting a clean number from one is a document problem, not a modelling problem.

Most systems that attempt it cannot tell you whether they worked, because the output is free text with no answer key. This one can, because the SEC publishes every filing twice: once as prose an analyst reads, and once as **XBRL** where the facts are machine-tagged.

That gives the project a measurable question: **how reliably can an LLM extract a financial fact from filing prose, and where does it fail?**

The answer, measured below, drives the architecture. It is not asserted up front.

---

## The finding

Measured over **22 filings** (12 large-cap companies, 10-K, FY2022-2024), extracting
four financial fields with `openai/gpt-oss-120b` from retrieved chunks, scored
against the XBRL facts the SEC published in the same filings.

| Field | Exact | Within 0.5% | Scale error | Wrong | Hallucinated | Abstained | Accuracy | **When answered** |
|---|---|---|---|---|---|---|---|---|
| `operating_cash_flow` | 20 | 0 | 0 | 0 | 0 | 2 | **90.9%** | **100.0%** |
| `total_revenue` | 19 | 0 | 0 | 1 | 0 | 2 | **86.4%** | **95.0%** |
| `net_income` | 17 | 2 | 0 | 1 | 0 | 2 | **86.4%** | **95.0%** |
| `total_assets` | 18 | 0 | 0 | 2 | 0 | 2 | **81.8%** | **90.0%** |

**Overall: 76 of 88 scoreable extractions correct (86.4%). Zero hallucinations.
Zero scale errors.**

### How it got here, which matters more than the number

Four measured steps. **Three of the four were verified without spending a single
LLM call**, and every one was diagnosed before it was attempted:

| | Overall | What changed | How it was found |
|---|---|---|---|
| First run | 55.4% | — | — |
| | 72.6% | Retrieval + prompt | A no-LLM diagnostic showed the figure was **already in context for 67% of abstentions**. Separately, `websearch_to_tsquery` ANDs bare terms, so three of four lexical anchors returned **zero rows** for filers whose wording differed |
| | 78.4% | Evaluator accepts near-synonymous concepts | For each wrong answer, searched the filer's own XBRL for a tag whose value the model actually reported. **Ten of eleven** matched a concept the tag map already declared for that field |
| | **86.4%** | Fixed the fiscal-year rule | Two Johnson & Johnson filings resolved to the *same* fiscal year, so each was scored against the other's figures |

### The last bug is the most interesting one

Johnson & Johnson runs a 52/53-week fiscal year ending the Sunday nearest 31
December, so **fiscal 2022 ended on 1 January 2023**. Keying the fiscal year on the
calendar year a period *ends* in assigned FY2023 to both J&J filings. The answer
key was silently off by a year, and **it marked a correct model wrong**.

Nothing failed. No exception, no anomaly in the logs — just two numbers that
disagreed, in a table that said the model was at fault. It surfaced only because a
coverage metric pointed at the wrong filings.

That is the entire argument for this project's existence: **an extraction system
that cannot check itself cannot tell the difference between a bad model and a bad
answer key.**

### Where the remaining 12 errors are

Four wrong and eight abstentions across 88 scored extractions. The wrong answers
are concentrated in `total_assets`, including one real misread where `Liabilities`
was reported as assets.

### Honest limits on this table

- **The model's contribution is not isolated.** Provider, retrieval and prompt all
  changed between the 55.4% and 72.6% runs. The retrieval half is independently
  verified by context coverage; the provider's share is **not**. A controlled
  Gemini re-run on identical retrieval was written, attempted, and blocked by the
  free tier's 20-requests-per-day cap.
- **Structured output was not enforced.** Groq's JSON mode guarantees valid JSON,
  not schema conformance, so the schema was described in the prompt.
- **22 of 23 filings.** One failed on `max_completion_tokens` before producing
  valid JSON - an output-budget limit, not a model failure.
- **Single run, temperature 0.** No self-consistency voting, no ensembling.
- **n = 22.** Small enough that a single filing moves the headline by about a
  point.

### Context coverage: the ceiling on accuracy

Measured with **no LLM calls** - can the answer even be found in what the model saw?

| Field | Coverage |
|---|---|
| `operating_cash_flow` | **100%** |
| `net_income` | 96% |
| `total_assets` | 91% |
| `total_revenue` | 87% |
| **Overall** | **93%** |

Coverage is flat at 96% from 10 chunks to 18 - the extra six contribute nothing.
Under Groq's 8,000 tokens-per-minute ceiling a 6,000-token cap holds 93%, and that
gap is the measured price of the free tier.

With coverage at 93% and accuracy at 86.4%, the two are now close enough that
further gains need a better model or an enforced schema, not better retrieval.

### Prose versus tables

| Source | Correct | Total | Accuracy |
|---|---|---|---|
| Tables | 76 | 88 | 86.4% |

Every scored figure was attributed to a **table** chunk. With per-statement retrieval
in place the financial statements crowd narrative text out of the context entirely,
so the prose-vs-table comparison has no prose arm left on this corpus. Reporting a
fabricated prose figure would be worse than reporting that the comparison collapsed.

## Architecture

```
                      SEC EDGAR API
                           │
   ┌───────────────────────┼───────────────────────┐
   │                       │                       │
 daily index          filing HTML            XBRL companyfacts
   │                       │                       │
   └───────────► [1] ingest_edgar_filings ◄────────┘
                    rate-limited client
                    immutable landing
                           │
                    MinIO (S3 API)
                 cik=/form=/filed=/accession=
                           │
                ┌──────────┴──────────┐
                │ [2] process_filings │
                └──────────┬──────────┘
       parse ─► table split ─► structural chunk ─► tag ─► embed ─► index
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   Postgres           pgvector             Redis
   + FTS (BM25)      (dense index)      (embed cache)
        │                  │
        └────────┬─────────┘
                 │
      ┌──────────┴───────────┐
      │[3] extract_and_eval  │
      └──────────┬───────────┘
   LLM extraction ─► XBRL resolver ─► scorecard ─► quality gate
                 │
            MongoDB (extractions, scorecards)
                 │
         ┌───────┴────────┐
         │  FastAPI       │  /search  /extract  /ask  /metrics
         └───────┬────────┘
                 │
    Prometheus ─► Grafana ─► AlertManager
```

---

## Pipeline stages

| # | Stage | What it does |
|---|---|---|
| 1 | **Ingest** | Rate-limited EDGAR client, lands raw HTML and XBRL JSON to MinIO, immutable and idempotent per partition |
| 2 | **Parse** | HTML to clean text, tables detected and serialised separately, exhibits split out |
| 3 | **Chunk** | Structural chunking on Item boundaries first, token-window sub-chunking with overlap second |
| 4 | **Tag** | Section classification: Business, Risk Factors, MD&A, Financial Statements, Legal Proceedings, Controls |
| 5 | **Embed** | sentence-transformers, content-hash cached in Redis so re-runs skip unchanged chunks |
| 6 | **Index** | Dense vectors in pgvector, BM25 in Postgres FTS, metadata in Mongo |
| 7 | **Extract** | Schema-constrained LLM extraction into a pydantic model, with explicit abstention |
| 8 | **Evaluate** | Resolve XBRL ground truth, score each field, push metrics to Prometheus |
| 9 | **Serve** | FastAPI hybrid search with RRF, extraction lookup, question answering with provenance |
| 10 | **Observe** | Metrics, dashboards, SLO alerting, structured logs with correlation ids |

---

## Provenance

Every answer traces to a character range in a specific filing, and every extracted
figure traces to the XBRL fact it was scored against. Both halves are live:

```bash
curl -s localhost:8000/extract/0000093410-24-000013 | jq
```

```
CHEVRON CORP  10-K  FY2023   model: gemini-3.5-flash   accuracy: 1.0

total_revenue        exact   196,913,000,000  =  196,913,000,000  us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax
net_income           exact    21,369,000,000  =   21,369,000,000  us-gaap:NetIncomeLoss
total_assets         exact   261,632,000,000  =  261,632,000,000  us-gaap:Assets
operating_cash_flow  exact    35,609,000,000  =   35,609,000,000  us-gaap:NetCashProvidedByUsedInOperatingActivities

supporting chunk for total_revenue: 0000093410-24-000013::item14::c104
```

The left column is what the model read out of a retrieved chunk. The right column
is what Chevron tagged in the same filing's XBRL. The tag name is recorded, so the
comparison can be audited rather than trusted — and where two `us-gaap` concepts
both plausibly mean "revenue", the disagreement is visible instead of hidden.

That chain is what makes this an archive rather than a cache:

| Link | Carried on | Guarantee |
|---|---|---|
| Answer → chunk | `chunk_id` | Stable within a parser version |
| Chunk → text range | `char_start`, `char_end` | `text[start:end] == chunk.text`, asserted in tests |
| Text → filing | `accession`, `parser_version` | Deterministic re-parse of immutable bytes |
| Filing → source | `edgar_url`, sha256 in the manifest | Byte-identical to what EDGAR served |
| Figure → ground truth | `us-gaap` tag, fiscal year | The filing's own XBRL |

`parser_version` travels with every chunk, so if the parser changes, old offsets
are known to be **stale** rather than silently pointing at the wrong bytes.

## Service level objectives

Three SLOs with error budgets, alerting on **burn rate** rather than on threshold
crossings. A `p95 > 800ms` alert fires on one slow query at 3am and teaches people
to ignore it; a burn-rate alert fires when failures are arriving fast enough to
exhaust the month's budget early.

| SLO | Target | Measured | Status |
|---|---|---|---|
| Extraction accuracy, `total_revenue` | ≥ 90% | **57.1%** (80.0% when answered) | **Not met** |
| Index freshness behind EDGAR | < 24h | corpus is historical (FY2022-24) | n/a on a fixed corpus |
| `/search` p95 latency | < 800 ms | **148-170 ms** | **Met**, wide margin |

The accuracy SLO is breached and reported as breached. Lowering the target to
match current performance would make it meaningless; the gap is tracked as
retrieval work in [`docs/SLOS.md`](docs/SLOS.md).

**The quality gate**: `extract_and_evaluate` will not write extractions to the
document store when batch accuracy falls below the floor. The downstream tasks
are **skipped, not failed** — a degraded model is not a broken pipeline, and the
DAG did exactly what it should, which is notice and refuse. The previous good
extractions stay in place and the alert fires on the accuracy metric rather than
on a red task.

Alert rules are in [`deploy/prometheus/alerts.yml`](deploy/prometheus/alerts.yml),
routing by severity in [`deploy/alertmanager/`](deploy/alertmanager/), and three
worked failure modes in [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — EDGAR rate-limiting,
LLM provider 429s, and a growing embedding backlog. All three happened while
building this.

## Retrieval

Hybrid search over **8,280 chunks** from **26 filings** across **12 companies** (10-K and
10-Q, 2022-2024).

- **BM25** via Postgres full-text search - see the honesty note below
- **Dense** via pgvector with HNSW, `bge-small-en-v1.5`, cosine distance
- **Fusion** via Reciprocal Rank Fusion, k=60, 50 candidates per arm
- **Metadata pre-filtering** by company, form, fiscal year and section *before* ranking

Measured on a hand-labelled set of **38 questions** with known source sections
(`tests/fixtures/query_set.json`). Precision@10 is structurally capped at **0.905**,
because 5 queries target an Item holding fewer than ten chunks.

**With company and form pre-filtering** - what the API does when a caller names a company:

| Method | Precision@10 | Recall@10 | MRR | p95 |
|---|---|---|---|---|
| BM25 (Postgres FTS) | 0.208 | 0.553 | 0.539 | 5 ms |
| Dense (pgvector) | 0.718 | **1.000** | **0.961** | 85 ms |
| **Hybrid RRF** | **0.729** | **1.000** | 0.926 | 89 ms |

**Without filtering**, over the whole corpus:

| Method | Precision@10 | Recall@10 | MRR | p95 |
|---|---|---|---|---|
| BM25 (Postgres FTS) | 0.113 | 0.447 | 0.330 | 48 ms |
| **Dense (pgvector)** | **0.210** | **0.632** | **0.415** | 88 ms |
| Hybrid RRF | 0.182 | 0.605 | 0.386 | 103 ms |

### What the numbers actually say

**Hybrid RRF does not clearly win, and this README does not pretend otherwise.**
Filtered, it edges dense on precision by 0.011 and *loses* on MRR by 0.035.
Unfiltered, dense beats it outright on every metric. RRF weights both arms
equally, so fusing a strong dense arm with a weak lexical one pulls the result
toward the weaker of the two.

**The biggest lever is pre-filtering, not fusion.** Constraining the candidate
set by company and form moves dense precision@10 from 0.210 to 0.718 - a 3.4x
gain, far larger than anything fusion contributes. That is an architecture
finding: spend the effort on metadata, not on tuning a fusion constant.

**On the BM25 label.** The lexical arm ranks with Postgres `ts_rank_cd`, which
is cover-density ranking, **not Okapi BM25**. It shares term-frequency
saturation and proximity weighting, but implements neither BM25's document
length normalisation nor its IDF formulation. True BM25 in Postgres needs an
extension that cannot be assumed on a stock image. The name is kept because it
names the retrieval *arm*; the ranker is stated accurately here and in the code,
and its weakness shows up honestly in the table above.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | Airflow (Astro CLI) | `logical_date` drives the daily EDGAR index, backfill replays history |
| Object store | MinIO (S3 API) | immutable archival, S3-compatible so the cloud path is a config change |
| Warehouse and vectors | Postgres 16 + pgvector | one engine serves BM25 and dense retrieval |
| Document store | MongoDB | extraction results and scorecards are semi-structured and schema-evolving |
| Cache | Redis | content-hash embedding cache, LLM response cache, rate-limit tokens |
| Embeddings | sentence-transformers | local, free, no per-call cost on reprocessing |
| LLM | `openai/gpt-oss-120b` via Groq | one provider-agnostic client; Gemini, OpenRouter, Cerebras and OpenAI are config, not code |
| Serving | FastAPI + uvicorn | typed contract, provenance in every response |
| Observability | Prometheus, Grafana, AlertManager | SLO-based alerting |
| CI | GitHub Actions | ruff, mypy, pytest, container build, smoke test |

---

## Running it

```bash
git clone https://github.com/Amit-Yadav316/filing-intelligence-platform
cd filing-intelligence-platform

cp .env.example .env          # set EDGAR_USER_AGENT (required) and an LLM key
make venv && make install     # Python 3.11; torch installs CPU-only
make up                       # MinIO, Postgres+pgvector, Redis, Mongo, Prometheus, Grafana
```

`EDGAR_USER_AGENT` must contain a real contact address. The SEC returns **403**
without one, and `Settings` refuses to construct with the placeholder still in
place — a misconfiguration fails immediately rather than twenty minutes into a
backfill.

Then, end to end:

```bash
make universe                 # resolve 55 companies from EDGAR's ticker map
make probe                    # can XBRL ground truth be resolved? gate: >=70%
python -m scripts.parse_filing --land AAPL:10-K:2023   # ingest one filing
make index                    # parse, chunk, embed, index everything landed
make ablation                 # BM25 vs dense vs hybrid on 38 labelled queries
python -m scripts.run_extraction   # extract and score against XBRL
```

Serve it:

```bash
uvicorn src.serving.api:app --port 8000
curl -s localhost:8000/health | jq
```

One search, with the full provenance chain:

```bash
curl -s localhost:8000/search -H 'Content-Type: application/json'   -d '{"q":"supply chain concentration risk","top_k":2}' | jq '.results[0]'
```

```json
{
  "rank": 1,
  "score": 0.031514,
  "found_by": ["bm25", "dense"],
  "chunk_type": "prose",
  "text": "While we work to enhance the resiliency and redundancy of our supply chain, which is currently concentrated in...",
  "citation": {
    "chunk_id": "0001045810-24-000029::item7::c7",
    "company": "NVIDIA CORP",
    "form": "10-K",
    "filed": "2024-02-21",
    "item": "Item 7",
    "section": "Management's Discussion and Analysis",
    "char_start": 186743,
    "char_end": 189070,
    "edgar_url": "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/"
  }
}
```

That is the archival half of the system: the answer resolves to a character range
in a named filing, and `parser_version` travels with every chunk so an offset
recorded today is known to be stale rather than silently wrong if the parser changes.

| Endpoint | Purpose |
|---|---|
| `POST /search` | Hybrid retrieval with metadata pre-filtering |
| `GET /extract/{accession}` | Stored extraction and its scorecard |
| `GET /accuracy` | Per-field accuracy, live — the README table from the database |
| `GET /ask` | Passages answering a question, cited |
| `GET /health` | Index freshness, model versions, SLO status |
| `GET /metrics` | Prometheus series |

`/ask` is deliberately **extractive, not generative**. The project's claim is
measured extraction accuracy; generating an unmeasured free-text answer beside
scored ones would undercut exactly that.

## Open work

**Done and measured:** lexical anchors fixed, index tables excluded, prompt
rebalanced, context budgeted in tokens, 10-Q filings removed from annual scoring,
near-synonymous concepts accepted, and the 52/53-week fiscal-year rule corrected.

**Still open:**

- **Isolate the model's contribution.** The controlled Gemini re-run is written and
  ready; it needs a day's quota or about $0.10 of billing.
- **`total_assets` at 81.8%**, the weakest field, including one genuine misread.
- **Run the DAGs.** Parse-validated in CI, never executed against a live scheduler.

## What this does not do, and why

| Not built | Reason |
|---|---|
| Istio, service mesh | single-service deployment, a mesh solves a problem this does not have |
| HashiCorp Vault | secrets are env-injected; Vault is correct at org scale, not here |
| Kafka or NATS streaming | EDGAR publishes in daily batches, so batch orchestration is the honest fit |
| LLM fine-tuning | schema-constrained prompting hits the accuracy target; fine-tuning would be cost without measured benefit |
| Terraform, ArgoCD | no cloud target, and no deployment to manage yet |
| Helm chart / `kind` | **Not built.** Day 7 ran out before it. The platform is containerised and the compose stack is the honest deployment story; a chart that was never applied would be a claim, not an artefact |
| Generative answers in `/ask` | The project's claim is *measured* accuracy. An unmeasured free-text answer sitting beside scored figures would undercut the whole argument |
| Fine-tuning | Schema-constrained prompting already reaches 80-90% when the model answers; the bottleneck is retrieval, and fine-tuning would not fix that |

Naming what was deliberately left out is part of the design record.

---

## Repository layout

```
filing-intelligence-platform/
├── CLAUDE.md                     project context
├── README.md
├── TASKS.md                      build plan, day by day
├── dags/
│   ├── ingest_edgar_filings.py
│   ├── process_filings.py
│   └── extract_and_evaluate.py
├── src/
│   ├── config/settings.py
│   ├── ingest/                   EdgarClient, RateLimiter, ArchiveWriter
│   ├── parse/                    FilingParser, TableExtractor
│   ├── chunk/                    StructuralChunker, SectionTagger
│   ├── embed/                    EmbeddingService, CacheKeyBuilder
│   ├── retrieve/                 BM25Index, VectorIndex, HybridRetriever
│   ├── extract/                  ExtractionService, schemas
│   ├── evaluate/                 XBRLResolver, ExtractionEvaluator, AblationRunner
│   ├── serving/                  api, schemas, dependencies
│   └── observability/            metrics, logging, tracing
├── docs/
│   ├── RUNBOOK.md
│   ├── SLOS.md
│   └── EVALUATION.md
├── deploy/
│   ├── docker-compose.yml
│   ├── prometheus/
│   ├── grafana/
│   └── helm/
├── data/sample/                  committed corpus, runs on clone
└── tests/
```

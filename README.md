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

Measured over **14 filings** (12 large-cap companies, 10-K and 10-Q, FY2022-2024),
extracting four financial fields with `gemini-3.5-flash` from retrieved chunks,
scored against the XBRL facts the SEC published in the same filings.

| Field | Exact | Within 0.5% | Scale error | Wrong | Hallucinated | Abstained | Accuracy | **When answered** |
|---|---|---|---|---|---|---|---|---|
| `total_assets` | 9 | 0 | 0 | 1 | 0 | 4 | 64.3% | **90.0%** |
| `operating_cash_flow` | 8 | 0 | 0 | 1 | 0 | 5 | 57.1% | **88.9%** |
| `total_revenue` | 8 | 0 | 0 | 2 | 0 | 4 | 57.1% | **80.0%** |
| `net_income` | 6 | 0 | 0 | 3 | 0 | 5 | 42.9% | **66.7%** |

**Overall: 31 of 56 scoreable extractions correct (55.4%).** Total cost: $0.06.

### What the numbers actually say

**Zero hallucinations. Zero scale errors.** Across 56 scored extractions the model
never invented a figure and never mis-scaled one, which is the failure mode that
makes an extraction system unusable. Every error is a *plausible* wrong answer.

**The gap between 55% and 80-90% is abstention.** When the model commits to a
figure it is right 80-90% of the time; it declined roughly 30% of the time.

An earlier version of this section attributed that abstention to retrieval. **That
was wrong, and a measurement corrected it.** A diagnostic that costs no LLM calls
(`scripts/diagnose_abstentions.py`) checked, for every abstained field, whether the
ground-truth figure was actually present in the chunks the model was shown:

| | Share | Diagnosis |
|---|---|---|
| Figure **was** in context | 67% | Prompt problem - the model had the number and declined |
| Figure was **not** in context | 33% | Retrieval problem - abstaining was correct |

Both causes were then fixed, and the retrieval half is verified below. The numbers
in the table above predate both fixes; see [Open work](#open-work).

**Almost every remaining error is a definitional ambiguity, not a mistake.**
The seven `wrong` verdicts are nearly all cases where two defensible answers exist:

| Company | Extracted | XBRL truth | What actually differs |
|---|---|---|---|
| ExxonMobil | 398,675 | 413,680 | Operating revenue vs. revenue *including other income* |
| UnitedHealth | 20,639 | 20,120 | Net earnings vs. net earnings *attributable to the parent* |
| Procter & Gamble | 14,738 | 14,653 | Same non-controlling-interest distinction |
| Johnson & Johnson | 35,153 | 17,941 | Total vs. continuing operations, around the Kenvue separation |

These are the tag-mapping problem showing up from the other side. `us-gaap` offers
several near-synonymous concepts, the answer key picks one, and the model picks
another equally correct one. Counting them as errors is the honest choice - but
calling them "model failures" would not be.

**One thing that did not work, reported because it is informative.** Rewriting the
prompt to spell out these distinctions explicitly made accuracy *worse*, from 62.5%
to 18.8% on the same sample: the model responded to the extra constraint by
abstaining rather than by answering more precisely. The terse prompt is kept for
that measured reason, and the finding is recorded in the source.

### Prose versus tables

| Source | Correct | Total | Accuracy |
|---|---|---|---|
| Tables | 31 | 56 | 55.4% |

Every scored figure was attributed to a **table** chunk, not prose - the model cited
statement tables for all four fields in all 14 filings. That is itself a result: with
per-statement retrieval in place, the financial statements crowd narrative text out
of the context entirely, so a prose-vs-table comparison has no prose arm left to
measure on this corpus. Reporting a fabricated prose figure would be worse than
reporting that the comparison collapsed.

### Honest limits on this table

- **14 filings, not the full 26.** The Gemini free tier caps `generateContent` at
  **20 requests per day per model**; the corpus needs 26 plus retries. The remaining
  filings are ingested, chunked, embedded and indexed - only the LLM call is
  outstanding. Re-running with billing enabled, or across two days, completes it.
- **Section-level, single-run.** Each filing was extracted once at temperature 0;
  no self-consistency voting, no ensembling.

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
| LLM | Any of Gemini, Groq, OpenRouter, Cerebras, OpenAI | one provider-agnostic client; swapping is config, not code |
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

The accuracy table above predates two fixes that are already in the code. Both
were driven by measurements that cost no LLM calls, which matters because the
free-tier quota is 20 requests per day and tuning against it directly would take
weeks.

**Retrieval is fixed and verified.** `scripts/measure_context_coverage.py` asks
whether the ground-truth figure is present in the chunks the model is shown —
the ceiling on accuracy, since nothing can report a number it never saw:

| Field | Coverage |
|---|---|
| `net_income` | **100%** |
| `operating_cash_flow` | **100%** |
| `total_revenue` | 91% |
| `total_assets` | 91% |
| **Overall** | **96%** across 23 10-K filings |

Two bugs caused the earlier misses. `websearch_to_tsquery` ANDs bare terms, so
the anchor *"net cash provided by operating activities"* matched nothing for a
filer writing *"Cash generated by operating activities"* — three of four anchors
returned zero rows for Apple while its statements sat in the index. And *"Index
to Consolidated Financial Statements"* was crowding the statements out: it names
every statement and contains no figures, so it is the perfect lexical match for a
query naming one. The statements carry no title inside the `<table>`, so they can
only be found by line item.

**The prompt is rebalanced** to narrow abstention to genuinely absent figures,
with the two rejected earlier versions and their measured effects recorded in the
source.

**What has not been re-measured is accuracy itself.** That needs one full
extraction run, and the day's quota is spent. The expectation is a substantial
rise, because 67% of abstentions were the prompt declining on figures it had and
the rest were retrieval misses now closed — but it is an expectation, not a
result, and this README does not print expectations as results.

**Scope correction.** The corpus's three 10-Q filings were being scored against
*annual* XBRL facts. A quarterly report cannot contain a full-year figure, so the
model was marked wrong for correctly not finding one; all three scored 0/4 or 1/4
on coverage while every 10-K scored 4/4. Annual fields are now scored against the
23 10-K filings only.

---

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

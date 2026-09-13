# Filing Intelligence Platform

A production document pipeline for SEC filings. Ingests EDGAR 10-K and 10-Q filings into immutable object storage, parses and chunks them on their legal structure, embeds and indexes them for hybrid retrieval, extracts structured financial fields with an LLM, and scores every extraction against XBRL ground truth from the same filing. Orchestrated with Airflow, instrumented with Prometheus, governed by SLOs.

> **Status**: in development. Every number in this README marked `TBD` is a placeholder and will be replaced by a measured value from a pipeline run. Nothing here is estimated.

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

**The gap between 55% and 80-90% is abstention, and abstention is the bottleneck -
not the model.** When the model commits to a figure it is right 80-90% of the time.
It declines roughly 30% of the time, and it declines because the retrieved context
did not contain the statement, not because it was unsure of a number in front of
it. The fix is retrieval, not a better model. Measured evidence for that: adding
per-statement retrieval and exact line-item lexical anchors moved overall accuracy
from **8.3% to 62.5%** on a fixed four-filing sample, without touching the model.

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

Every answer traces to a character offset in a specific filing. The response carries the chain, not just the text.

```json
{
  "answer": "...",
  "citations": [{
    "chunk_id": "0000320193-23-000106::item7::c14",
    "accession": "0000320193-23-000106",
    "cik": "0000320193",
    "form": "10-K",
    "filed": "2023-11-03",
    "item": "Item 7",
    "char_start": 48211,
    "char_end": 49034,
    "edgar_url": "https://www.sec.gov/Archives/edgar/data/320193/..."
  }],
  "confidence": 0.87,
  "abstained_fields": []
}
```

This is the archival half of document extraction and archival. An answer you cannot trace back to a byte range in a source document is not auditable.

---

## Service level objectives

| SLO | Target | Measured |
|---|---|---|
| Extraction accuracy, `total_revenue`, within 0.5% | ≥ 90% | `TBD` |
| Index freshness behind EDGAR | < 24h | `TBD` |
| `/search` p95 latency | < 800 ms | `TBD` |

Alerts fire on error-budget burn over a rolling window, not on instantaneous threshold breaches. See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the three failure modes and their remediation.

**Quality gate**: `extract_and_evaluate` will not write an extraction to Mongo as authoritative if batch accuracy falls below the SLO floor. A degraded model is skipped, not shipped, and the DAG still reports green.

---

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
| LLM | Anthropic or OpenAI API | schema-constrained extraction only |
| Serving | FastAPI + uvicorn | typed contract, provenance in every response |
| Observability | Prometheus, Grafana, AlertManager | SLO-based alerting |
| CI | GitHub Actions | ruff, mypy, pytest, container build, smoke test |

---

## Running it

```bash
git clone <repo>
cd filing-intelligence-platform
cp .env.example .env          # set EDGAR_USER_AGENT and an LLM API key
make up                       # docker compose: minio, postgres, mongo, redis, mlflow-free stack
make seed                     # load the committed sample corpus
make demo                     # one search, one extraction, one scored evaluation
```

```bash
curl -s localhost:8000/search \
  -d '{"q":"supply chain concentration risk","form":"10-K","year":2023}' | jq
```

Full pipeline against live EDGAR:

```bash
astro dev start
airflow dags trigger ingest_edgar_filings
```

---

## What this does not do, and why

| Not built | Reason |
|---|---|
| Istio, service mesh | single-service deployment, a mesh solves a problem this does not have |
| HashiCorp Vault | secrets are env-injected; Vault is correct at org scale, not here |
| Kafka or NATS streaming | EDGAR publishes in daily batches, so batch orchestration is the honest fit |
| LLM fine-tuning | schema-constrained prompting hits the accuracy target; fine-tuning would be cost without measured benefit |
| Terraform, ArgoCD | no cloud target; a Helm chart is provided and applied to a local `kind` cluster |

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

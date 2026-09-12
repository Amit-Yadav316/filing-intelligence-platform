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

`TBD` — replace with the measured extraction accuracy table after the day-4 evaluation run.

| Field | Exact | Within 0.5% | Wrong | Hallucinated | Abstained | Ground truth resolvable |
|---|---|---|---|---|---|---|
| total_revenue | | | | | | |
| net_income | | | | | | |
| total_assets | | | | | | |
| operating_cash_flow | | | | | | |

Split by source type:

| Source | Accuracy within tolerance |
|---|---|
| Prose (MD&A narrative) | `TBD` |
| Tables (financial statements) | `TBD` |

**Design consequence**: `TBD` — state the routing decision the numbers justify. If table-derived extraction underperforms, numeric fields route to XBRL relational lookup and only narrative questions touch the embedding pipeline.

---

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

Hybrid search over `TBD` chunks from `TBD` filings across `TBD` companies.

- **BM25** via Postgres full-text search
- **Dense** via pgvector, `bge-small-en-v1.5`
- **Fusion** via Reciprocal Rank Fusion, k=60
- **Metadata pre-filtering** by company, form, fiscal year and section before ranking

Ablation on a hand-labelled query set of `TBD` questions:

| Method | Precision@10 | MRR | p95 latency |
|---|---|---|---|
| BM25 only | | | |
| Dense only | | | |
| Hybrid RRF | | | |

Report the result even if hybrid does not win. It often does not on keyword-heavy financial queries.

---

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

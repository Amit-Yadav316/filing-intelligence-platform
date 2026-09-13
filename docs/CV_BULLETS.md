# CV bullets

Every number here is measured and reproducible from the repository. The commands
that produce each are in the README.

---

## Short form (3 lines)

- Built an Airflow-orchestrated document pipeline over SEC EDGAR filings — immutable
  S3-compatible archival, structural chunking on statutory Item boundaries, hybrid
  BM25 + pgvector retrieval, and LLM extraction **scored against XBRL ground truth
  from the same filing**, giving per-field accuracy with a six-way failure taxonomy
  rather than a demo.
- Measured rather than asserted: hybrid RRF **lost** to dense retrieval alone, while
  metadata pre-filtering lifted precision@10 from **0.210 to 0.718** — so the
  engineering effort went to filtering, not to tuning a fusion constant.
- Operated it like a service: three SLOs with error budgets, burn-rate alerting
  rather than threshold alerting, a quality gate that blocks publication when
  extraction accuracy falls below its floor, and a runbook for three failure modes
  that actually occurred.

---

## Long form (bullet bank)

**Data engineering**
- Ingested 10-K/10-Q filings from SEC EDGAR through a rate-limited client (token
  bucket under the published 10 req/s ceiling, retry with full jitter, `Retry-After`
  honoured, circuit breaker) into MinIO, Hive-partitioned and idempotent: re-running
  a date is a verified no-op, asserted by counting writes rather than comparing output.
- Parsed inline-XBRL HTML into deterministic, versioned text where **every chunk
  round-trips its character offsets**, so a citation resolves to a byte range in a
  named filing years later.
- Rebuilt financial-statement tables from markup where **59% of cells are layout
  spacers** and currency symbols sit in their own cells, recovering column alignment
  that flat text extraction destroys — verified against XBRL (Apple FY2023 net sales
  383,285 → matches `us-gaap` exactly).

**Retrieval**
- Indexed 8,280 chunks from 26 filings across 12 companies in Postgres serving both
  BM25 (FTS) and dense (pgvector/HNSW) retrieval from one table, so metadata
  pre-filtering is a `WHERE` clause rather than a cross-system join.
- Built a 38-question hand-labelled evaluation set and measured all three arms;
  reported that hybrid RRF does not win, and that precision@10 is structurally
  capped at 0.905 by the labelling.
- Cut re-indexing cost by keying the embedding cache on **content hash rather than
  chunk id**: a chunker change that renumbered every chunk still hit **98.9%** of
  cached vectors (3,301s → 314s).

**LLM systems**
- Schema-constrained extraction with explicit abstention, retrieval-grounded rather
  than whole-document, with per-field provenance back to the chunks cited.
- Achieved **zero hallucinations and zero scale errors** across 56 scored extractions;
  80–90% accuracy when the model commits to a figure, with the gap to 55% overall
  traced to retrieval coverage, not model quality — evidenced by a retrieval change
  that moved a fixed sample from 8.3% to 62.5% with the model untouched.
- Diagnosed remaining errors as definitional (`us-gaap:Revenues` vs. revenue including
  other income; net income vs. net income attributable to the parent) rather than
  as model failure.

**Reliability**
- Defined three SLOs with error budgets and wrote burn-rate alerts (14.4× fast burn
  pages, 6× slow burn tickets) instead of threshold alerts.
- Implemented a `ShortCircuitOperator` quality gate that **skips rather than fails**
  when accuracy drops below the floor — a degraded model is contained, not escalated.
- Made a daily-quota 429 non-retryable after finding the generic retry path spending
  four units of a 20/day budget on a single rejected request.

---

## Interview anchors

Things worth being able to talk about, because each was found by measurement:

| Finding | Why it matters |
|---|---|
| Hybrid RRF lost to dense alone | Willing to report a negative result about my own design |
| Pre-filtering beat fusion 3.4× | Knowing where the leverage actually is |
| A more explicit prompt made accuracy *worse* (62.5% → 18.8%) | Prompt changes need measurement, not intuition |
| `filings.recent` is capped, so a form-count check flags the biggest filers | Reading the API's actual behaviour, not its docs |
| Ticker→CIK breaks across reorganisations (XOM, BLK) | Real corpus-construction hazard |
| One prose block spans several Items, orphaning 69 chunks | Found because a retrieval metric looked wrong, not because anything crashed |

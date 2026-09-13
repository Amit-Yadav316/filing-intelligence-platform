# CV bullets

Every number is measured and reproducible from this repository. The commands that
produce each are in the README.

---

## Short form (3 lines)

- Built an Airflow-orchestrated document pipeline over SEC EDGAR filings — immutable
  S3-compatible archival, structural chunking on statutory Item boundaries, hybrid
  BM25 + pgvector retrieval, and LLM extraction **scored against XBRL ground truth
  from the same filing** — reaching **86.4% field-level accuracy with zero
  hallucinations and zero scale errors** across 88 scored extractions.
- Raised accuracy from 55.4% to 86.4% in four measured steps, **three of them
  verified without spending a single LLM call**, using a context-coverage metric
  that isolates retrieval failures from model failures.
- Operated it like a service: three SLOs with error budgets, burn-rate alerting
  rather than threshold alerting, a quality gate that blocks publication when
  accuracy falls below its floor, and a runbook for three failure modes that
  actually occurred.

---

## Long form (bullet bank)

**Evaluation and measurement — the differentiator**

- Built an evaluation harness that scores every extracted figure against the XBRL
  fact the SEC published in the same filing, producing per-field accuracy with a
  **six-way failure taxonomy** (exact, within-tolerance, scale error, wrong,
  hallucinated, abstained) rather than a single number that hides how it fails.
- Designed a **context-coverage metric costing no API calls** — is the ground-truth
  figure present in the chunks the model was shown? — which separates "retrieval
  never found it" from "the model had it and got it wrong". This made tuning
  possible against a free tier capped at 20 requests per day.
- Found and fixed a **ground-truth bug that was marking a correct model wrong**:
  Johnson & Johnson's 52/53-week fiscal 2022 ended 1 January 2023, so keying the
  fiscal year on the calendar year a period ends in collapsed two filings onto one
  year. Nothing errored; the numbers were simply wrong.
- **Published a correction to my own README** after a diagnostic disproved a claim I
  had made in it: abstention was 67% prompt-caused, not retrieval-caused as stated.

**Data engineering**

- Ingested 10-K/10-Q filings through a rate-limited EDGAR client (token bucket under
  the published 10 req/s ceiling, full-jitter retry, `Retry-After` honoured, circuit
  breaker) into MinIO — Hive-partitioned, immutable, and idempotent: a replayed date
  performs **zero writes**, asserted by counting them rather than by comparing output.
- Parsed inline-XBRL HTML into deterministic, versioned text where **every chunk
  round-trips its character offsets**, so a citation resolves to a byte range in a
  named filing years later.
- Rebuilt financial-statement tables from markup where **59% of cells are layout
  spacers** and currency symbols occupy their own cells, recovering the column
  alignment that flat text extraction destroys — verified against XBRL.

**Retrieval**

- Indexed 8,280 chunks across 26 filings in Postgres serving **both** BM25 (FTS) and
  dense (pgvector/HNSW) retrieval from one table, so metadata pre-filtering is a
  `WHERE` clause rather than a cross-system join.
- Measured all three retrieval arms on a 38-question hand-labelled set and reported
  that **hybrid RRF does not win**; metadata pre-filtering lifted precision@10 from
  0.210 to 0.718, a far larger effect than fusion.
- Cut re-indexing cost by keying the embedding cache on **content hash rather than
  chunk id**: a chunker change that renumbered every chunk still hit **98.9%** of
  cached vectors (3,301s → 314s).

**LLM systems**

- Schema-constrained extraction with explicit abstention, retrieval-grounded rather
  than whole-document, with per-field provenance back to the chunks cited.
- Wrote a **provider-agnostic client** covering Gemini plus any OpenAI-compatible
  provider (Groq, OpenRouter, Cerebras), with one canonical JSON Schema and a
  converter to Gemini's dialect so the two cannot drift.
- Engineered around a free tier's real constraints: **budgeted context in tokens
  rather than chunks** (a dense table chunk costs several times a prose chunk),
  calibrated the token estimator against the provider's own count after finding the
  usual chars/4 rule **1.6× optimistic**, and discovered `max_tokens` counts toward
  tokens-per-minute — a 4096 reservation for a 200-token answer was shrinking the
  prompt that fit.
- Made a daily-quota 429 **non-retryable** after finding the generic retry path
  spending four units of a 20/day budget on one rejected request.

**Reliability**

- Three SLOs with error budgets and **burn-rate alerts** (14.4× fast burn pages, 6×
  slow burn tickets) rather than threshold alerts that fire on a single slow query.
- A `ShortCircuitOperator` quality gate that **skips rather than fails** when accuracy
  drops below the floor — a degraded model is contained, not escalated to a page.
- 203 tests, `mypy --strict` clean, ruff clean, CI green across three jobs including
  DAG parse validation and `promtool`/`amtool` config checks.

---

## Interview anchors

Each was found by measurement, not by reading code:

| Finding | What it signals |
|---|---|
| A ground-truth bug marked a **correct model wrong** for two filings | Skepticism toward your own evaluation, not just the model |
| I published a **correction to my own README** when a diagnostic disproved it | You'll report bad news about your own work |
| Hybrid RRF **lost** to dense alone | Willing to report a negative result about your own design |
| Pre-filtering beat fusion **3.4×** | Knowing where the leverage actually is |
| A more explicit prompt made accuracy **worse** (62.5% → 18.8%) | Prompt changes need measurement, not intuition |
| `filings.recent` is capped, so a form-count check flags the **biggest** filers | Reading API behaviour, not API docs |
| Ticker→CIK breaks across reorganisations (XOM, BLK) | Real corpus-construction hazard |
| One prose block spanning several Items orphaned **69 chunks** | Found because a metric looked wrong — nothing crashed |

---

## Claims to avoid

Accuracy in this project is honest because of what it refuses to claim. Keep that:

- **Don't say "switching to Groq improved accuracy."** Provider, retrieval and prompt
  changed together; the model's share is unmeasured, and the README says so.
- **Qualify "Airflow-orchestrated" accurately.** The ingest DAG has executed
  end to end against a live scheduler on real EDGAR data, with catchup and
  max_active_runs verified. process_filings and extract_and_evaluate are
  parse-validated and wired, but have not run a full pass.
- **Don't quote 86.4% without n.** It is 22 filings and 88 extractions; one filing
  moves it about a point.
- **Don't imply schema enforcement.** The final run used JSON mode with the schema
  described in the prompt, not enforced by the provider.

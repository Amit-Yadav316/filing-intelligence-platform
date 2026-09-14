# Design decisions

Every non-obvious choice in this repository, with the reasoning and — where one
exists — the measurement that settled it.

This is written to be defended. If a decision here cannot be justified beyond
"it is the default", it says so.

---

## Retrieval

### Why chunk on Item boundaries before applying a token window

A fixed 512-token window over a 10-K cuts across the legal structure of the
document. Half of Risk Factors lands in the same vector as the start of
Properties, so a query about litigation risk retrieves a chunk that is mostly
about real estate.

Filings have a statutory structure — Item 1 Business, Item 1A Risk Factors,
Item 7 MD&A — and that structure is *semantic*, not cosmetic. Chunking on it
first means every chunk is about one thing. The token window then only sub-divides
within a section, where the content is already coherent.

**Cost of the alternative:** measured indirectly. When a block-attribution bug
put 69 chunks under "no item", `total_assets` abstained on every filing because
the balance sheet could not be filtered to.

### Why RRF `k=60`

RRF scores a document as `Σ 1/(k + rank)`. `k` controls how sharply rank 1 is
favoured over rank 2.

- `k` small (say 1): rank 1 scores 0.5, rank 2 scores 0.33 — a 34% drop. One
  arm's top hit dominates the fused list.
- `k` large (say 1000): 0.000999 vs 0.000998 — ranks barely differ, and fusion
  degenerates into "appeared in both lists".

60 is the value from the original RRF paper (Cormack et al., 2009), chosen
because it is large enough that a single arm's confident-but-wrong top hit
cannot dominate, and small enough that being ranked 1st still beats being
ranked 20th.

**Honest caveat:** I did not tune this on my own query set. It is a defensible
default, not a measured optimum, and the ablation shows fusion was not where the
leverage was anyway.

### Why HNSW rather than IVFFlat

IVFFlat needs a training step over representative data and a sensible `lists`
parameter, both of which assume you already have the corpus. HNSW needs neither,
so an index built on 200 chunks behaves the same way as one built on 200,000.

For a corpus that is rebuilt every time the chunker changes, "no training step"
matters more than IVFFlat's lower memory. At this scale the recall difference is
not measurable.

### Why Postgres serves both BM25 and vectors, rather than a dedicated vector DB

The single most valuable operation in this system is **metadata pre-filtering**:
narrowing to a company and form *before* ranking. Measured effect — precision@10
goes from 0.210 to 0.718, a 3.4× gain, far larger than anything fusion
contributes.

With chunks, embeddings and metadata in one table that pre-filter is a `WHERE`
clause. Split across a vector database and a relational store, it becomes either
a distributed join or post-filtering, which wastes the top-k budget on documents
the user already excluded.

**When I would change this:** past roughly 10M vectors, pgvector's index build
time and memory become the constraint and a dedicated store earns its place. At
8,280 chunks it would be architecture for its own sake.

### Why hybrid retrieval lost, and why it stayed

Measured: dense alone beat hybrid RRF unfiltered (0.210 vs 0.182 precision@10).
RRF weights both arms equally, so fusing a strong dense arm with a weak lexical
one pulls toward the weaker.

It stayed because the lexical arm is doing a different job in extraction: exact
line-item phrases like `"total assets"` are precisely what BM25 is good at and
what embeddings blur. The field anchors that lifted accuracy are BM25 lookups.

So the honest position is: **hybrid is not better for general search on this
corpus, and the lexical arm is essential for targeted fact-finding.** Both are
true, and the README reports both.

---

## Evaluation

### Why 0.5% tolerance

Filings round in narrative text. MD&A says "approximately $383.3 billion" where
the statement says 383,285. Scoring that as wrong measures the document's
readability, not the extractor.

0.5% is wide enough to absorb narrative rounding at the billions scale (383.3 vs
383.285 is 0.004%) and narrow enough that a genuinely different line item never
slips through — the closest real confusion in this corpus, net income versus net
income attributable to the parent, differs by 2–4%.

### Why abstention is its own verdict rather than a wrong answer

An extraction system that guesses is unusable, because you cannot tell which
figures to trust. One that says "not found" is usable with a human in the loop.

Scoring them identically would train the pipeline toward confident wrongness —
the single worst outcome for financial data. Abstention stays in the accuracy
*denominator* though, because a model that abstains on everything is not 100%
accurate, it is 0% useful.

**That distinction is why the README reports two numbers** — overall accuracy
and accuracy-when-answered. The gap between them tells you whether the problem
is the model or the retrieval.

### Why six verdicts instead of right/wrong

"Accuracy 55%" tells you nothing actionable. The taxonomy separates causes that
need different fixes:

| Verdict | What it implies you should fix |
|---|---|
| `scale_error` | The prompt — the model missed an "in millions" heading |
| `wrong` | Retrieval or tag mapping — right magnitude, wrong line |
| `hallucinated` | The model, or the grounding |
| `abstained` | Retrieval coverage |
| `unresolvable` | The answer key, not the model |

Measured payoff: zero scale errors across 92 extractions told me the units
instruction was working, so I never spent time there.

### Why accepting alternative `us-gaap` concepts is not gaming the benchmark

"Net income" is genuinely two concepts: `NetIncomeLoss` (attributable to the
parent) and `ProfitLoss` (including non-controlling interests). Both are correct
answers to the question as asked.

The rule is narrow on purpose: **only concepts the tag map already declared for
that field** are accepted, and that map was written before the failures were
seen. The same diagnostic showed UnitedHealth's `total_assets` answer
coincidentally equalled its `Liabilities`, and a Chevron `net_income` answer
equalled a debt-maturity line. Those still score wrong.

The distinction: accepting a *declared synonym* is fixing the answer key.
Accepting *anything that matches* would be fitting the metric to the results.

### Why context coverage is measured separately from accuracy

Two failures look identical from outside — the model abstained because the
balance sheet was never retrieved, versus abstained while looking straight at
it. They need opposite fixes.

Coverage answers "was the answer even in the context?" and costs **no API
calls**, which is what made tuning possible against a 20-requests-per-day free
tier. It is also the *ceiling*: accuracy cannot exceed it.

Measured payoff: coverage said 67% of abstentions had the figure present, which
disproved my own published claim that abstention was a retrieval problem.

---

## Reliability

### Where 14.4× and 6× come from

Burn rate = how fast the error budget is being consumed relative to the rate
that would exactly exhaust it over the SLO window.

For `/search`: SLO is p95 < 800 ms, so the budget is **5%** of requests.

| Alert | Burn | Error rate | Success threshold | Budget consumed |
|---|---|---|---|---|
| Fast (1h) | 14.4× | 14.4 × 5% = 72% | < 0.28 | 1/720 × 14.4 = **2% in one hour** |
| Slow (6h) | 6× | 6 × 5% = 30% | < 0.70 | 6/720 × 6 = **5% in six hours** |

14.4 is chosen so that one hour of sustained burn consumes 2% of a 30-day
budget — fast enough to warrant waking someone, slow enough that a brief blip
does not. 6× over six hours catches a steady degradation that would exhaust the
budget before month end but is not an emergency, so it files a ticket.

**Why not a threshold alert.** `p95 > 800ms` fires on a single slow query at 3am
and teaches people to ignore the alert. Burn-rate alerting fires on *sustained*
damage to the thing the user actually experiences.

### Why a batch job pushes its own registry, not the shared one

`push_to_gateway` publishes everything in the registry it is given. Handing it a
process-wide registry therefore asserts that every metric in the process is a
finding of this job - and an untouched Gauge reads 0, which Prometheus cannot
distinguish from a real measurement of zero.

Measured consequence, not a hypothetical: pushing the shared registry published
`index_freshness_seconds = 0`. It stayed hidden while the API was up, because the
API's own exporter carried the true 663-day value and `IndexStale` fired on that.
When the API died, the pushed `0` was all that remained and the alert silently
cleared on a 663-day-stale index.

So `push_metrics` takes the registry as a required argument. A default would
have made the unsafe call the easy one, and the unsafe call is the one that
makes monitoring lie.

### Why there are `absent()` alerts as well as threshold alerts

A threshold rule compares a number to a bound. An absent series is not a number,
so it breaches nothing - which means a dead exporter does not trigger the alert,
it *deletes* it. Staleness alerting that depends on the stale component still
reporting is not alerting.

`IndexFreshnessUnreported` and `ExtractionAccuracyUnreported` fire on absence.
They are deliberately different severities: freshness is scraped continuously so
absence means something broke and pages; accuracy arrives by push from a periodic
DAG, so it needs a 6h window and files a ticket.

**The distinction worth stating:** an SLO can be met, breached, or *unmeasured*.
The third is not a special case of the first, and only an absence rule can tell
them apart.

### Why the inhibition rule exists

If the index is stale, extraction accuracy will also degrade — it is scoring
against filings that were never updated. Both alerts fire, but there is one
problem.

The rule suppresses the accuracy alert while `IndexStale` is firing, so the
on-call sees the *cause*, not the cause plus its symptom. Getting two pages for
one incident is how alert fatigue starts.

### Why the quality gate skips rather than fails

A `ShortCircuitOperator` returning `False` marks downstream tasks **skipped**,
not failed. That is deliberate.

A model whose accuracy dropped below the floor is not a broken pipeline — the
DAG did exactly what it should, which is notice and refuse to publish. Failing
the run would page someone for a regression that is already safely contained.
Skipping leaves the previous good extractions in place, still records the
accuracy metric, and lets the alert fire on the *metric* rather than on a red
task.

**The distinction I would state in an interview:** failure means the system is
broken; this is the system working. Those should not look the same on a
dashboard.

### Why 8 requests/second against a published limit of 10

The SEC's limit is enforced per IP, and the penalty is an hours-long block.
Headroom costs throughput; exceeding the ceiling costs the corpus.

Two requesters can also share one IP — `max_active_runs=1` on the ingest DAG
exists for the same reason, since parallel runs would each hold their own
limiter and collectively exceed the ceiling.

### Why a daily-quota 429 is not retried

Retrying a rate limit is correct. Retrying a *quota exhaustion* spends the very
budget that ran out. Measured: the generic retry path consumed **four units of a
20-per-day allowance** on a single rejected request before this was separated.

The two are distinguished by parsing the provider's error detail for a `PerDay`
quota id, not by guessing from the status code.

---

## Data

### Why the embedding cache is keyed on content hash, not chunk id

Chunk ids encode position — `accession::item7::c14`. Adding a paragraph to Item 1
renumbers every chunk after it, so an id-keyed cache misses on text that has not
changed.

Measured: after a chunker change that altered **every** chunk id, a content-hash
cache still served **98.9%** of vectors. 3,301s → 314s.

The key also includes the model name and dimension, because a swapped model must
invalidate everything — otherwise you get 384-dimension vectors for a
768-dimension model, failing deep inside pgvector with an error that names
neither cause.

### Why the raw layer is immutable and the parser is versioned

Every downstream stage is a pure function of the archived bytes. A chunker change
is replayed by reprocessing, never by re-fetching — which matters because
re-fetching means re-consuming a public API that asked you to be polite.

`parser_version` travels with every chunk because character offsets are only
meaningful against the text a specific parser produced. When the parser changes,
old offsets become **known-stale** rather than silently pointing at wrong bytes.

### Why MongoDB for scorecards when Postgres was already there

A scorecard is a nested document read and written whole: a filing, four fields,
each with a verdict, extracted value, truth, tag, relative error and supporting
chunk ids. Its shape changes whenever the schema version does.

Relationally that is a three-table join to read one scorecard, or a JSONB column
that is a document store wearing a Postgres badge. The chunks table is genuinely
relational and stayed in Postgres.

**The honest version:** this is the weakest justification in the project. A JSONB
column would have worked. Mongo earns its place on document shape and on schema
evolution, not on anything I measured.

---

## Things I would do differently

Stated plainly, because being able to criticise your own design is the point.

- **`n = 23` is small.** Enough to find bugs, not enough for confident
  comparisons. The corpus is limited by an LLM free tier, not by the pipeline —
  ingest and indexing handle the full 55-company universe.
- **RRF `k` was never tuned** on my own query set.
- **The prose-vs-table comparison collapsed.** Per-statement retrieval crowded
  narrative out of the context entirely, so there is no prose arm left to
  compare. Reported rather than fabricated.
- **Two of 23 filings ran with a smaller context** to fit a token ceiling, so
  they are not strictly comparable to the other 21.
- **Only the ingest DAG has run a full pass.** The other two are parse-validated
  and wired.
- **The observability stack shipped with a bug that made it lie**, and an
  unrelated OOM kill is what exposed it. Instrumentation deserves the same
  scepticism as the code it measures; mine did not get it until it failed.

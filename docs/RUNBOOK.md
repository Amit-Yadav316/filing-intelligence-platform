# Runbook

Three failure modes that have actually occurred while building this platform,
each with the signal that detects it, the first diagnostic, and the remediation.

Every entry follows the same shape on purpose: at 3am the useful thing is a
decision tree, not prose.

---

## EDGAR rate-limited

### Detection

| Signal | Meaning |
|---|---|
| `filing_intel_edgar_circuit_state == 2` | Breaker open — no requests leaving the process |
| `EdgarCircuitOpen` alert (page) | Above, sustained 5 minutes |
| `filing_intel_edgar_requests_total{status="429"}` rising | SEC is throttling |
| `ingest_edgar_filings` task retries climbing | Symptom, not cause |

### First diagnostic

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  -H "User-Agent: $EDGAR_USER_AGENT" \
  https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json
```

- **403** → the User-Agent is missing or malformed. This is the common one.
  EDGAR rejects requests without a descriptive agent carrying a contact address,
  and returns 403 rather than 429, so it looks like a permissions problem.
  Confirm `EDGAR_USER_AGENT` is set and contains a real email.
- **429** → genuinely throttled. Check whether several DAG runs are active at
  once; each holds its own `RateLimiter`, so three concurrent runs make three
  times the configured request rate.
- **200** → EDGAR is fine and the breaker is stale. It half-opens after
  `edgar_breaker_reset_seconds` (60s) and closes on the next success.

### Remediation

1. Confirm `max_active_runs=1` on `ingest_edgar_filings`. It is set for this
   reason; parallel runs multiply the effective request rate.
2. Lower `EDGAR_RATE_LIMIT_PER_SEC` (default 8, SEC ceiling 10) and restart.
3. Wait out the breaker rather than restarting the process to clear it —
   restarting resets the breaker and resumes exactly the traffic that caused
   the block, which is how a throttle becomes an IP ban.
4. If the IP is blocked, SEC blocks last hours. Stop the DAG, do not retry.

### Prevention in place

One shared token bucket below the published ceiling, a required contact
User-Agent validated at settings construction, and `Retry-After` honoured with
its own 300s ceiling rather than being clamped to the jitter ceiling — an
earlier bug turned a server-requested 7s wait into 0.002s.

---

## LLM provider returning 429

### Detection

| Signal | Meaning |
|---|---|
| `extraction_llm_failed` log events with `429` | Requests being rejected |
| `ExtractionAccuracyBelowSLO` (page) | Downstream effect: fewer scored extractions |
| `filing_intel_llm_cost_usd_total` flat while the DAG runs | Nothing is being billed, so nothing is succeeding |

### First diagnostic

**Determine whether it is a rate limit or a spent quota — they need opposite
responses.**

```bash
curl -s -X POST -H "x-goog-api-key: $GEMINI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"contents":[{"parts":[{"text":"ping"}]}]}' \
  "https://generativelanguage.googleapis.com/v1beta/models/$LLM_MODEL:generateContent" \
  | python -c "import json,sys; e=json.load(sys.stdin).get('error',{}); print(e.get('code'), [v.get('quotaId') for d in e.get('details',[]) if 'QuotaFailure' in d.get('@type','') for v in d.get('violations',[])])"
```

- `quotaId` containing **`PerDay`** → daily quota spent. Waiting will not help
  before midnight Pacific. The free tier allows **20 requests/day/model**.
- `quotaId` containing **`PerMinute`** → transient. The client already honours
  `retryDelay`; it will recover on its own.

### Remediation

| Situation | Action |
|---|---|
| Daily quota spent, work is urgent | Switch `LLM_MODEL` — quota is **per model**, so a sibling model has its own budget. Do not mix models within one accuracy run; the table would blend two models. |
| Daily quota spent, work can wait | Do nothing. Extractions already completed are cached in Redis, keyed on `(accession, schema_version, model)`, so a re-run resumes rather than restarting. |
| Sustained need | Enable billing on the API project. |
| Per-minute limit | No action. Backoff handles it. |

### Prevention in place

A daily-quota 429 raises `QuotaExhaustedError`, which is **not retried** and does
not trip the circuit breaker — the provider is healthy, we are simply out of
budget. This was a real bug: the generic retry path spent four quota units on a
single rejected request, a fifth of the daily allowance per filing.

---

## Embedding backlog growing

### Detection

| Signal | Meaning |
|---|---|
| `filing_intel_index_freshness_seconds` climbing while ingest succeeds | Filings land but are not indexed |
| `EmbeddingCacheHitRateCollapsed` (ticket) | Cache is missing where it should hit |
| `process_filings` duration rising run over run | Embedding is dominating |
| Postgres `count(*) - count(embedding)` > 0 | Chunks stored without vectors |

### First diagnostic

```bash
# Is the cache actually working?
docker exec filing-intel-redis redis-cli INFO stats | grep keyspace
docker exec filing-intel-redis redis-cli DBSIZE

# How many chunks are missing a vector?
docker exec filing-intel-postgres psql -U filing -d filings -c \
  "SELECT count(*) FILTER (WHERE embedding IS NULL) AS unembedded, count(*) AS total FROM chunks;"
```

| Finding | Cause |
|---|---|
| `DBSIZE` near zero after a run | Redis was flushed or restarted without persistence. Expected on a container restart — the cache is deliberately not durable. |
| `DBSIZE` large but hit rate low | The embedding model or dimension changed. The cache key includes both, so every entry correctly invalidated. Expensive but correct. |
| `unembedded` > 0 and static | The embed stage is failing. Check `process_filings` logs. |
| Redis unreachable | Not fatal by design — embedding degrades to recomputation and logs `redis_unavailable`. Slow, not broken. |

### Remediation

1. **Do not** flush Redis to "fix" it. A cold cache is what caused the backlog;
   re-running a full corpus embed takes ~55 minutes on CPU against ~5 minutes warm.
2. If Redis restarted: nothing to do. The first run after repopulates it.
3. If the backlog is genuine, re-run indexing for the affected filings only:
   ```bash
   python -m scripts.index_corpus --accession 0000320193-23-000106
   ```
4. If the model changed deliberately, expect one expensive full re-embed and
   confirm `embedding_dim` matches the `vector(N)` column — a mismatch fails
   inside pgvector with an error that names neither the model nor the cache.

### Prevention in place

Content-hash cache keys, not chunk-id keys: a chunker change that renumbers every
chunk still hits the cache for unchanged text. Measured at **98.9%** across a full
re-index after exactly that change. Redis is capped at 512MB with `allkeys-lru`,
so a full cache evicts cold entries rather than refusing writes.

---

## Quick reference

```bash
make up                      # start the stack
docker compose -f deploy/docker-compose.yml ps   # health of every service
curl -s localhost:8000/health | jq               # index freshness, model versions
curl -s localhost:8000/metrics | grep filing_    # live series
python -m scripts.probe_xbrl_resolution          # is ground truth still resolvable
python -m scripts.index_corpus                   # re-index; safe, converges
```

"""Prometheus metric definitions.

One module owns every series name so that a dashboard query and the code that
emits it cannot drift. Metrics are declared against an explicit registry rather
than the process-global default: Airflow tasks push a *fresh* registry to the
pushgateway per run, and sharing the default registry across DAG runs leaks
series from one run into the next.

Adding a counter while writing the function it measures costs nothing.
Retrofitting metrics across ten modules costs half a day.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from src.config.settings import get_settings

_NS = get_settings().metrics_namespace

REGISTRY = CollectorRegistry()

# --- Ingest ---------------------------------------------------------------
FILINGS_INGESTED = Counter(
    f"{_NS}_filings_ingested_total",
    "Filings landed in object storage.",
    ["form", "status"],
    registry=REGISTRY,
)

EDGAR_REQUESTS = Counter(
    f"{_NS}_edgar_requests_total",
    "HTTP requests issued to SEC EDGAR.",
    ["endpoint", "status"],
    registry=REGISTRY,
)

EDGAR_REQUEST_LATENCY = Histogram(
    f"{_NS}_edgar_request_latency_seconds",
    "Wall time of a single EDGAR HTTP request, excluding rate-limiter wait.",
    ["endpoint"],
    registry=REGISTRY,
)

EDGAR_RATE_LIMIT_WAIT = Histogram(
    f"{_NS}_edgar_rate_limit_wait_seconds",
    "Time blocked in the token bucket before a request was allowed out.",
    buckets=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

EDGAR_CIRCUIT_STATE = Gauge(
    f"{_NS}_edgar_circuit_state",
    "EDGAR client circuit breaker: 0=closed, 1=half_open, 2=open.",
    registry=REGISTRY,
)

# --- Pipeline -------------------------------------------------------------
STAGE_LATENCY = Histogram(
    f"{_NS}_extraction_latency_seconds",
    "Wall time per pipeline stage.",
    ["stage"],
    registry=REGISTRY,
)

TASK_FAILURES = Counter(
    f"{_NS}_pipeline_task_failures_total",
    "Airflow task failures.",
    ["dag", "task"],
    registry=REGISTRY,
)

# --- Quality --------------------------------------------------------------
# Declared once and reused by the batch registry below, so the series a DAG run
# pushes and the series the API exposes can never drift apart.
_ACCURACY_NAME = f"{_NS}_extraction_accuracy_ratio"
_ACCURACY_DOC = "Share of extractions matching XBRL ground truth within tolerance."
_RESOLUTION_NAME = f"{_NS}_ground_truth_resolution_ratio"
_RESOLUTION_DOC = "Share of filings for which an XBRL fact could be resolved for this field."

EXTRACTION_ACCURACY = Gauge(_ACCURACY_NAME, _ACCURACY_DOC, ["field"], registry=REGISTRY)

GROUND_TRUTH_RESOLUTION = Gauge(_RESOLUTION_NAME, _RESOLUTION_DOC, ["field"], registry=REGISTRY)

INDEX_FRESHNESS = Gauge(
    f"{_NS}_index_freshness_seconds",
    "Now minus the filing date of the most recently indexed filing.",
    registry=REGISTRY,
)

# --- Cost -----------------------------------------------------------------
LLM_TOKENS = Counter(
    f"{_NS}_llm_tokens_total",
    "Tokens billed by the LLM provider.",
    ["model", "direction"],
    registry=REGISTRY,
)

LLM_COST_USD = Counter(
    f"{_NS}_llm_cost_usd_total",
    "Cumulative LLM spend in USD.",
    ["model"],
    registry=REGISTRY,
)

# --- Serving --------------------------------------------------------------
RETRIEVAL_LATENCY = Histogram(
    f"{_NS}_retrieval_latency_seconds",
    "End-to-end /search latency.",
    ["mode"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
    registry=REGISTRY,
)

EMBEDDING_CACHE = Counter(
    f"{_NS}_embedding_cache_total",
    "Embedding cache lookups by outcome.",
    ["outcome"],
    registry=REGISTRY,
)


def extraction_eval_registry() -> tuple[CollectorRegistry, Gauge, Gauge]:
    """A registry holding ONLY the two gauges the extraction batch job sets.

    Pushing the module-level ``REGISTRY`` instead would publish every metric it
    owns, including the ones this job never touches - and an untouched Gauge
    reads 0, which is not "no data" but a confident measurement of zero.

    That is not hypothetical. An earlier version pushed the whole registry, so
    ``index_freshness_seconds`` went out as 0 under the batch job's labels. While
    the API was up its true value (663 days) also existed and ``IndexStale``
    fired correctly. When the API went down the pushed 0 was the only series
    left, and the alert silently cleared - the monitoring reported a
    663-day-stale index as perfectly fresh.

    A batch job must publish what it measured and nothing else.
    """
    registry = CollectorRegistry()
    accuracy = Gauge(_ACCURACY_NAME, _ACCURACY_DOC, ["field"], registry=registry)
    resolution = Gauge(_RESOLUTION_NAME, _RESOLUTION_DOC, ["field"], registry=registry)
    return registry, accuracy, resolution

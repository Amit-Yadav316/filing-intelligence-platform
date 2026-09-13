"""The HTTP surface.

Every response carries provenance. A search hit is not a snippet, it is a
chunk id, an Item number, a character range and the EDGAR URL the bytes came
from - so any answer can be checked against the source document rather than
trusted. That chain is the "archival" half of document extraction and
archival, and it is the reason the parser is deterministic and versioned.

A correlation id is attached to every request and threaded through retrieval,
the LLM call and the response, so one question's journey through the system is
a single ``grep``.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.embed.embedding_service import EmbeddingService
from src.evaluate.store import ScorecardStore
from src.observability.logging import configure_logging, get_logger, set_correlation_id
from src.observability.metrics import INDEX_FRESHNESS, REGISTRY, RETRIEVAL_LATENCY
from src.retrieve.hybrid import HybridRetriever
from src.retrieve.indexes import BM25Index, MetadataFilter, VectorIndex
from src.retrieve.store import ChunkStore

log = get_logger(__name__)

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model and open connections once, not per request.

    The embedding model takes seconds to load. Doing that inside a request
    handler would put a cold start on a user's latency budget and break the
    p95 SLO on the first call after every deploy.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    store = ChunkStore(settings)
    store.connect()
    embedder = EmbeddingService(settings)
    _state.update(
        settings=settings,
        store=store,
        embedder=embedder,
        retriever=HybridRetriever(
            BM25Index(store, settings), VectorIndex(store, settings), embedder, settings
        ),
    )
    # Warm the model so the first real request is not the one that pays.
    embedder.embed_query("warmup")
    log.info("api_ready", model=settings.embedding_model)
    try:
        yield
    finally:
        store.close()


app = FastAPI(
    title="Filing Intelligence Platform",
    version="0.1.0",
    description=(
        "Hybrid retrieval and XBRL-scored extraction over SEC filings. "
        "Every response carries the provenance chain back to a character "
        "range in a specific filing."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def correlate(request: Request, call_next):
    cid = set_correlation_id(request.headers.get("X-Correlation-Id"))
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Correlation-Id"] = cid
    log.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return response


# --- schemas --------------------------------------------------------------
class Citation(BaseModel):
    chunk_id: str
    accession: str
    cik: str
    company: str | None
    form: str
    filed: str
    item: str | None
    section: str | None
    char_start: int
    char_end: int
    edgar_url: str


class SearchHitOut(BaseModel):
    rank: int
    score: float = Field(description="Reciprocal Rank Fusion score.")
    found_by: list[str] = Field(description="Which retrieval arms returned this chunk.")
    chunk_type: str
    text: str
    citation: Citation


class SearchResponse(BaseModel):
    query: str
    count: int
    took_ms: float
    results: list[SearchHitOut]


class SearchRequest(BaseModel):
    q: str
    top_k: int = 10
    cik: str | None = None
    form: str | None = None
    year: int | None = None
    section: str | None = None
    chunk_type: str | None = None


# --- routes ---------------------------------------------------------------
@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    """Liveness plus the two things worth knowing: how stale the index is and
    which model version produced the vectors in it."""
    settings = _state["settings"]
    store: ChunkStore = _state["store"]
    stats = store.stats()
    freshness = store.index_freshness_seconds()
    if freshness is not None:
        INDEX_FRESHNESS.set(freshness)
    budget = settings.slo_index_freshness_seconds
    return {
        "status": "ok",
        "index": {
            "chunks": stats.chunks,
            "embedded": stats.embedded,
            "filings": stats.filings,
            "newest_filing": stats.newest_filed.isoformat() if stats.newest_filed else None,
            "freshness_seconds": freshness,
            "freshness_slo_seconds": budget,
            "within_freshness_slo": freshness is not None and freshness < budget,
        },
        "models": {
            "embedding": settings.embedding_model,
            "llm": settings.llm_model,
            "extraction_schema": settings.extraction_schema_version,
        },
    }


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def _to_out(fused: Any, rank: int) -> SearchHitOut:
    hit = fused.hit
    return SearchHitOut(
        rank=rank,
        score=round(fused.rrf_score, 6),
        found_by=list(fused.found_by),
        chunk_type=hit.chunk_type,
        text=hit.text,
        citation=Citation(**hit.citation()),
    )


@app.post("/search", response_model=SearchResponse, tags=["retrieval"])
def search(request: SearchRequest) -> SearchResponse:
    """Hybrid search with metadata pre-filtering.

    Filtering happens before ranking, not after: the ablation measured
    precision@10 rising from 0.210 to 0.718 when company and form narrow the
    candidate set, which is a larger effect than fusion itself.
    """
    if not request.q.strip():
        raise HTTPException(status_code=422, detail="q must not be empty")

    started = time.perf_counter()
    with RETRIEVAL_LATENCY.labels(mode="api").time():
        fused = _state["retriever"].search(
            request.q,
            top_k=request.top_k,
            filters=MetadataFilter(
                cik=request.cik,
                form=request.form,
                fiscal_year=request.year,
                section=request.section,
                chunk_type=request.chunk_type,
            ),
        )
    took = (time.perf_counter() - started) * 1000
    return SearchResponse(
        query=request.q,
        count=len(fused),
        took_ms=round(took, 1),
        results=[_to_out(f, i + 1) for i, f in enumerate(fused)],
    )


@app.get("/extract/{accession}", tags=["extraction"])
def get_extraction(accession: str) -> dict[str, Any]:
    """The stored extraction and its scorecard for one filing.

    Reads what the pipeline already scored rather than re-running the model, so
    the endpoint is fast and returns exactly the figures the accuracy table was
    built from.
    """
    with ScorecardStore(_state["settings"]) as store:
        card = store.get(accession)
    if card is None:
        raise HTTPException(
            status_code=404,
            detail=f"no scored extraction for {accession}. Run extract_and_evaluate first.",
        )
    card.pop("_id", None)
    return dict(card)


@app.get("/accuracy", tags=["extraction"])
def accuracy() -> dict[str, Any]:
    """Per-field accuracy across everything scored - the README table, live."""
    with ScorecardStore(_state["settings"]) as store:
        by_field = store.accuracy_by_field()
        total = store.count()
    return {
        "filings_scored": total,
        "accuracy_by_field": {k: round(v, 4) for k, v in by_field.items()},
        "slo": {
            "field": "total_revenue",
            "floor": _state["settings"].slo_revenue_accuracy_floor,
            "meeting_slo": by_field.get("total_revenue", 0.0)
            >= _state["settings"].slo_revenue_accuracy_floor,
        },
    }


@app.get("/ask", tags=["retrieval"])
def ask(
    q: str = Query(..., description="A question about the corpus."),
    top_k: int = Query(5, ge=1, le=20),
    cik: str | None = None,
) -> dict[str, Any]:
    """Retrieve the passages that answer a question, with citations.

    Deliberately extractive rather than generative. The project's claim is
    measured extraction accuracy; adding an unmeasured free-text answer on top
    would put an unscored assertion next to scored ones, which is exactly the
    thing the README argues against.
    """
    fused = _state["retriever"].search(q, top_k=top_k, filters=MetadataFilter(cik=cik))
    return {
        "question": q,
        "passages": [_to_out(f, i + 1).model_dump() for i, f in enumerate(fused)],
        "note": (
            "Extractive by design: these are the source passages, cited. "
            "Structured financial figures come from /extract, where they are "
            "scored against XBRL ground truth."
        ),
    }

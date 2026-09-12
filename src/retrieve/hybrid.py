"""Hybrid retrieval: lexical and dense, fused with Reciprocal Rank Fusion.

Why RRF rather than a weighted score blend
------------------------------------------
``ts_rank_cd`` and cosine similarity are not on the same scale and their
distributions differ per query, so ``alpha * lexical + (1 - alpha) * dense``
needs normalisation that is itself query-dependent, and the tuned alpha rarely
survives contact with a different query mix.

RRF ignores scores entirely and fuses **ranks**:

    score(d) = sum over retrievers of  1 / (k + rank(d))

with ``k`` damping the influence of any single arm's top result. It has no
per-query tuning, no normalisation step, and it degrades gracefully when one arm
returns nothing - which is exactly what happens on this corpus, where a query of
pure financial jargon can miss lexically while matching densely, and a query for
a specific dollar figure does the reverse.

The claim that hybrid beats both arms is **not assumed here**. It is measured by
:class:`~src.retrieve.ablation.AblationRunner` on a hand-labelled query set, and
the README reports the result even where hybrid loses - which on keyword-heavy
financial queries it sometimes does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger
from src.observability.metrics import RETRIEVAL_LATENCY
from src.retrieve.indexes import BM25Index, MetadataFilter, SearchHit, VectorIndex

log = get_logger(__name__)


class Embedder(Protocol):
    """Just enough of the embedding service for retrieval to depend on."""

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class FusedHit:
    """A hit with its fusion score and where it came from."""

    hit: SearchHit
    rrf_score: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def found_by(self) -> tuple[str, ...]:
        return tuple(sorted(self.ranks))

    @property
    def in_both_arms(self) -> bool:
        return len(self.ranks) > 1


def reciprocal_rank_fusion(
    runs: dict[str, Sequence[SearchHit]], *, k: int = 60, top_k: int | None = None
) -> list[FusedHit]:
    """Fuse ranked lists. ``k`` damps the weight of any one arm's top result."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    hits: dict[str, SearchHit] = {}

    for retriever, run in runs.items():
        for position, hit in enumerate(run, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(hit.chunk_id, {})[retriever] = position
            # Keep the first copy seen; both arms select the same columns, so
            # they differ only in `score` and `retriever`.
            hits.setdefault(hit.chunk_id, hit)

    fused = [
        FusedHit(hit=hits[chunk_id], rrf_score=score, ranks=ranks[chunk_id])
        for chunk_id, score in scores.items()
    ]
    # Ties broken by chunk_id so results are stable across runs - an unstable
    # ordering makes an ablation impossible to reproduce.
    fused.sort(key=lambda f: (-f.rrf_score, f.hit.chunk_id))
    return fused[:top_k] if top_k else fused


class HybridRetriever:
    """Runs both arms, fuses with RRF, returns cited hits."""

    name = "hybrid"

    def __init__(
        self,
        bm25: BM25Index,
        vector: VectorIndex,
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.embedder = embedder
        self.settings = settings or get_settings()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        filters: MetadataFilter | None = None,
        embedding: Sequence[float] | None = None,
    ) -> list[FusedHit]:
        top_k = top_k or self.settings.retrieval_top_k
        candidate_k = candidate_k or self.settings.retrieval_candidate_k

        with RETRIEVAL_LATENCY.labels(mode=self.name).time():
            runs: dict[str, Sequence[SearchHit]] = {
                "bm25": self.bm25.search(query, k=candidate_k, filters=filters)
            }

            if embedding is None and self.embedder is not None:
                embedding = self.embedder.embed_query(query)
            if embedding is not None:
                runs["dense"] = self.vector.search(embedding, k=candidate_k, filters=filters)
            else:
                # Lexical-only is a degraded mode, not a silent one: an
                # unavailable embedder must be visible in the logs rather than
                # quietly halving retrieval quality.
                log.warning("dense_arm_skipped", reason="no embedder and no embedding given")

            fused = reciprocal_rank_fusion(runs, k=self.settings.rrf_k, top_k=top_k)

        log.info(
            "hybrid_search",
            query_chars=len(query),
            bm25_hits=len(runs.get("bm25", ())),
            dense_hits=len(runs.get("dense", ())),
            fused=len(fused),
            both_arms=sum(1 for f in fused if f.in_both_arms),
            filtered=not (filters or MetadataFilter()).is_empty,
        )
        return fused

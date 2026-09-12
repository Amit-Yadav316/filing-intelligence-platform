"""Measure the retrieval arms against a hand-labelled query set.

The point of this module is to make the hybrid-retrieval claim falsifiable. It
is easy to assert that BM25 plus dense plus RRF beats either arm; on a corpus of
financial filings, where queries are often exact phrases from the document, a
lexical index is a strong baseline and sometimes wins outright. The README
reports whatever this measures, including that outcome.

Metrics
-------
* **precision@k** - of the k returned, how many were from the right section.
* **recall@k** - did any relevant chunk make the cut at all. With section-level
  labels the relevant set is large, so this is closer to a hit rate than to
  classical recall, and is reported as such.
* **MRR** - reciprocal rank of the first relevant hit, which is what a reader
  scanning a result list actually experiences.
* **p95 latency** - because retrieval quality that misses the latency SLO is
  not a usable result.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger
from src.retrieve.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.retrieve.indexes import BM25Index, MetadataFilter, SearchHit, VectorIndex

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LabelledQuery:
    id: str
    query: str
    ticker: str
    form: str
    items: tuple[str, ...]
    cik: str = ""

    def is_relevant(self, hit: SearchHit) -> bool:
        if self.cik and hit.cik.lstrip("0") != self.cik.lstrip("0"):
            return False
        if hit.form.upper() != self.form.upper():
            return False
        return (hit.item_number or "").upper() in {i.upper() for i in self.items}


@dataclass
class ArmResult:
    name: str
    precision_at_k: list[float] = field(default_factory=list)
    recall_at_k: list[float] = field(default_factory=list)
    reciprocal_ranks: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    empty: int = 0

    def summary(self, k: int) -> dict[str, Any]:
        def mean(xs: Sequence[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        ordered = sorted(self.latencies_ms)
        p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))] if ordered else 0.0
        return {
            "arm": self.name,
            "queries": len(self.precision_at_k),
            f"precision@{k}": round(mean(self.precision_at_k), 4),
            f"recall@{k}": round(mean(self.recall_at_k), 4),
            "mrr": round(mean(self.reciprocal_ranks), 4),
            "p50_ms": round(statistics.median(ordered), 1) if ordered else 0.0,
            "p95_ms": round(p95, 1),
            "empty_results": self.empty,
        }


def load_query_set(path: Path, universe_path: Path | None = None) -> list[LabelledQuery]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cik_by_ticker: dict[str, str] = {}
    universe_path = universe_path or get_settings().universe_path
    if universe_path.exists():
        universe = json.loads(universe_path.read_text(encoding="utf-8"))
        cik_by_ticker = {c["ticker"]: c["cik"] for c in universe["companies"]}

    return [
        LabelledQuery(
            id=q["id"],
            query=q["query"],
            ticker=q["ticker"],
            form=q["form"],
            items=tuple(q["items"]),
            cik=cik_by_ticker.get(q["ticker"], ""),
        )
        for q in raw["queries"]
    ]


class AblationRunner:
    """Runs each retrieval arm over the labelled set and reports the numbers."""

    def __init__(
        self,
        bm25: BM25Index,
        vector: VectorIndex,
        hybrid: HybridRetriever,
        embedder: Any,
        settings: Settings | None = None,
    ) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.hybrid = hybrid
        self.embedder = embedder
        self.settings = settings or get_settings()

    def _score(
        self, arm: ArmResult, query: LabelledQuery, hits: Sequence[SearchHit], k: int
    ) -> None:
        top = hits[:k]
        if not top:
            arm.empty += 1
        flags = [query.is_relevant(h) for h in top]
        arm.precision_at_k.append(sum(flags) / k)
        arm.recall_at_k.append(1.0 if any(flags) else 0.0)
        arm.reciprocal_ranks.append(next((1.0 / (i + 1) for i, ok in enumerate(flags) if ok), 0.0))

    def run(
        self, queries: Sequence[LabelledQuery], *, k: int | None = None, filtered: bool = False
    ) -> dict[str, Any]:
        """Run all three arms.

        ``filtered`` applies the query's company and form as a metadata
        pre-filter. Both are reported: unfiltered shows raw retrieval quality,
        filtered shows what the API actually does when a user names a company.
        """
        k = k or self.settings.retrieval_top_k
        arms = {name: ArmResult(name) for name in ("bm25", "dense", "hybrid")}

        for q in queries:
            filters = MetadataFilter(cik=q.cik, form=q.form) if filtered and q.cik else None
            embedding = self.embedder.embed_query(q.query)

            started = time.perf_counter()
            lexical = self.bm25.search(
                q.query, k=self.settings.retrieval_candidate_k, filters=filters
            )
            arms["bm25"].latencies_ms.append((time.perf_counter() - started) * 1000)
            self._score(arms["bm25"], q, lexical, k)

            started = time.perf_counter()
            dense = self.vector.search(
                embedding, k=self.settings.retrieval_candidate_k, filters=filters
            )
            arms["dense"].latencies_ms.append((time.perf_counter() - started) * 1000)
            self._score(arms["dense"], q, dense, k)

            # Fuse the runs already computed rather than re-querying, so the
            # comparison isolates fusion rather than measuring warm caches.
            started = time.perf_counter()
            fused = reciprocal_rank_fusion(
                {"bm25": lexical, "dense": dense}, k=self.settings.rrf_k, top_k=k
            )
            arms["hybrid"].latencies_ms.append(
                (time.perf_counter() - started) * 1000
                + arms["bm25"].latencies_ms[-1]
                + arms["dense"].latencies_ms[-1]
            )
            self._score(arms["hybrid"], q, [f.hit for f in fused], k)

        results = {
            "k": k,
            "filtered": filtered,
            "queries": len(queries),
            "rrf_k": self.settings.rrf_k,
            "candidate_k": self.settings.retrieval_candidate_k,
            "arms": [arms[name].summary(k) for name in ("bm25", "dense", "hybrid")],
        }
        log.info("ablation_complete", **{"filtered": filtered, "queries": len(queries)})
        return results

    @staticmethod
    def to_markdown(results: dict[str, Any]) -> str:
        k = results["k"]
        lines = [
            f"| Method | Precision@{k} | Recall@{k} | MRR | p50 | p95 |",
            "|---|---|---|---|---|---|",
        ]
        label = {"bm25": "BM25 (Postgres FTS)", "dense": "Dense (pgvector)", "hybrid": "Hybrid RRF"}
        best = max(a[f"precision@{k}"] for a in results["arms"])
        for arm in results["arms"]:
            name = label.get(arm["arm"], arm["arm"])
            marker = " **" if arm[f"precision@{k}"] == best else " "
            lines.append(
                f"|{marker}{name}{(marker.strip() and '**') or ''} | {arm[f'precision@{k}']:.3f} "
                f"| {arm[f'recall@{k}']:.3f} | {arm['mrr']:.3f} "
                f"| {arm['p50_ms']:.0f} ms | {arm['p95_ms']:.0f} ms |"
            )
        return "\n".join(lines)

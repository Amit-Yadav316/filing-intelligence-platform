"""Measure BM25, dense and hybrid retrieval against the labelled query set.

Writes the result into ``docs/EVALUATION.md`` alongside the XBRL resolution
probe, so the two measured claims of the project sit in one file.

The outcome is reported as measured. On a corpus of financial filings a lexical
index is a strong baseline - queries are often phrases lifted from the document
itself - and hybrid fusion does not automatically win.

Usage:
    python -m scripts.run_ablation
    python -m scripts.run_ablation --k 10 --no-write
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.config.settings import get_settings
from src.embed.embedding_service import EmbeddingService
from src.observability.logging import configure_logging, get_logger
from src.retrieve.ablation import AblationRunner, load_query_set
from src.retrieve.hybrid import HybridRetriever
from src.retrieve.indexes import BM25Index, VectorIndex
from src.retrieve.store import ChunkStore

log = get_logger(__name__)

QUERY_SET = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "query_set.json"
MARKER = "## Retrieval ablation"


def relevance_ceiling(store, queries) -> tuple[float, int]:
    """Highest precision@10 the labelling can award.

    Some Items are genuinely short - Controls and Procedures runs to a few
    chunks - so fewer than 10 relevant chunks exist and precision@10 is
    capped below 1.0 no matter how good retrieval is. Reporting the numbers
    without this makes every arm look worse than it is.
    """
    conn = store.connect()
    caps = []
    for q in queries:
        n = conn.execute(
            "SELECT count(*) FROM chunks WHERE cik=%s AND upper(form)=upper(%s) "
            "AND upper(item_number)=ANY(%s)",
            (q.cik.zfill(10), q.form, [i.upper() for i in q.items]),
        ).fetchone()[0]
        caps.append(min(n, 10) / 10)
    return (sum(caps) / len(caps) if caps else 0.0), sum(1 for c in caps if c < 1.0)


def render(
    unfiltered: dict, filtered: dict, store_stats, query_count: int, ceiling=(1.0, 0)
) -> str:
    out = [
        MARKER,
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} over "
        f"{store_stats.chunks:,} chunks from {store_stats.filings} filings, "
        f"against {query_count} hand-labelled queries._",
        "",
        "A retrieved chunk counts as relevant if it comes from the expected company's "
        "filing **and** falls in the expected Item. Labelling at section level rather "
        "than chunk level is deliberate: chunk ids change whenever the chunker changes, "
        "so chunk-level labels would need redoing on every tuning run.",
        "",
        f"RRF constant k={unfiltered['rrf_k']}, candidate depth "
        f"{unfiltered['candidate_k']} per arm before fusion.",
        "",
        f"**Precision@10 is structurally capped at {ceiling[0]:.3f}**, not 1.0. "
        f"{ceiling[1]} of the {query_count} queries target an Item holding fewer than "
        "ten chunks - Controls and Procedures runs to a handful - so no retriever can "
        "fill ten slots with relevant results. Read the precision figures against that "
        "ceiling rather than against a perfect score.",
        "",
        "### Without metadata filtering",
        "",
        "Raw retrieval quality over the whole corpus - the hard case, where a query "
        "about supply chain risk competes against every company that discusses it.",
        "",
        AblationRunner.to_markdown(unfiltered),
        "",
        "### With company and form pre-filtering",
        "",
        "What the API actually does when the caller names a company. The candidate set "
        "is narrowed before ranking, not after, so the top-k budget is not spent on "
        "documents the user already excluded.",
        "",
        AblationRunner.to_markdown(filtered),
        "",
    ]
    return "\n".join(out)


def merge_into_evaluation(section: str) -> Path:
    path = get_settings().docs_dir / "EVALUATION.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Evaluation\n\n"
    if MARKER in existing:
        head, _, tail = existing.partition(MARKER)
        # Keep anything after this section that belongs to a later one.
        rest = tail.partition("\n## ")
        remainder = ("\n## " + rest[2]) if rest[1] else ""
        updated = head + section + remainder
    else:
        updated = existing.rstrip() + "\n\n" + section
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)

    queries = load_query_set(QUERY_SET)
    print(f"Loaded {len(queries)} labelled queries.")

    with ChunkStore(settings) as store:
        stats = store.stats()
        if stats.embedded == 0:
            print("No embedded chunks in the store. Run `python -m scripts.index_corpus` first.")
            return 1
        print(
            f"Store: {stats.chunks:,} chunks, {stats.embedded:,} embedded, {stats.filings} filings.\n"
        )

        embedder = EmbeddingService(settings)
        bm25 = BM25Index(store, settings)
        vector = VectorIndex(store, settings)
        hybrid = HybridRetriever(bm25, vector, embedder, settings)
        runner = AblationRunner(bm25, vector, hybrid, embedder, settings)

        ceiling = relevance_ceiling(store, queries)
        print(f"Structural precision@10 ceiling: {ceiling[0]:.3f}")
        print("Running unfiltered ablation...")
        unfiltered = runner.run(queries, k=args.k, filtered=False)
        print("Running filtered ablation...")
        filtered = runner.run(queries, k=args.k, filtered=True)

    section = render(unfiltered, filtered, stats, len(queries), ceiling)
    print("\n" + section)

    if not args.no_write:
        path = merge_into_evaluation(section)
        print(f"Wrote {path}")
        raw = get_settings().docs_dir / "ablation.json"
        raw.write_text(
            json.dumps({"unfiltered": unfiltered, "filtered": filtered}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

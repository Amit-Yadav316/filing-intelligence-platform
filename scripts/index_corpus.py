"""Parse, chunk, embed and index every landed filing.

This is the day-3 gate made runnable. The property it proves is that a second
run over an unchanged corpus does no embedding work at all - which is what makes
reprocessing after a chunker change affordable, and is the justification for
Redis being in the stack.

Usage:
    python -m scripts.index_corpus                 # index everything landed
    python -m scripts.index_corpus --accession X   # one filing
    python -m scripts.index_corpus --force         # ignore both caches
    python -m scripts.index_corpus --reset         # drop and rebuild the table
"""

from __future__ import annotations

import argparse
import time

from src.config.settings import get_settings
from src.observability.logging import configure_logging, get_logger
from src.retrieve.pipeline import IndexingPipeline
from src.retrieve.store import ChunkStore

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", action="append", default=None)
    parser.add_argument("--force", action="store_true", help="Re-embed even unchanged chunks.")
    parser.add_argument("--reset", action="store_true", help="Drop the chunks table first.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)

    store = ChunkStore(settings)
    if args.reset:
        print("Dropping the chunks table...")
        store.drop_all()
    store.apply_schema()

    pipeline = IndexingPipeline(settings, store=store)
    targets = args.accession or pipeline.list_landed_filings()
    if args.limit:
        targets = targets[: args.limit]

    print(f"Indexing {len(targets)} filings (force={args.force})...\n")
    started = time.perf_counter()
    results = []
    for i, accession in enumerate(targets, start=1):
        result = pipeline.index_filing(accession, force=args.force)
        results.append(result)
        status = "ok " if result.ok else "FAIL"
        detail = (
            f"chunks {result.chunks:>4}  embedded {result.embedded:>4}  "
            f"skipped {result.skipped_unchanged:>4}"
            if result.ok
            else result.error
        )
        company = (result.company or "?")[:22]
        print(f"  [{i:>2}/{len(targets)}] {status} {company:<24} {accession}  {detail}")
    elapsed = time.perf_counter() - started

    cache = pipeline.embedder.stats
    stats = store.stats()
    embedded_total = sum(r.embedded for r in results)
    skipped_total = sum(r.skipped_unchanged for r in results)
    failed = [r for r in results if not r.ok]

    print(f"\n{'=' * 74}\nINDEXING SUMMARY\n{'=' * 74}")
    print(f"  filings indexed     : {len(results) - len(failed)} / {len(results)}")
    print(f"  chunks in store     : {stats.chunks} ({stats.embedded} embedded)")
    print(f"  distinct filings    : {stats.filings}")
    print(f"  newest filing       : {stats.newest_filed}")
    print(f"  elapsed             : {elapsed:.1f}s")
    print()
    print("  Embedding work this run")
    print(f"    embedded          : {embedded_total}")
    print(f"    skipped unchanged : {skipped_total}  (content hash already in Postgres)")
    print(f"    cache hits        : {cache.hits}")
    print(f"    cache misses      : {cache.misses}")
    print(f"    in-batch repeats  : {cache.duplicates}")
    print(f"    cache hit rate    : {cache.hit_rate:.1%}")

    if failed:
        print("\n  Failures:")
        for r in failed:
            print(f"    {r.accession}: {r.error}")

    print(f"\n{'=' * 74}")
    if embedded_total == 0 and skipped_total > 0:
        print("GATE PASSED: the corpus was already indexed and nothing was re-embedded.")
    elif cache.lookups and cache.hit_rate > 0:
        print(
            f"Indexed with a {cache.hit_rate:.1%} cache hit rate. "
            "Re-run to confirm a second pass embeds nothing."
        )
    else:
        print("Cold run complete. Re-run to confirm the cache and hash skip work.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

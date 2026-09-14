"""Land and index a small corpus, so a fresh clone has something to query.

Fetches a handful of 10-K filings from EDGAR rather than shipping committed
HTML. Two reasons: a filing is one to six megabytes and several of them would
dominate the repository, and fetching exercises the real ingest path - the
rate-limited client, the partitioned archive, the idempotent writer - which a
committed fixture would quietly skip.

The cost is that seeding needs network access and a valid EDGAR_USER_AGENT.
That is the honest trade, and the README says so.

Idempotent: re-running lands nothing new and re-indexes nothing, because the
archive converges on content hash and the chunk store skips unchanged text.

Usage:
    make seed
    python -m scripts.seed_sample --tickers AAPL MSFT --year 2023
"""

from __future__ import annotations

import argparse

from src.config.settings import get_settings
from src.observability.logging import configure_logging, get_logger
from src.retrieve.pipeline import IndexingPipeline
from src.retrieve.store import ChunkStore

log = get_logger(__name__)

DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--year", default="2023")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Land the filings but do not embed them. Embedding needs the ML extras.",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)

    if not settings.universe_path.exists():
        print("config/universe.json is missing. Run `make universe` first.")
        return 1

    # Imported here so that landing works without the ML extras installed.
    from scripts.parse_filing import land_by_spec

    print(f"Landing {len(args.tickers)} filings from EDGAR...\n")
    accessions: list[str] = []
    for ticker in args.tickers:
        spec = f"{ticker}:10-K:{args.year}"
        try:
            accession = land_by_spec(spec)
        except Exception as exc:
            print(f"  {spec}: failed - {exc}")
            continue
        if accession:
            accessions.append(accession)

    if not accessions:
        print("\nNothing landed. Check EDGAR_USER_AGENT in .env and network access.")
        return 1

    if args.skip_index:
        print(f"\nLanded {len(accessions)} filings. Skipping indexing as requested.")
        return 0

    print(f"\nIndexing {len(accessions)} filings (first run downloads the embedding model)...\n")
    with ChunkStore(settings) as store:
        store.apply_schema()
        pipeline = IndexingPipeline(settings, store=store)
        results = pipeline.index_all(accessions)
        stats = store.stats()

    failed = [r for r in results if not r.ok]
    for result in results:
        status = "ok  " if result.ok else "FAIL"
        detail = (
            f"chunks {result.chunks:>4}  embedded {result.embedded:>4}"
            if result.ok
            else result.error
        )
        print(f"  {status} {(result.company or '?')[:24]:<26}{detail}")

    print(f"\nIndexed corpus: {stats.chunks:,} chunks from {stats.filings} filings.")
    print("Next: `make demo` for a search and a scored extraction.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

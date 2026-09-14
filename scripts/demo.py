"""One command that shows the whole platform working.

Runs against the live stack and prints, in order: what is indexed, a hybrid
search with its full provenance chain, a scored extraction with the XBRL tag
each figure was checked against, and the SLO status.

Nothing here is fabricated for the demo - every number is read from Postgres,
MongoDB or the retrieval path at the moment it runs. If the pipeline has not
been run, this says so rather than printing a plausible-looking example.

Usage:
    make demo
    python -m scripts.demo --query "supply chain concentration risk"
"""

from __future__ import annotations

import argparse
import sys
import time

from src.config.settings import get_settings
from src.embed.embedding_service import EmbeddingService
from src.evaluate.store import ScorecardStore
from src.observability.logging import configure_logging
from src.retrieve.hybrid import HybridRetriever
from src.retrieve.indexes import BM25Index, VectorIndex
from src.retrieve.store import ChunkStore

RULE = "=" * 78


def heading(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="supply chain concentration risk")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", json_output=False)

    with ChunkStore(settings) as store:
        stats = store.stats()
        if stats.chunks == 0:
            print(
                "Nothing is indexed yet.\n\n"
                "  make up          start the stack\n"
                "  make seed        land and index a small corpus\n"
                "\nThen run `make demo` again."
            )
            return 1

        heading("1. WHAT IS INDEXED")
        print(f"  chunks          : {stats.chunks:,} ({stats.embedded:,} embedded)")
        print(f"  filings         : {stats.filings}")
        print(f"  newest filing   : {stats.newest_filed}")
        print(f"  embedding model : {settings.embedding_model}")

        heading(f"2. HYBRID SEARCH - {args.query!r}")
        print("  BM25 + pgvector, fused with Reciprocal Rank Fusion.\n")

        retriever = HybridRetriever(
            BM25Index(store, settings),
            VectorIndex(store, settings),
            EmbeddingService(settings),
            settings,
        )
        # Load the embedding model BEFORE timing. The first call pays a
        # multi-second import and model load, and reporting that next to an
        # 800 ms SLO would look like a catastrophic breach rather than a cold
        # start. The SLO is defined over steady state, so that is what is
        # measured here - and the cold cost is reported separately rather than
        # hidden.
        cold_start = time.perf_counter()
        EmbeddingService(settings).embed_query("warmup")
        cold_ms = (time.perf_counter() - cold_start) * 1000

        started = time.perf_counter()
        results = retriever.search(args.query, top_k=args.top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000

        if not results:
            print("  no results")
        for i, fused in enumerate(results, start=1):
            hit = fused.hit
            cite = hit.citation()
            snippet = " ".join(hit.text.split())[:100]
            print(f"  [{i}] {cite['company']}  {cite['form']}  {cite['filed']}")
            print(f"      {cite['item']} ({cite['section']}) - {hit.chunk_type}")
            print(f"      found by : {', '.join(fused.found_by)}   rrf={fused.rrf_score:.4f}")
            print(f"      offsets  : chars {cite['char_start']}-{cite['char_end']}")
            print(f"      source   : {cite['edgar_url']}")
            print(f"      text     : {snippet}...\n")
        budget_ms = settings.slo_search_p95_seconds * 1000
        verdict = "within" if elapsed_ms < budget_ms else "OVER"
        print(f"  retrieved in {elapsed_ms:.0f} ms - {verdict} the {budget_ms:.0f} ms p95 SLO")
        print(f"  (one-off model load on first query: {cold_ms / 1000:.0f} s, excluded above)")

    heading("3. A SCORED EXTRACTION")
    print(
        "  Every figure checked against the XBRL fact the SEC published in the\n"
        "  same filing. `tag` is the us-gaap concept it was compared against.\n"
    )

    with ScorecardStore(settings) as scores:
        try:
            latest = scores.latest(limit=1)
            total_scored = scores.count()
        except Exception as exc:
            print(f"  MongoDB unavailable ({exc}). Run `make extract` after `make up`.")
            latest, total_scored = [], 0

        if not latest:
            print("  No scored extractions yet. Run: python -m scripts.run_extraction")
        else:
            card = latest[0]
            print(f"  {card['company']}  {card['form']}  FY{card['fiscal_year']}")
            print(f"  model: {card['model']}    filing accuracy: {card['accuracy']:.0%}\n")
            print(f"  {'field':<22}{'verdict':<18}{'extracted':>18}{'truth':>18}")
            print(f"  {'-' * 74}")
            for s in card["scores"]:
                extracted = f"{int(float(s['extracted'])):,}" if s["extracted"] else "-"
                truth = f"{int(float(s['truth'])):,}" if s["truth"] else "-"
                print(f"  {s['field']:<22}{s['verdict']:<18}{extracted:>18}{truth:>18}")
                if s.get("tag"):
                    print(f"  {'':<22}us-gaap:{s['tag']}")

            heading("4. SLO STATUS")
            by_field = scores.accuracy_by_field()
            floor = settings.slo_revenue_accuracy_floor
            print(f"  Model: {settings.llm_model}")
            print(f"  Scored across {total_scored} filings.\n")
            for field, value in sorted(by_field.items(), key=lambda kv: -kv[1]):
                print(f"  {field:<24}{value:>7.1%}")
            revenue = by_field.get("total_revenue", 0.0)
            verdict = "MET" if revenue >= floor else "NOT MET"
            print(f"\n  SLO: total_revenue accuracy >= {floor:.0%}  ->  {revenue:.1%}  [{verdict}]")
            if revenue < floor:
                print(
                    "  The quality gate blocks publication below this floor, so a\n"
                    "  breach withholds extractions rather than shipping bad ones."
                )

    print(f"\n{RULE}\nFull measured results: docs/EVALUATION.md and README.md\n{RULE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

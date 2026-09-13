"""How often does retrieval actually put the answer in front of the model?

This is the leading indicator for abstention, and it costs nothing to measure:
for every filing and every scored field, is the XBRL ground-truth figure present
somewhere in the retrieved context?

Measuring it separately from accuracy matters because the two failures look
identical from the outside. A model that abstains because the balance sheet was
never retrieved and a model that abstains while staring at the number are the
same row in the accuracy table and completely different bugs. Coverage isolates
the retrieval half, and it can be iterated on without spending a single LLM call
- which is the difference between tuning retrieval in an afternoon and tuning it
at 20 requests per day.

Usage:
    python -m scripts.measure_context_coverage
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from decimal import Decimal
from typing import Any

from src.config.settings import get_settings
from src.embed.embedding_service import EmbeddingService
from src.evaluate.xbrl_resolver import XBRLResolver
from src.extract.extraction_service import ExtractionService
from src.extract.schemas import SCORED_FIELDS
from src.observability.logging import configure_logging
from src.retrieve.hybrid import HybridRetriever
from src.retrieve.indexes import BM25Index, VectorIndex
from src.retrieve.store import ChunkStore, fiscal_year_for


def value_forms(value: Decimal) -> list[str]:
    """The ways a filing might print one figure: full units, thousands, millions."""
    forms: list[str] = []
    for divisor in (1, 1_000, 1_000_000):
        scaled = value / divisor
        if scaled != scaled.to_integral_value():
            continue
        n = int(scaled)
        if n:
            forms += [f"{n:,}", str(n)]
    return forms


def figure_in_context(value: Decimal | None, context: str) -> bool:
    return value is not None and any(f in context for f in value_forms(abs(value)))


def load_facts(cik: str) -> dict[str, Any] | None:
    path = get_settings().data_dir / "raw" / "companyfacts" / f"CIK{cik}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return dict(json.load(fh))


def main() -> int:
    settings = get_settings()
    configure_logging("WARNING", json_output=False)
    resolver = XBRLResolver.from_config()

    with ChunkStore(settings) as store:
        rows = (
            store.connect()
            .execute(
                """
            SELECT accession, cik, max(company), max(form), max(filed)
            FROM chunks
            -- Annual fields are measured against annual reports. A 10-Q cannot
            -- contain a full-year figure, so including one measures the corpus
            -- rather than the retriever.
            WHERE upper(form) LIKE '10-K%'
            GROUP BY accession, cik ORDER BY max(filed)
            """
            )
            .fetchall()
        )

        service = ExtractionService(
            HybridRetriever(
                BM25Index(store, settings),
                VectorIndex(store, settings),
                EmbeddingService(settings),
                settings,
            ),
            settings,
        )

        per_field: dict[str, Counter[str]] = {f: Counter() for f in SCORED_FIELDS}
        worst: list[tuple[str, str, str]] = []

        print(f"{'Company':<24}{'FY':<7}{'covered':<10}missing")
        print("-" * 78)
        for accession, cik, company, form, filed in rows:
            facts = load_facts(cik)
            if facts is None:
                continue
            fiscal_year = resolver.fiscal_year_for_accession(
                facts, accession, fiscal_year_for(form, filed)
            )
            truth = resolver.resolve(facts, fiscal_year, list(SCORED_FIELDS))
            context = service.render_context(service.build_context(accession))

            missing = []
            covered = 0
            for field in SCORED_FIELDS:
                fact = truth.get(field)
                if fact is None:
                    per_field[field]["unresolvable"] += 1
                    continue
                if figure_in_context(fact.value, context):
                    per_field[field]["covered"] += 1
                    covered += 1
                else:
                    per_field[field]["missing"] += 1
                    missing.append(field)
                    worst.append(((company or "?")[:20], field, str(fact.value)))
            print(
                f"{(company or '?')[:22]:<24}{fiscal_year:<7}{covered}/4       "
                f"{', '.join(missing) if missing else '-'}"
            )

    print(f"\n{'=' * 78}\nCONTEXT COVERAGE BY FIELD\n{'=' * 78}")
    print(f"{'Field':<24}{'covered':>9}{'missing':>9}{'coverage':>11}")
    total_cov = total_all = 0
    for field in SCORED_FIELDS:
        c = per_field[field]
        pool = c["covered"] + c["missing"]
        total_cov += c["covered"]
        total_all += pool
        rate = c["covered"] / pool if pool else 0.0
        print(f"{field:<24}{c['covered']:>9}{c['missing']:>9}{rate:>10.0%}")
    overall = total_cov / total_all if total_all else 0.0
    print(f"\n{'OVERALL':<24}{total_cov:>9}{total_all - total_cov:>9}{overall:>10.0%}")
    print(
        "\nCoverage is the ceiling on accuracy: the model cannot report a figure it\n"
        "was never shown. Everything below this ceiling is a prompt or reasoning\n"
        "problem; the gap to 100% is a retrieval problem."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

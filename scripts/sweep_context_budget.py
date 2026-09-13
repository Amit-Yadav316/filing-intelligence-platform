"""How small can the extraction context get before coverage suffers?

Context size is a three-way trade: bigger context means better coverage, more
tokens per request, and a harder time fitting inside a provider's tokens-per-minute
limit. Groq's free tier allows 8,000 TPM for the models that honour JSON mode,
while an 18-chunk context runs 9,500-12,700 tokens - so a single request exceeds
the entire per-minute budget.

Rather than guess a smaller number, this sweeps the budget and measures both
sides at once, with **no LLM calls**: context coverage (can the answer be found)
against approximate prompt size (will the request be accepted).

Usage:
    python -m scripts.sweep_context_budget
    python -m scripts.sweep_context_budget --budgets 8 10 12 14 18
"""

from __future__ import annotations

import argparse
import gzip
import json
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
    out: list[str] = []
    for divisor in (1, 1_000, 1_000_000):
        scaled = value / divisor
        if scaled == scaled.to_integral_value() and int(scaled):
            out += [f"{int(scaled):,}", str(int(scaled))]
    return out


def present(value: Decimal | None, context: str) -> bool:
    return value is not None and any(f in context for f in value_forms(abs(value)))


def approx_tokens(text: str) -> int:
    """Rough token count without loading a tokenizer.

    Deliberately an estimate: it is used to decide whether a request will fit a
    provider's limit with margin, not to bill anyone. Financial text runs dense
    with digits and separators, so characters-over-four understates slightly;
    the margin below accounts for it.
    """
    return len(text) // 4


def load_facts(cik: str) -> dict[str, Any] | None:
    path = get_settings().data_dir / "raw" / "companyfacts" / f"CIK{cik}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return dict(json.load(fh))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 10, 12, 14, 18])
    parser.add_argument("--tpm-limit", type=int, default=8000)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", json_output=False)
    resolver = XBRLResolver.from_config()

    with ChunkStore(settings) as store:
        rows = (
            store.connect()
            .execute(
                """
            SELECT accession, cik, max(company), max(form), max(filed)
            FROM chunks WHERE upper(form) LIKE '10-K%'
            GROUP BY accession, cik ORDER BY max(filed)
            """
            )
            .fetchall()
        )

        retriever = HybridRetriever(
            BM25Index(store, settings),
            VectorIndex(store, settings),
            EmbeddingService(settings),
            settings,
        )

        # Ground truth once; it does not change with the budget.
        truths: dict[str, dict[str, Any]] = {}
        for accession, cik, _company, form, filed in rows:
            facts = load_facts(cik)
            if facts is None:
                continue
            fy = resolver.fiscal_year_for_accession(facts, accession, fiscal_year_for(form, filed))
            truths[accession] = resolver.resolve(facts, fy, list(SCORED_FIELDS))

        print(f"{'budget':<9}{'coverage':<12}{'median tok':<13}{'max tok':<11}fits 8k TPM?")
        print("-" * 66)
        for budget in args.budgets:
            scoped = settings.model_copy(update={"extraction_context_chunks": budget})
            service = ExtractionService(retriever, scoped)

            covered = total = 0
            sizes: list[int] = []
            for accession, *_ in rows:
                if accession not in truths:
                    continue
                context = service.render_context(service.build_context(accession))
                sizes.append(approx_tokens(context))
                for field in SCORED_FIELDS:
                    fact = truths[accession].get(field)
                    if fact is None:
                        continue
                    total += 1
                    covered += present(fact.value, context)

            sizes.sort()
            median = sizes[len(sizes) // 2] if sizes else 0
            largest = sizes[-1] if sizes else 0
            # Leave room for the system prompt and the completion.
            fits = "yes" if largest + 1500 < args.tpm_limit else "NO"
            rate = covered / total if total else 0.0
            print(f"{budget:<9}{rate:<12.0%}{median:<13,}{largest:<11,}{fits}")

    print(
        "\nCoverage is the ceiling on accuracy. The useful budget is the smallest\n"
        "one that holds coverage while fitting the provider's per-minute limit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

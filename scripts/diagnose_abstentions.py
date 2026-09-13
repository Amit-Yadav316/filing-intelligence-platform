"""Why did the model abstain? Retrieval or reasoning?

Abstention is the single largest source of lost accuracy - the model is right
80-90% of the time when it answers, and declines about 30% of the time. Those
are two very different bugs with two very different fixes, and guessing which
one is in play would be exactly the mistake this project exists to avoid.

So this asks a precise question, with **no LLM calls at all**: for every field
the model declined, was the ground-truth figure actually present in the chunks
it was shown?

* present and abstained  -> the model had the number and did not use it.
  A prompt or reasoning problem.
* absent and abstained   -> retrieval never gave it the statement.
  A retrieval problem, and abstaining was the correct behaviour.

The figure is matched in the several forms a filing can print it: full units,
thousands, millions, with and without separators - because a table headed
"in millions" prints 383,285 for 383,285,000,000, and a naive substring search
would call that absent and blame the wrong component.

Usage:
    python -m scripts.diagnose_abstentions
"""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from typing import Any

from src.config.settings import get_settings
from src.embed.embedding_service import EmbeddingService
from src.extract.extraction_service import ExtractionService
from src.observability.logging import configure_logging
from src.retrieve.hybrid import HybridRetriever
from src.retrieve.indexes import BM25Index, VectorIndex
from src.retrieve.store import ChunkStore


def value_forms(value: Decimal) -> list[str]:
    """The ways a filing might print one figure."""
    forms: list[str] = []
    for divisor in (1, 1_000, 1_000_000):
        scaled = value / divisor
        if scaled != scaled.to_integral_value():
            continue
        n = int(scaled)
        if n == 0:
            continue
        forms.append(f"{n:,}")
        forms.append(str(n))
    return forms


def figure_in_context(value: Decimal | None, context: str) -> bool:
    if value is None:
        return False
    return any(form in context for form in value_forms(abs(value)))


def main() -> int:
    settings = get_settings()
    configure_logging("WARNING", json_output=False)

    cards = json.loads((settings.docs_dir / "scorecards.json").read_text(encoding="utf-8"))

    with ChunkStore(settings) as store:
        retriever = HybridRetriever(
            BM25Index(store, settings),
            VectorIndex(store, settings),
            EmbeddingService(settings),
            settings,
        )
        service = ExtractionService(retriever, settings)

        verdicts: Counter[str] = Counter()
        rows: list[dict[str, Any]] = []

        for card in cards:
            abstained = [s for s in card["scores"] if s["verdict"] == "abstained"]
            if not abstained:
                continue

            hits = service.build_context(card["accession"])
            context = service.render_context(hits)

            for score in abstained:
                truth = Decimal(score["truth"]) if score["truth"] else None
                present = figure_in_context(truth, context)
                verdicts["present" if present else "absent"] += 1
                rows.append(
                    {
                        "company": (card.get("company") or "?")[:20],
                        "field": score["field"],
                        "truth": truth,
                        "present": present,
                        "chunks": len(hits),
                        "tables": sum(1 for h in hits if h.chunk_type == "table"),
                    }
                )

    print(f"\n{'=' * 78}\nABSTENTION DIAGNOSIS\n{'=' * 78}\n")
    print(f"{'Company':<22}{'Field':<22}{'Truth':>18}  in context?")
    print("-" * 78)
    for r in sorted(rows, key=lambda x: (x["field"], x["company"])):
        flag = "YES - model had it" if r["present"] else "no  - retrieval miss"
        truth = f"{int(r['truth']):,}" if r["truth"] is not None else "-"
        print(f"{r['company']:<22}{r['field']:<22}{truth:>18}  {flag}")

    total = sum(verdicts.values())
    print(f"\n{'=' * 78}")
    if not total:
        print("No abstentions to diagnose.")
        return 0

    present = verdicts["present"]
    print(f"Abstentions analysed : {total}")
    print(f"  figure WAS in context : {present:>3}  ({present / total:.0%})  -> prompt/reasoning")
    print(
        f"  figure was NOT        : {total - present:>3}  ({1 - present / total:.0%})  -> retrieval"
    )
    print()
    if present / total >= 0.5:
        print(
            "VERDICT: mostly a PROMPT problem. The model was shown the figure and "
            "declined anyway.\nFixing retrieval further would not help; the "
            "instruction to abstain is too strong."
        )
    else:
        print(
            "VERDICT: mostly a RETRIEVAL problem. The statement never reached the "
            "context, so\nabstaining was correct behaviour. Fix what is retrieved, "
            "not how the model is asked."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Extract and score every indexed filing - the day-4 gate.

Produces the table the README leads with: per-field accuracy against XBRL
ground truth, the prose-versus-table split, and a failure taxonomy.

The prose/table split is the finding the architecture rests on. It is measured,
not assumed: each field's verdict is attributed to the chunks the model itself
cited in ``field_sources``, and those chunks carry a ``chunk_type`` from the
chunker. So "was this number read out of a table or out of narrative text" is
answered by the provenance chain rather than by guessing.

Usage:
    python -m scripts.run_extraction
    python -m scripts.run_extraction --limit 5 --no-cache
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.embed.embedding_service import EmbeddingService
from src.evaluate.evaluator import CORRECT, ExtractionEvaluator, ScoreCard, Verdict
from src.evaluate.xbrl_resolver import XBRLResolver
from src.extract.extraction_service import ExtractionService
from src.extract.schemas import SCORED_FIELDS
from src.ingest.edgar_client import EdgarClient
from src.observability.logging import configure_logging, get_logger
from src.observability.metrics import EXTRACTION_ACCURACY
from src.retrieve.hybrid import HybridRetriever
from src.retrieve.indexes import BM25Index, VectorIndex
from src.retrieve.store import ChunkStore, fiscal_year_for

log = get_logger(__name__)

MARKER = "## Extraction accuracy"
VERDICT_ORDER = (
    Verdict.EXACT,
    Verdict.WITHIN_TOLERANCE,
    Verdict.SCALE_ERROR,
    Verdict.WRONG,
    Verdict.HALLUCINATED,
    Verdict.ABSTAINED,
    Verdict.UNRESOLVABLE,
)


def load_facts(cik: str, client: EdgarClient) -> dict[str, Any]:
    """companyfacts from the local gzip cache, falling back to EDGAR."""
    path = get_settings().data_dir / "raw" / "companyfacts" / f"CIK{cik}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return dict(json.load(fh))
    facts = client.get_company_facts(cik)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(facts, fh)
    return facts


def annual_filings_only(filings: list[dict]) -> list[dict]:
    """Keep 10-K filings when scoring annual fields.

    The four scored fields are annual: full-year revenue, full-year net income,
    year-end total assets, full-year operating cash flow. The XBRL resolver
    deliberately accepts only durations of 340-400 days, so the ground truth is
    an annual fact by construction.

    A 10-Q reports a quarter. Asking a quarterly report for the full year's
    revenue and scoring the absence as a failure measures nothing about the
    extractor - the figure is genuinely not in the document, and abstaining is
    the correct answer. Context-coverage measurement made this visible: all
    three 10-Qs in the corpus scored 0/4 or 1/4 while every 10-K scored 4/4.

    Excluding them is a narrowing of scope, not a filter chosen because it
    flatters the numbers, and the README says how many filings were excluded.
    """
    return [f for f in filings if str(f.get("form", "")).upper().startswith("10-K")]


def indexed_filings(store: ChunkStore) -> list[dict[str, Any]]:
    rows = (
        store.connect()
        .execute(
            """
        SELECT accession, cik, max(company), max(form), max(filed), count(*)
        FROM chunks GROUP BY accession, cik ORDER BY max(filed)
        """
        )
        .fetchall()
    )
    return [
        {
            "accession": r[0],
            "cik": r[1],
            "company": r[2],
            "form": r[3],
            "filed": r[4],
            "chunks": r[5],
        }
        for r in rows
    ]


def chunk_types(store: ChunkStore, chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    rows = (
        store.connect()
        .execute("SELECT chunk_id, chunk_type FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
        .fetchall()
    )
    return {r[0]: r[1] for r in rows}


def source_kind(types: dict[str, str], cited: tuple[str, ...]) -> str:
    """Was this field read from a table or from prose?

    Decided by the chunks the model itself cited. Mixed citations are counted
    as table-derived when any table was cited, because a figure available in a
    statement table is realistically taken from it.
    """
    kinds = [types.get(c) for c in cited if types.get(c)]
    if not kinds:
        return "unknown"
    return "table" if "table" in kinds else "prose"


def render(
    cards: list[ScoreCard],
    splits: dict[str, Counter[str]],
    failures: list[dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    agg = ExtractionEvaluator.aggregate(cards)

    lines = [
        MARKER,
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} - "
        f"{meta['filings']} filings, {meta['model']}, "
        f"{meta['context_chunks']} retrieved chunks per filing, "
        f"${meta['cost']:.4f} total._",
        "",
        "Every figure below is scored against the XBRL fact the SEC published in "
        "the same filing. `unresolvable` means no XBRL fact could be resolved to "
        "compare against; those cases are excluded from the accuracy denominator "
        "and shown separately rather than quietly dropped.",
        "",
        "| Field | Exact | Within 0.5% | Scale error | Wrong | Hallucinated | Abstained | "
        "**Accuracy** | **When answered** | Unresolvable |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in SCORED_FIELDS:
        a = agg.get(name)
        if a is None:
            continue
        c = a.counts
        answered = a.scoreable - c[str(Verdict.ABSTAINED)]
        when_answered = a.correct / answered if answered else 0.0
        lines.append(
            f"| `{name}` | {c[str(Verdict.EXACT)]} | {c[str(Verdict.WITHIN_TOLERANCE)]} "
            f"| {c[str(Verdict.SCALE_ERROR)]} | {c[str(Verdict.WRONG)]} "
            f"| {c[str(Verdict.HALLUCINATED)]} | {c[str(Verdict.ABSTAINED)]} "
            f"| **{a.accuracy:.1%}** | **{when_answered:.1%}** | "
            f"{c[str(Verdict.UNRESOLVABLE)]} |"
        )

    overall = [s for card in cards for s in card.scoreable]
    correct = sum(1 for s in overall if s.verdict in CORRECT)
    lines += [
        "",
        f"**Overall: {correct}/{len(overall)} scoreable field extractions correct "
        f"({correct / len(overall):.1%})** across {meta['filings']} filings.",
        "",
        "**Accuracy and 'when answered' are two different questions, and the gap "
        "between them is the finding.** Overall accuracy counts an abstention as "
        "not-correct, because a model that abstains on everything is useless. "
        "'When answered' excludes abstentions and asks the different question: "
        "when this model does commit to a figure, how often is it right? A large "
        "gap means the model is precise but under-served by retrieval - the fix "
        "is the context, not the model. A small gap with low accuracy would mean "
        "the opposite.",
        "",
        "### Prose versus tables",
        "",
        "Attributed by the chunks the model cited for each field, so this is measured "
        "from the provenance chain rather than assumed.",
        "",
        "| Source | Correct | Total | Accuracy |",
        "|---|---|---|---|",
    ]
    for kind in ("prose", "table", "unknown"):
        data = splits.get(kind)
        if not data or not data["total"]:
            continue
        lines.append(
            f"| {kind.title()} | {data['correct']} | {data['total']} | "
            f"{data['correct'] / data['total']:.1%} |"
        )

    lines += ["", "### Failure taxonomy", ""]
    if failures:
        taxonomy = Counter(f["verdict"] for f in failures)
        lines.append("| Failure mode | Count | What it means |")
        lines.append("|---|---|---|")
        meanings = {
            "scale_error": "Right digits, wrong magnitude - the model missed an "
            "'in millions' heading. A prompt problem, not a model problem.",
            "wrong": "A different figure of the same magnitude - wrong line item "
            "or the prior-year comparative column.",
            "hallucinated": "No plausible relationship to the reported fact.",
        }
        for verdict, count in taxonomy.most_common():
            lines.append(f"| `{verdict}` | {count} | {meanings.get(verdict, '')} |")
        lines += ["", "Worst cases:", ""]
        for f in failures[:10]:
            lines.append(
                f"- `{f['field']}` in {f['company']} {f['form']} ({f['accession']}): "
                f"extracted `{f['extracted']}` against `{f['truth']}` - {f['note']}"
            )
    else:
        lines.append("No incorrect extractions to categorise.")

    return "\n".join(lines) + "\n"


def merge_into_evaluation(section: str) -> Path:
    path = get_settings().docs_dir / "EVALUATION.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Evaluation\n\n"
    if MARKER in existing:
        head, _, tail = existing.partition(MARKER)
        rest = tail.partition("\n## ")
        remainder = ("\n## " + rest[2]) if rest[1] else ""
        updated = head + section + remainder
    else:
        updated = existing.rstrip() + "\n\n" + section
    path.write_text(updated, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-quarterly",
        action="store_true",
        help="Also score 10-Q filings. They cannot contain annual figures, "
        "so this exists to demonstrate the effect rather than to be used.",
    )
    parser.add_argument(
        "--accession",
        action="append",
        default=None,
        help="Score only these filings. Useful for an A/B on a fixed sample.",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)

    resolver = XBRLResolver.from_config()
    evaluator = ExtractionEvaluator(resolver, settings)

    with ChunkStore(settings) as store, EdgarClient(settings) as edgar:
        filings = indexed_filings(store)
        if not args.include_quarterly:
            before = len(filings)
            filings = annual_filings_only(filings)
            if before != len(filings):
                print(
                    f"Scoring annual fields: {len(filings)} 10-K filings "
                    f"({before - len(filings)} quarterly filings excluded)."
                )
        if args.accession:
            wanted = set(args.accession)
            filings = [f for f in filings if f["accession"] in wanted]
        if args.limit:
            filings = filings[: args.limit]
        if not filings:
            print("No indexed filings. Run `python -m scripts.index_corpus` first.")
            return 1

        embedder = EmbeddingService(settings)
        retriever = HybridRetriever(
            BM25Index(store, settings), VectorIndex(store, settings), embedder, settings
        )
        service = ExtractionService(retriever, settings)

        cards: list[ScoreCard] = []
        splits: dict[str, Counter[str]] = defaultdict(Counter)
        failures: list[dict[str, Any]] = []
        total_cost = 0.0
        started = time.perf_counter()

        print(f"Extracting and scoring {len(filings)} filings with {settings.llm_model}...\n")
        for i, filing in enumerate(filings, start=1):
            facts = load_facts(filing["cik"], edgar)
            fiscal_year = resolver.fiscal_year_for_accession(
                facts,
                filing["accession"],
                fiscal_year_for(filing["form"], filing["filed"]),
            )

            outcome = service.extract(
                filing["accession"],
                cik=filing["cik"],
                form=filing["form"],
                fiscal_year=fiscal_year,
                company=filing["company"],
                use_cache=not args.no_cache,
            )
            total_cost += outcome.cost_usd

            if not outcome.ok or outcome.extraction is None:
                print(f"  [{i:>2}/{len(filings)}] FAIL {filing['company']}: {outcome.error}")
                continue

            card = evaluator.score(outcome.extraction, facts)
            cards.append(card)

            cited = sorted({c for s in card.scores for c in s.source_chunks})
            types = chunk_types(store, cited)
            for score in card.scoreable:
                kind = source_kind(types, score.source_chunks)
                splits[kind]["total"] += 1
                if score.verdict in CORRECT:
                    splits[kind]["correct"] += 1
                elif score.verdict is not Verdict.ABSTAINED:
                    failures.append(
                        {
                            "field": score.field,
                            "verdict": str(score.verdict),
                            "company": filing["company"],
                            "form": filing["form"],
                            "accession": filing["accession"],
                            "extracted": score.extracted,
                            "truth": score.truth,
                            "note": score.note,
                        }
                    )

            flag = "cached" if outcome.cached else f"${outcome.cost_usd:.4f}"
            print(
                f"  [{i:>2}/{len(filings)}] ok   {(filing['company'] or '?')[:22]:<24} "
                f"FY{fiscal_year}  accuracy {card.accuracy:>6.1%}  {flag}"
            )

        elapsed = time.perf_counter() - started

    if not cards:
        print("\nNothing scored.")
        return 1

    agg = ExtractionEvaluator.aggregate(cards)
    for name, a in agg.items():
        EXTRACTION_ACCURACY.labels(field=name).set(a.accuracy)

    meta = {
        "filings": len(cards),
        "model": settings.llm_model,
        "context_chunks": settings.extraction_context_chunks,
        "cost": total_cost,
    }
    section = render(cards, splits, failures, meta)
    print("\n" + section)
    print(f"Elapsed {elapsed:.0f}s, total cost ${total_cost:.4f}")

    if not args.no_write:
        print(f"Wrote {merge_into_evaluation(section)}")
        (settings.docs_dir / "scorecards.json").write_text(
            json.dumps([c.as_dict() for c in cards], indent=2), encoding="utf-8"
        )
        # Also into Mongo, so /extract and /accuracy serve the same numbers the
        # README reports rather than a separate copy that can drift from it.
        try:
            from src.evaluate.store import ScorecardStore

            with ScorecardStore(settings) as mongo:
                mongo.ensure_indexes()
                written = mongo.upsert_many([c.as_dict() for c in cards])
            print(f"Published {written} scorecards to MongoDB")
        except Exception as exc:
            print(f"MongoDB unavailable, scorecards saved to JSON only: {exc}")

    floor = settings.slo_revenue_accuracy_floor
    revenue = agg.get("total_revenue")
    if revenue and revenue.accuracy < floor:
        print(f"\nBelow the SLO floor: total_revenue {revenue.accuracy:.1%} < {floor:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

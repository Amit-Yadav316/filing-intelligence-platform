"""Block 1.3 - the resolution probe. Run this before building anything downstream.

The project's central claim is that LLM extraction from filing prose can be
scored against XBRL ground truth. That claim has a precondition: the ground
truth must actually be resolvable. If ``us-gaap`` tag variance means revenue
can only be pinned down for half the corpus, the accuracy denominator collapses
and the evaluation premise needs rethinking - and it is far cheaper to learn
that on day one than on day four.

So this script answers one question, for real companies and real filings:
**for what share of the corpus can each field be resolved, and which tag did it?**

The tag-hit distribution is the useful by-product. It is how
``config/xbrl_tag_map.yaml`` gets extended: run the probe, read which companies
failed, look up what they actually tag, add it, run again.

Gate: if resolution is below 70% for a field, stop and fix the tag map.

Usage:
    python -m scripts.probe_xbrl_resolution
    python -m scripts.probe_xbrl_resolution --years 2022 2023 2024 --refresh
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

from src.config.settings import get_settings
from src.evaluate.xbrl_resolver import XBRLResolver
from src.ingest.edgar_client import EdgarClient, EdgarError
from src.observability.logging import configure_logging, get_logger
from src.observability.metrics import GROUND_TRUTH_RESOLUTION

log = get_logger(__name__)

GATE = 0.70


def _cache_path(cik: str) -> Path:
    # companyfacts payloads run to tens of megabytes; gzip keeps the local
    # cache around a tenth of that and makes a re-run effectively free.
    return get_settings().data_dir / "raw" / "companyfacts" / f"CIK{cik}.json.gz"


def load_company_facts(client: EdgarClient, cik: str, *, refresh: bool = False) -> dict[str, Any]:
    path = _cache_path(cik)
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return dict(json.load(fh))

    facts = client.get_company_facts(cik)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(facts, fh)
    return facts


def probe(years: list[int], *, refresh: bool = False) -> dict[str, Any]:
    settings = get_settings()
    resolver = XBRLResolver.from_config()
    universe = json.loads(settings.universe_path.read_text(encoding="utf-8"))
    companies = universe["companies"]
    fields = list(resolver.fields)

    # (year, field) -> outcome records
    resolved: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    failed: dict[tuple[int, str], list[str]] = defaultdict(list)
    fetch_errors: list[str] = []

    with EdgarClient(settings) as client:
        for i, company in enumerate(companies, start=1):
            cik, ticker = company["cik"], company["ticker"]
            try:
                facts = load_company_facts(client, cik, refresh=refresh)
            except EdgarError as exc:
                fetch_errors.append(f"{ticker} ({cik}): {exc}")
                log.warning("companyfacts_fetch_failed", ticker=ticker, cik=cik, error=str(exc))
                continue

            for year in years:
                outcome = resolver.resolve(facts, year, fields)
                for field, fact in outcome.items():
                    if fact is None:
                        failed[(year, field)].append(ticker)
                    else:
                        resolved[(year, field)].append(
                            {
                                "ticker": ticker,
                                "sector": company["sector"],
                                "tag": fact.tag,
                                "value": fact.value,
                                "period_end": fact.period_end.isoformat(),
                                "form": fact.form,
                                "accession": fact.accession,
                                "restated": fact.restated,
                            }
                        )
            print(f"  [{i:>2}/{len(companies)}] {ticker:<6} probed", flush=True)

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "company_count": len(companies),
        "years": years,
        "fields": fields,
        "resolved": resolved,
        "failed": failed,
        "fetch_errors": fetch_errors,
    }


def _rate(n_resolved: int, total: int) -> float:
    return n_resolved / total if total else 0.0


def report(result: dict[str, Any]) -> tuple[str, bool]:
    """Render the probe result as markdown. Returns (markdown, gate_passed)."""
    total = result["company_count"] - len(result["fetch_errors"])
    lines: list[str] = []

    # A gate that passes vacuously is worse than no gate: an empty field list or
    # an empty corpus means the probe did not run, not that it succeeded.
    gate_passed = bool(result["fields"]) and total > 0

    lines.append("## XBRL ground-truth resolution probe\n")
    lines.append(
        f"_Generated {result['generated_at']} against {total} companies "
        f"from `config/universe.json`._\n"
    )
    lines.append(
        "This is the precondition for every accuracy number in this project: a field "
        "that cannot be resolved to an XBRL fact cannot be scored, and drops out of "
        "the denominator. Tags are tried in the order set by "
        "`config/xbrl_tag_map.yaml`; the first match wins.\n"
    )

    for year in result["years"]:
        rows = []
        for field in result["fields"]:
            hits = result["resolved"][(year, field)]
            misses = result["failed"][(year, field)]
            rate = _rate(len(hits), total)
            if rate < GATE:
                gate_passed = False
            tags = Counter(h["tag"] for h in hits)
            tag_summary = ", ".join(f"`{tag}` x{n}" for tag, n in tags.most_common(4)) or "-"
            restated = sum(1 for h in hits if h["restated"])
            rows.append(
                [
                    f"`{field}`",
                    f"{len(hits)}/{total}",
                    f"{rate:.0%}",
                    restated,
                    tag_summary,
                    ", ".join(misses[:8]) + ("..." if len(misses) > 8 else "") or "-",
                ]
            )

        lines.append(f"\n### FY{year}\n")
        lines.append(
            tabulate(
                rows,
                headers=["Field", "Resolved", "Rate", "Restated", "Tags that matched", "Failed"],
                tablefmt="github",
            )
        )
        lines.append("")

    if result["fetch_errors"]:
        lines.append("\n### Companies whose companyfacts could not be fetched\n")
        for err in result["fetch_errors"]:
            lines.append(f"- {err}")
        lines.append("")

    return "\n".join(lines), gate_passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument(
        "--refresh", action="store_true", help="Re-download companyfacts, ignoring the cache."
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)
    years = args.years or list(settings.fiscal_years)

    print(f"Probing XBRL resolution for FY{years} across the universe...\n")
    result = probe(years, refresh=args.refresh)
    markdown, gate_passed = report(result)

    print("\n" + markdown)

    # Publish the rates so the gate is visible on a dashboard, not only in a file.
    total = result["company_count"] - len(result["fetch_errors"])
    headline = max(years)
    for field in result["fields"]:
        GROUND_TRUTH_RESOLUTION.labels(field=field).set(
            _rate(len(result["resolved"][(headline, field)]), total)
        )

    out = settings.docs_dir / "EVALUATION.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Evaluation\n\n"
        "> Generated by `python -m scripts.probe_xbrl_resolution`. Do not edit by hand.\n"
        "> Hand-written findings about the corpus live in "
        "[`DATA_NOTES.md`](DATA_NOTES.md).\n\n"
        "Measured results. Every number here comes from a run, not an estimate.\n\n"
    )
    out.write_text(header + markdown, encoding="utf-8")
    print(f"\nWrote {out}")

    if not result["fields"] or total == 0:
        print(
            "GATE FAILED: the probe resolved nothing. "
            f"fields={len(result['fields'])}, companies_probed={total}. "
            "Check config/xbrl_tag_map.yaml and config/universe.json."
        )
        return 1
    if gate_passed:
        print(f"\nGATE PASSED: every field resolves for at least {GATE:.0%} of the corpus.")
        return 0
    print(
        f"\nGATE FAILED: at least one field resolves for under {GATE:.0%} of the corpus.\n"
        "Extend config/xbrl_tag_map.yaml using the failed tickers above before "
        "building anything downstream."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

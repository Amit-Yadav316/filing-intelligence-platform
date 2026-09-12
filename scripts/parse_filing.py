"""Parse and chunk archived filings, and verify the result.

This is the day-2 gate made runnable. It does not just emit chunks - it asserts
the two properties that fail silently if they ever break:

* **offset round-trip** - ``text[char_start:char_end]`` equals the chunk text,
  so a citation resolves to the right bytes;
* **no chunk spans an Item boundary** - so a Risk Factors query cannot retrieve
  a chunk that is half Properties.

Usage:
    python -m scripts.parse_filing --accession 0000320193-23-000106
    python -m scripts.parse_filing --land AAPL:10-K:2023 MSFT:10-K:2023
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from typing import Any

from src.chunk.structural_chunker import Chunk, StructuralChunker
from src.config.settings import get_settings
from src.ingest.archive_writer import ArchiveWriter
from src.ingest.edgar_client import EdgarClient, FilingRef, pad_cik
from src.ingest.ingest_service import FilingIngestService
from src.observability.logging import configure_logging, get_logger
from src.parse.filing_parser import FilingParser

log = get_logger(__name__)


def load_universe() -> dict[str, dict[str, Any]]:
    raw = json.loads(get_settings().universe_path.read_text(encoding="utf-8"))
    return {c["ticker"]: c for c in raw["companies"]}


def land_by_spec(spec: str) -> str | None:
    """Land one filing named as TICKER:FORM:YEAR, returning its accession."""
    ticker, form, year = spec.split(":")
    company = load_universe().get(ticker.upper())
    if company is None:
        log.warning("ticker_not_in_universe", ticker=ticker)
        return None

    cik = pad_cik(company["cik"])
    with EdgarClient() as client:
        submissions = client.get_submissions(cik)
        recent = submissions["filings"]["recent"]
        for acc, f, filed, name in zip(
            recent["accessionNumber"],
            recent["form"],
            recent["filingDate"],
            recent.get("primaryDocument", [""] * len(recent["form"])),
            strict=False,
        ):
            if f.upper() != form.upper() or not filed.startswith(str(year)):
                continue
            ref = FilingRef(
                cik=cik,
                company=submissions.get("name", ticker),
                form=f,
                filed=date.fromisoformat(filed),
                path=f"edgar/data/{cik}/{acc}.txt",
            )
            with FilingIngestService(client=client) as svc:
                result = svc.ingest_filing(ref)
            if result.ok:
                print(f"  landed {ticker} {f} {filed}  {acc}  ({name})")
                return acc
            print(f"  FAILED {ticker} {f} {filed}: {result.error}")
            return None
    log.warning("no_matching_filing", spec=spec)
    return None


def find_partition(accession: str) -> tuple[str, dict[str, Any]] | None:
    """Locate a landed partition by accession, via its manifest."""
    writer = ArchiveWriter()
    paginator = writer._client.get_paginator("list_objects_v2")
    needle = f"accession={accession}/"
    for page in paginator.paginate(Bucket=writer.bucket):
        for item in page.get("Contents", []):
            key = item["Key"]
            if needle in key and key.endswith("_manifest.json"):
                prefix = key[: -len("_manifest.json")]
                manifest = writer.read_manifest(prefix)
                if manifest is not None:
                    return prefix, manifest.model_dump(mode="json")
    return None


def verify(chunks: list[Chunk], parsed: Any, chunker: StructuralChunker) -> dict[str, Any]:
    regions = chunker.find_item_regions(parsed)
    bad_offsets = [c.chunk_id for c in chunks if parsed.text[c.char_start : c.char_end] != c.text]

    # Keyed by item number is wrong: a 10-Q has an Item 1 in Part I and another
    # in Part II, so a dict keeps only the second and reports every Part I
    # chunk as out of bounds. A chunk is in bounds if it sits inside ANY region
    # carrying its item number.
    by_item: dict[str, list[Any]] = {}
    for r in regions:
        by_item.setdefault(r.item_number, []).append(r)

    crossing = [
        c.chunk_id
        for c in chunks
        if c.item_number
        and c.item_number in by_item
        and not any(
            r.char_start <= c.char_start and c.char_end <= r.char_end
            for r in by_item[c.item_number]
        )
    ]
    prose = [c.token_count for c in chunks if c.chunk_type == "prose"]
    budget = chunker.settings.chunk_token_window
    return {
        "chunks": len(chunks),
        "prose": len(prose),
        "tables": sum(1 for c in chunks if c.chunk_type == "table"),
        "regions": len(regions),
        "items": sorted({r.item_number for r in regions}, key=lambda i: (len(i), i)),
        "bad_offsets": bad_offsets,
        "crossing_boundary": crossing,
        "oversized": [
            c.chunk_id for c in chunks if c.chunk_type == "prose" and c.token_count > budget * 1.1
        ],
        "token_min": min(prose) if prose else 0,
        "token_max": max(prose) if prose else 0,
        "token_mean": sum(prose) // len(prose) if prose else 0,
    }


def process(accession: str, *, write_json: bool = True) -> dict[str, Any] | None:
    found = find_partition(accession)
    if found is None:
        print(f"  {accession}: not found in the archive - land it first")
        return None
    _prefix, manifest = found

    writer = ArchiveWriter()
    document = next(
        (o for o in manifest["objects"] if o["filename"].endswith((".htm", ".html"))), None
    )
    if document is None:
        print(f"  {accession}: no HTML document in the partition")
        return None

    raw = writer._get(document["key"])
    if raw is None:
        print(f"  {accession}: object missing from storage")
        return None

    parsed = FilingParser().parse(raw)
    chunker = StructuralChunker()
    chunks, _ = chunker.chunk(
        parsed,
        accession=manifest["accession"],
        cik=manifest["cik"],
        form=manifest["form"],
        filed=date.fromisoformat(manifest["filed"]),
    )
    report = verify(chunks, parsed, chunker)
    report["accession"] = accession
    report["company"] = manifest.get("company")
    report["form"] = manifest["form"]
    report["chars"] = parsed.n_chars

    if write_json:
        out_dir = get_settings().data_dir / "interim" / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{accession}.json"
        out.write_text(
            json.dumps([{**asdict(c), "filed": c.filed.isoformat()} for c in chunks], indent=2),
            encoding="utf-8",
        )
        report["output"] = str(out)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", action="append", default=[])
    parser.add_argument(
        "--land", nargs="+", default=[], help="TICKER:FORM:YEAR specs to fetch first"
    )
    args = parser.parse_args()

    configure_logging(get_settings().log_level, json_output=False)

    accessions = list(args.accession)
    if args.land:
        print("Landing filings:")
        for spec in args.land:
            acc = land_by_spec(spec)
            if acc:
                accessions.append(acc)
        print()

    if not accessions:
        parser.error("nothing to do: pass --accession or --land")

    reports = [r for r in (process(a) for a in accessions) if r]
    failures = 0

    print(f"\n{'=' * 78}\nPARSE AND CHUNK VERIFICATION\n{'=' * 78}")
    for r in reports:
        ok = not (r["bad_offsets"] or r["crossing_boundary"] or r["oversized"])
        failures += 0 if ok else 1
        print(f"\n{r['company']}  {r['form']}  {r['accession']}")
        print(
            f"  chars {r['chars']:>7} | chunks {r['chunks']:>4} "
            f"(prose {r['prose']}, tables {r['tables']}) | regions {r['regions']} | items {len(r['items'])}"
        )
        print(f"  tokens: min {r['token_min']} max {r['token_max']} mean {r['token_mean']}")
        print(f"  items : {', '.join(r['items'])}")
        print(
            f"  offsets round-trip      : {'OK' if not r['bad_offsets'] else str(len(r['bad_offsets'])) + ' BROKEN'}"
        )
        print(
            f"  no Item boundary spanned: {'OK' if not r['crossing_boundary'] else str(len(r['crossing_boundary'])) + ' CROSSING'}"
        )
        print(
            f"  within token budget     : {'OK' if not r['oversized'] else str(len(r['oversized'])) + ' OVERSIZED'}"
        )

    print(f"\n{'=' * 78}")
    if reports and failures == 0:
        print(
            f"GATE PASSED: {len(reports)} filings parsed and chunked with all invariants holding."
        )
        return 0
    print(f"GATE FAILED: {failures} of {len(reports)} filings broke an invariant.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

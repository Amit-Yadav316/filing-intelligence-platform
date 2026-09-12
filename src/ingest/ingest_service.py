"""Discover, fetch and land - the ingest stage as a service.

The Airflow DAG for this stage is deliberately thin: it supplies a
``logical_date`` and calls into here. Keeping the logic in ``src/`` means the
vertical slice can be run and tested from a shell long before an orchestrator
exists, which is the order CLAUDE.md asks for - never orchestrate before the
thing works once.

Storage layout, and why XBRL is not stored per filing
-----------------------------------------------------
``companyfacts`` is published per *company*, not per filing, and runs to tens of
megabytes. Copying it into all six of a company's filing partitions would
multiply the archive for no gain, so it lands once per company:

    cik=0000320193/xbrl/companyfacts.json
    cik=0000320193/form=10-K/filed=2023-11-03/accession=.../aapl-20230930.htm

The filing partition holds what is unique to that filing; the company prefix
holds what is shared across them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.config.settings import Settings, get_settings
from src.ingest.archive_writer import (
    ArchiveWriter,
    LandingStatus,
    PartitionManifest,
    json_bytes,
)
from src.ingest.edgar_client import EdgarClient, EdgarError, FilingRef, pad_cik
from src.observability.logging import get_logger
from src.observability.metrics import STAGE_LATENCY

log = get_logger(__name__)

# index.json carries a `type` field, but it holds an icon filename such as
# "text.gif" rather than a document type, so it cannot be used to identify the
# primary document. Name and size are the signals that actually work.
_EXHIBIT = re.compile(r"exhibit|(^|[^a-z0-9])ex-?\d", re.IGNORECASE)
_NOT_PRIMARY = ("-index", "-index-headers", "filingsummary")


class IngestError(RuntimeError):
    """A filing could not be ingested."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    ref: FilingRef
    manifest: PartitionManifest | None
    status: LandingStatus | str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.manifest is not None


def choose_primary_document(items: list[dict[str, Any]]) -> str | None:
    """Pick the filing's main document out of its directory listing.

    A 10-K directory holds around a hundred files: the primary document, its
    exhibits, the XBRL instance, schemas and rendering artefacts. The primary
    document is an HTML file that is not an exhibit and is, by a wide margin,
    the largest - Apple's FY2023 10-K is 1.5 MB against exhibits of 5-122 KB.
    Filtering by name then ranking by size gets it right without relying on a
    field EDGAR does not populate usefully.
    """
    candidates: list[tuple[int, str]] = []
    for item in items:
        name = str(item.get("name", ""))
        low = name.lower()
        if not low.endswith((".htm", ".html")):
            continue
        if any(marker in low for marker in _NOT_PRIMARY):
            continue
        if _EXHIBIT.search(low):
            continue
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        candidates.append((size, name))

    if not candidates:
        return None
    # Largest wins; name breaks ties so the choice is deterministic.
    return max(candidates, key=lambda c: (c[0], c[1]))[1]


class FilingIngestService:
    """Turns a date, or a filing reference, into landed bytes."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: EdgarClient | None = None,
        writer: ArchiveWriter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self.client = client or EdgarClient(self.settings)
        self.writer = writer or ArchiveWriter(self.settings)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> FilingIngestService:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- discovery --------------------------------------------------------
    def discover(
        self,
        on: date,
        *,
        forms: tuple[str, ...] | None = None,
        ciks: set[str] | None = None,
    ) -> list[FilingRef]:
        """Filings disseminated on a date, narrowed to the corpus.

        ``ciks`` is matched on the zero-padded form so that the daily index's
        unpadded CIKs line up with ``config/universe.json``.
        """
        refs = self.client.get_daily_index(on, forms or self.settings.target_forms)
        if ciks is not None:
            wanted = {pad_cik(c) for c in ciks}
            refs = [r for r in refs if pad_cik(r.cik) in wanted]
        log.info("discovered", date=on.isoformat(), count=len(refs))
        return refs

    def universe_ciks(self) -> set[str]:
        import json

        raw = json.loads(self.settings.universe_path.read_text(encoding="utf-8"))
        return {pad_cik(c["cik"]) for c in raw["companies"]}

    # --- fetch and land ---------------------------------------------------
    def resolve_primary_document(self, ref: FilingRef) -> str:
        index = self.client.get_filing_index(ref.cik, ref.accession)
        items = index.get("directory", {}).get("item", [])
        name = choose_primary_document(items)
        if name is None:
            raise IngestError(
                f"no primary document found for {ref.accession} among {len(items)} directory items"
            )
        return name

    def ingest_filing(self, ref: FilingRef, *, overwrite: bool = False) -> IngestResult:
        """Fetch one filing's primary document and land it in its partition."""
        with STAGE_LATENCY.labels(stage="ingest").time():
            try:
                filename = self.resolve_primary_document(ref)
                body = self.client.fetch_filing_document(ref.cik, ref.accession, filename)
                source = (
                    f"{self.settings.edgar_archives_base}/edgar/data/"
                    f"{ref.cik.lstrip('0')}/{ref.accession_nodash}/{filename}"
                )
                manifest, status = self.writer.land_filing(
                    cik=pad_cik(ref.cik),
                    form=ref.form,
                    filed=ref.filed,
                    accession=ref.accession,
                    company=ref.company,
                    documents={filename: (body, source, "text/html")},
                    overwrite=overwrite,
                )
            except (EdgarError, IngestError) as exc:
                log.warning("ingest_failed", accession=ref.accession, cik=ref.cik, error=str(exc))
                return IngestResult(ref=ref, manifest=None, status="failed", error=str(exc))
        return IngestResult(ref=ref, manifest=manifest, status=status)

    def ingest_company_facts(
        self, cik: str, *, overwrite: bool = False
    ) -> tuple[PartitionManifest | None, str]:
        """Land one company's XBRL facts - the ground truth - once per company.

        Stored under ``cik={cik}/xbrl/`` rather than inside each filing
        partition, because the payload is per-company and large.
        """
        padded = pad_cik(cik)
        prefix = f"cik={padded}/xbrl/"
        url = f"{self.settings.edgar_data_api_base}/companyfacts/CIK{padded}.json"

        try:
            facts = self.client.get_company_facts(padded)
        except EdgarError as exc:
            log.warning("companyfacts_fetch_failed", cik=padded, error=str(exc))
            return None, "failed"

        body = json_bytes(facts)
        manifest, status = self.writer.land_filing(
            cik=padded,
            form="XBRL",
            filed=date.today(),
            accession=f"companyfacts-{padded}",
            company=str(facts.get("entityName") or "") or None,
            documents={"companyfacts.json": (body, url, "application/json")},
            prefix=prefix,
            edgar_url=url,
            overwrite=overwrite,
        )
        log.info("companyfacts_landed", cik=padded, bytes=len(body), status=status)
        return manifest, status

    # --- the day's work ---------------------------------------------------
    def ingest_date(
        self,
        on: date,
        *,
        forms: tuple[str, ...] | None = None,
        ciks: set[str] | None = None,
        overwrite: bool = False,
        with_facts: bool = True,
    ) -> list[IngestResult]:
        """Land every corpus filing disseminated on a date.

        Idempotent by construction: the writer converges each partition, so
        replaying a date is a no-op rather than a duplicate.
        """
        refs = self.discover(on, forms=forms, ciks=ciks)
        results = [self.ingest_filing(ref, overwrite=overwrite) for ref in refs]

        if with_facts:
            # One call per distinct company, not per filing.
            for cik in sorted({pad_cik(r.ref.cik) for r in results if r.ok}):
                self.ingest_company_facts(cik, overwrite=overwrite)

        landed = sum(1 for r in results if r.ok)
        log.info(
            "ingest_date_complete",
            date=on.isoformat(),
            discovered=len(refs),
            landed=landed,
            failed=len(results) - landed,
        )
        return results

"""ArchiveWriter tests.

Idempotency is the property worth testing hardest here. A backfill replays
dates, and CLAUDE.md requires a re-run to produce identical state - so these
tests count writes rather than settling for "it didn't crash".
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from src.ingest.archive_writer import (
    MANIFEST_NAME,
    ArchiveWriter,
    safe_segment,
    sha256_hex,
)

FILED = date(2023, 11, 3)
ACC = "0000320193-23-000106"
CIK = "0000320193"
PREFIX = f"cik={CIK}/form=10-K/filed=2023-11-03/accession={ACC}/"


class FakeS3:
    """An in-memory stand-in for the S3 API surface ArchiveWriter uses.

    Counts writes so a test can assert that an idempotent re-run performed
    none, which is the actual contract.
    """

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[str] = []
        self.buckets: set[str] = {"filings"}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:  # noqa: N803
        self.store[(Bucket, Key)] = Body
        self.put_calls.append(Key)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if (Bucket, Key) not in self.store:
            raise self._no_such_key()
        payload = self.store[(Bucket, Key)]
        return {"Body": _Body(payload)}

    def head_bucket(self, *, Bucket: str) -> None:  # noqa: N803
        if Bucket not in self.buckets:
            raise self._no_such_key()

    def create_bucket(self, *, Bucket: str) -> None:  # noqa: N803
        self.buckets.add(Bucket)

    def get_paginator(self, _name: str) -> Any:
        return _Paginator(self.store)

    @staticmethod
    def _no_such_key() -> Exception:
        return type("NoSuchKey", (Exception,), {})()


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _Paginator:
    def __init__(self, store: dict[tuple[str, str], bytes]) -> None:
        self._store = store

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
        contents = [
            {"Key": key}
            for (bucket, key) in sorted(self._store)
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return [{"Contents": contents}] if contents else [{}]


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def writer(settings: Any, s3: FakeS3) -> ArchiveWriter:
    return ArchiveWriter(settings, client=s3, bucket="filings")


def docs(html: bytes = b"<html>filing</html>") -> dict[str, tuple[bytes, str, str]]:
    return {
        "aapl-20230930.htm": (html, "https://sec.gov/a.htm", "text/html"),
        "companyfacts.json": (b'{"cik":320193}', "https://data.sec.gov/f.json", "application/json"),
    }


# --- layout ---------------------------------------------------------------
def test_partition_prefix_is_hive_style() -> None:
    assert ArchiveWriter.partition_prefix(CIK, "10-K", FILED, ACC) == PREFIX


def test_form_with_a_slash_does_not_split_the_partition() -> None:
    """An amended annual report is filed as 10-K/A. Left raw, the slash would
    silently create a nested prefix and split one filing across two partitions."""
    prefix = ArchiveWriter.partition_prefix(CIK, "10-K/A", FILED, ACC)

    assert "form=10-K-A/" in prefix
    assert prefix.count("/") == 4, "exactly four path segments, no extra nesting"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("10-K", "10-K"), ("10-K/A", "10-K-A"), ("N-CSR/A", "N-CSR-A"), ("  10-Q ", "10-Q")],
)
def test_safe_segment(raw: str, expected: str) -> None:
    assert safe_segment(raw) == expected


def test_edgar_url_strips_cik_padding() -> None:
    url = ArchiveWriter.edgar_url(CIK, ACC)
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/"


# --- landing --------------------------------------------------------------
def test_lands_documents_and_a_manifest(writer: ArchiveWriter, s3: FakeS3) -> None:
    manifest, status = writer.land_filing(
        cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs(), company="Apple Inc."
    )

    assert status == "landed"
    assert sorted(writer.list_partition(PREFIX)) == [
        PREFIX + MANIFEST_NAME,
        PREFIX + "aapl-20230930.htm",
        PREFIX + "companyfacts.json",
    ]
    assert manifest.total_bytes > 0
    assert manifest.edgar_url.endswith("/000032019323000106/")


def test_manifest_records_hash_size_and_source_for_every_object(
    writer: ArchiveWriter,
) -> None:
    """The manifest is what makes the archive verifiable without re-fetching."""
    body = b"<html>filing</html>"
    manifest, _ = writer.land_filing(
        cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs(body)
    )

    entry = manifest.by_name()["aapl-20230930.htm"]
    assert entry.sha256 == sha256_hex(body)
    assert entry.size_bytes == len(body)
    assert entry.source_url == "https://sec.gov/a.htm"
    assert entry.key == PREFIX + "aapl-20230930.htm"


def test_manifest_round_trips_through_storage(writer: ArchiveWriter) -> None:
    written, _ = writer.land_filing(
        cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs()
    )

    read_back = writer.read_manifest(PREFIX)

    assert read_back is not None
    assert read_back.accession == written.accession
    assert read_back.by_name().keys() == written.by_name().keys()


def test_read_manifest_returns_none_for_an_unlanded_partition(writer: ArchiveWriter) -> None:
    assert writer.read_manifest("cik=999/form=10-K/filed=2020-01-01/accession=x/") is None


# --- idempotency ----------------------------------------------------------
def test_relanding_identical_content_writes_nothing(writer: ArchiveWriter, s3: FakeS3) -> None:
    """The contract a backfill depends on: a second run over the same date is
    a genuine no-op, not a rewrite that happens to produce the same bytes."""
    writer.land_filing(cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs())
    writes_after_first = len(s3.put_calls)

    _, status = writer.land_filing(
        cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs()
    )

    assert status == "unchanged"
    assert len(s3.put_calls) == writes_after_first, "re-landing performed a write"


def test_relanding_preserves_the_original_fetch_timestamp(writer: ArchiveWriter) -> None:
    """The archival record is when the bytes were FIRST obtained. Refreshing it
    on every replay would destroy the only provenance the manifest carries."""
    first, _ = writer.land_filing(
        cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs()
    )

    second, status = writer.land_filing(
        cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs()
    )

    assert status == "unchanged"
    assert second.fetched_at == first.fetched_at


def test_changed_content_is_refused_rather_than_silently_absorbed(
    writer: ArchiveWriter, s3: FakeS3
) -> None:
    """EDGAR serving different bytes for a landed accession is a real event,
    not noise. Silently overwriting would destroy the original without a trace."""
    writer.land_filing(cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs())
    writes_before = len(s3.put_calls)

    manifest, status = writer.land_filing(
        cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs(b"<html>DIFFERENT</html>")
    )

    assert status == "unchanged"
    assert len(s3.put_calls) == writes_before, "a conflicting re-land wrote anyway"
    assert manifest.by_name()["aapl-20230930.htm"].sha256 == sha256_hex(b"<html>filing</html>")


def test_overwrite_flag_replaces_the_partition(writer: ArchiveWriter) -> None:
    writer.land_filing(cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs())

    manifest, status = writer.land_filing(
        cik=CIK,
        form="10-K",
        filed=FILED,
        accession=ACC,
        documents=docs(b"<html>AMENDED</html>"),
        overwrite=True,
    )

    assert status == "overwritten"
    assert manifest.by_name()["aapl-20230930.htm"].sha256 == sha256_hex(b"<html>AMENDED</html>")


def test_landing_one_partition_leaves_others_untouched(writer: ArchiveWriter) -> None:
    """'Overwrites exactly that partition, nothing else.'"""
    other_acc = "0000320193-22-000108"
    writer.land_filing(
        cik=CIK, form="10-K", filed=date(2022, 10, 28), accession=other_acc, documents=docs()
    )
    other_prefix = ArchiveWriter.partition_prefix(CIK, "10-K", date(2022, 10, 28), other_acc)
    before = writer.list_partition(other_prefix)

    writer.land_filing(
        cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs(b"new"), overwrite=True
    )

    assert writer.list_partition(other_prefix) == before
    assert all(writer.verify_partition(other_prefix).values()), "neighbouring partition altered"


# --- verification ---------------------------------------------------------
def test_verify_partition_confirms_every_stored_hash(writer: ArchiveWriter) -> None:
    writer.land_filing(cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs())

    assert writer.verify_partition(PREFIX) == {
        "aapl-20230930.htm": True,
        "companyfacts.json": True,
    }


def test_verify_partition_detects_corruption(writer: ArchiveWriter, s3: FakeS3) -> None:
    writer.land_filing(cik=CIK, form="10-K", filed=FILED, accession=ACC, documents=docs())

    s3.store[("filings", PREFIX + "aapl-20230930.htm")] = b"corrupted"

    result = writer.verify_partition(PREFIX)
    assert result["aapl-20230930.htm"] is False
    assert result["companyfacts.json"] is True


def test_verify_partition_of_nothing_is_empty(writer: ArchiveWriter) -> None:
    assert writer.verify_partition("cik=1/form=10-K/filed=2020-01-01/accession=z/") == {}


# --- bucket bootstrap -----------------------------------------------------
def test_ensure_bucket_creates_a_missing_bucket(settings: Any) -> None:
    s3 = FakeS3()
    s3.buckets.clear()

    ArchiveWriter(settings, client=s3, bucket="filings").ensure_bucket()

    assert "filings" in s3.buckets

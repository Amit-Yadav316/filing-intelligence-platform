"""The immutable landing zone.

Raw filings go to object storage exactly as EDGAR served them, partitioned so
that a single filing is one self-describing prefix:

    cik=0000320193/form=10-K/filed=2023-11-03/accession=0000320193-23-000106/
        aapl-20230930.htm          the primary document, byte-for-byte
        companyfacts.json          the XBRL answer key for this filer
        _manifest.json             what was fetched, from where, and its hash

Two properties make this an archive rather than a cache.

**Immutability.** Bytes are stored undecoded and unmodified. Every later stage -
parsing, chunking, extraction - is a pure function of this layer, so a chunker
change is replayed by reprocessing, never by re-fetching. That is what keeps the
character offsets in a citation meaningful years later.

**Idempotency.** Re-running a date range must converge, not accumulate. A
partition whose content hashes already match is left completely untouched,
including its manifest: re-landing a filing does not rewrite its fetch
timestamp, because the honest archival record is when the bytes were *first*
obtained. Only the named partition is ever touched; nothing else is deleted.

If EDGAR serves different bytes for an accession already on disk, that is a
real event - filings do get amended - so it is logged loudly and requires an
explicit ``overwrite`` rather than being silently absorbed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger
from src.observability.metrics import FILINGS_INGESTED

log = get_logger(__name__)

MANIFEST_NAME = "_manifest.json"

LandingStatus = Literal["landed", "unchanged", "overwritten"]


def sha256_hex(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def safe_segment(value: str) -> str:
    """Make a value safe for one path segment.

    Form types are the reason this exists: an amended annual report is filed as
    ``10-K/A``, and the slash would silently split one partition into two.
    """
    cleaned = value.strip().replace("/", "-").replace("\\", "-")
    return "".join(c for c in cleaned if c.isalnum() or c in "-_.") or "unknown"


class ArchivedObject(BaseModel):
    """One stored file, with enough detail to verify it without re-fetching."""

    filename: str
    key: str
    size_bytes: int
    sha256: str
    source_url: str
    content_type: str


class PartitionManifest(BaseModel):
    """The record of one filing's landing. Written once, at first landing."""

    cik: str
    company: str | None = None
    form: str
    filed: date
    accession: str
    prefix: str
    edgar_url: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    objects: list[ArchivedObject] = Field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(o.size_bytes for o in self.objects)

    def by_name(self) -> dict[str, ArchivedObject]:
        return {o.filename: o for o in self.objects}


class ArchiveWriter:
    """Writes raw filings to S3-compatible object storage, idempotently."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any = None,
        bucket: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bucket = bucket or self.settings.minio_bucket
        self._client = client if client is not None else self._build_client(self.settings)

    @staticmethod
    def _build_client(settings: Settings) -> Any:
        import boto3
        from botocore.config import Config

        return boto3.client(
            "s3",
            endpoint_url=settings.minio_endpoint,
            aws_access_key_id=settings.minio_access_key.get_secret_value(),
            aws_secret_access_key=settings.minio_secret_key.get_secret_value(),
            region_name="us-east-1",
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    # --- layout -----------------------------------------------------------
    @staticmethod
    def partition_prefix(cik: str, form: str, filed: date, accession: str) -> str:
        """Hive-style partitioning, so the layout is self-describing on disk and
        readable by engines that understand ``key=value`` path segments."""
        return (
            f"cik={safe_segment(cik)}/"
            f"form={safe_segment(form)}/"
            f"filed={filed.isoformat()}/"
            f"accession={safe_segment(accession)}/"
        )

    @staticmethod
    def edgar_url(cik: str, accession: str) -> str:
        bare = cik.lstrip("0") or "0"
        return f"https://www.sec.gov/Archives/edgar/data/{bare}/{accession.replace('-', '')}/"

    # --- primitives -------------------------------------------------------
    def _put(self, key: str, body: bytes, content_type: str) -> None:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType=content_type)

    def _get(self, key: str) -> bytes | None:
        try:
            return bytes(self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read())
        except Exception as exc:
            if exc.__class__.__name__ in {"NoSuchKey", "ClientError", "404"}:
                return None
            raise

    def read_manifest(self, prefix: str) -> PartitionManifest | None:
        raw = self._get(prefix + MANIFEST_NAME)
        if raw is None:
            return None
        return PartitionManifest.model_validate_json(raw)

    # --- landing ----------------------------------------------------------
    def land_filing(
        self,
        *,
        cik: str,
        form: str,
        filed: date,
        accession: str,
        documents: dict[str, tuple[bytes, str, str]],
        company: str | None = None,
        overwrite: bool = False,
        prefix: str | None = None,
        edgar_url: str | None = None,
    ) -> tuple[PartitionManifest, LandingStatus]:
        """Land one filing's documents into its own partition.

        ``documents`` maps filename to ``(body, source_url, content_type)``.

        ``prefix`` overrides the derived Hive layout, for payloads that belong
        to a company rather than to a single filing - companyfacts is stored
        once per company and reuses this convergence logic rather than
        reimplementing it.

        Returns the manifest and what actually happened. ``unchanged`` means
        every byte already present matched and nothing was written - the state
        a second run over the same date must produce.
        """
        prefix = prefix or self.partition_prefix(cik, form, filed, accession)
        incoming = {
            name: (body, sha256_hex(body), url, ctype)
            for name, (body, url, ctype) in documents.items()
        }

        existing = self.read_manifest(prefix)
        if existing is not None and not overwrite:
            stored = existing.by_name()
            same = {name for name in incoming if name in stored} and all(
                name in stored and stored[name].sha256 == digest
                for name, (_, digest, _, _) in incoming.items()
            )
            if same:
                # Converged. Leave the original manifest, fetch timestamp and
                # all, so re-running a backfill is a genuine no-op.
                log.info(
                    "partition_unchanged",
                    accession=accession,
                    prefix=prefix,
                    objects=len(stored),
                )
                FILINGS_INGESTED.labels(form=form, status="unchanged").inc()
                return existing, "unchanged"

            changed = [
                name
                for name, (_, digest, _, _) in incoming.items()
                if name not in stored or stored[name].sha256 != digest
            ]
            log.warning(
                "partition_content_changed",
                accession=accession,
                prefix=prefix,
                changed=changed,
                action="refusing to overwrite; pass overwrite=True to replace",
            )
            FILINGS_INGESTED.labels(form=form, status="conflict").inc()
            return existing, "unchanged"

        objects: list[ArchivedObject] = []
        for name, (body, digest, url, ctype) in sorted(incoming.items()):
            key = prefix + name
            self._put(key, body, ctype)
            objects.append(
                ArchivedObject(
                    filename=name,
                    key=key,
                    size_bytes=len(body),
                    sha256=digest,
                    source_url=url,
                    content_type=ctype,
                )
            )

        manifest = PartitionManifest(
            cik=cik,
            company=company,
            form=form,
            filed=filed,
            accession=accession,
            prefix=prefix,
            edgar_url=edgar_url or self.edgar_url(cik, accession),
            objects=objects,
        )
        self._put(
            prefix + MANIFEST_NAME,
            manifest.model_dump_json(indent=2).encode("utf-8"),
            "application/json",
        )

        status: LandingStatus = "overwritten" if existing is not None else "landed"
        log.info(
            "partition_landed",
            accession=accession,
            prefix=prefix,
            objects=len(objects),
            bytes=manifest.total_bytes,
            status=status,
        )
        FILINGS_INGESTED.labels(form=form, status=status).inc()
        return manifest, status

    # --- verification -----------------------------------------------------
    def verify_partition(self, prefix: str) -> dict[str, bool]:
        """Re-hash every stored object against its manifest entry.

        An archive nobody checks is a directory. This is what the integration
        test and any future consistency job assert on.
        """
        manifest = self.read_manifest(prefix)
        if manifest is None:
            return {}
        results: dict[str, bool] = {}
        for obj in manifest.objects:
            body = self._get(obj.key)
            results[obj.filename] = body is not None and sha256_hex(body) == obj.sha256
        return results

    def list_partition(self, prefix: str) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        return sorted(keys)

    def ensure_bucket(self) -> None:
        """Create the bucket if it is absent. Safe to call repeatedly."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except Exception:
            self._client.create_bucket(Bucket=self.bucket)
            log.info("bucket_created", bucket=self.bucket)


def json_bytes(payload: dict[str, Any]) -> bytes:
    """Stable JSON encoding, so an unchanged payload hashes identically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

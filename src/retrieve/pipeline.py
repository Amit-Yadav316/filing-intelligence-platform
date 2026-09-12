"""Parse, chunk, embed and index one filing - the process stage as a service.

Day 5's ``process_filings`` DAG is a thin wrapper over this. Keeping the logic
here means the whole stage runs from a shell, which is how it gets debugged.

Skipping unchanged work
-----------------------
The corpus is reprocessed every time the chunker changes, and most chunks come
back byte-identical. Two layers stop that costing anything:

* the **content hash** already stored in Postgres tells us a chunk is unchanged
  and still has a vector, so it is not re-embedded and not rewritten;
* the **Redis cache** catches the remainder - a chunk whose id moved because an
  earlier Item grew, but whose text did not change.

The measured effect is reported per run rather than asserted, because a cache
that silently stops working looks exactly like a cache that is working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.chunk.structural_chunker import Chunk, StructuralChunker
from src.config.settings import Settings, get_settings
from src.embed.embedding_service import EmbeddingService
from src.ingest.archive_writer import ArchiveWriter
from src.observability.logging import get_logger
from src.observability.metrics import STAGE_LATENCY
from src.parse.filing_parser import FilingParser
from src.retrieve.store import ChunkStore

log = get_logger(__name__)


@dataclass
class IndexingResult:
    accession: str
    company: str | None = None
    form: str = ""
    chunks: int = 0
    embedded: int = 0
    skipped_unchanged: int = 0
    written: int = 0
    pruned: int = 0
    error: str | None = None
    items: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


class IndexingPipeline:
    """Archive bytes to indexed, searchable chunks."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        writer: ArchiveWriter | None = None,
        parser: FilingParser | None = None,
        chunker: StructuralChunker | None = None,
        embedder: EmbeddingService | None = None,
        store: ChunkStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.writer = writer or ArchiveWriter(self.settings)
        self.parser = parser or FilingParser()
        self.chunker = chunker or StructuralChunker(self.settings)
        self.embedder = embedder or EmbeddingService(self.settings)
        self.store = store or ChunkStore(self.settings)

    # --- locating a landed filing ----------------------------------------
    def find_manifest(self, accession: str) -> dict[str, Any] | None:
        paginator = self.writer._client.get_paginator("list_objects_v2")
        needle = f"accession={accession}/"
        for page in paginator.paginate(Bucket=self.writer.bucket):
            for item in page.get("Contents", []):
                key = item["Key"]
                if needle in key and key.endswith("_manifest.json"):
                    manifest = self.writer.read_manifest(key[: -len("_manifest.json")])
                    if manifest is not None:
                        return manifest.model_dump(mode="json")
        return None

    def list_landed_filings(self) -> list[str]:
        """Every filing accession currently in the archive."""
        paginator = self.writer._client.get_paginator("list_objects_v2")
        found: set[str] = set()
        for page in paginator.paginate(Bucket=self.writer.bucket):
            for item in page.get("Contents", []):
                key = item["Key"]
                if "accession=" in key and key.endswith("_manifest.json"):
                    acc = key.split("accession=")[1].split("/")[0]
                    if not acc.startswith("companyfacts-"):
                        found.add(acc)
        return sorted(found)

    # --- the stage --------------------------------------------------------
    def index_filing(self, accession: str, *, force: bool = False) -> IndexingResult:
        result = IndexingResult(accession=accession)

        manifest = self.find_manifest(accession)
        if manifest is None:
            result.error = "not landed in the archive"
            return result
        result.company = manifest.get("company")
        result.form = manifest["form"]

        document = next(
            (o for o in manifest["objects"] if o["filename"].endswith((".htm", ".html"))),
            None,
        )
        if document is None:
            result.error = "no HTML document in the partition"
            return result

        raw = self.writer._get(document["key"])
        if raw is None:
            result.error = "object missing from storage"
            return result

        with STAGE_LATENCY.labels(stage="parse").time():
            parsed = self.parser.parse(raw)
        with STAGE_LATENCY.labels(stage="chunk").time():
            chunks, stats = self.chunker.chunk(
                parsed,
                accession=manifest["accession"],
                cik=manifest["cik"],
                form=manifest["form"],
                filed=date.fromisoformat(manifest["filed"]),
            )
        result.chunks = len(chunks)
        result.items = stats.items

        if not chunks:
            result.error = "chunker produced nothing"
            return result

        hashes = self.embedder.content_hashes([c.text for c in chunks])

        pending, pending_hashes = self._select_pending(chunks, hashes, force=force)
        result.skipped_unchanged = len(chunks) - len(pending)

        if pending:
            with STAGE_LATENCY.labels(stage="embed_batch").time():
                vectors = self.embedder.embed_texts([c.text for c in pending])
            result.embedded = len(vectors)
            result.written = self.store.upsert_chunks(
                pending,
                embeddings=vectors,
                company=result.company,
                content_hashes=pending_hashes,
            )

        result.pruned = self.store.prune_filing(accession, [c.chunk_id for c in chunks])

        log.info(
            "filing_indexed",
            accession=accession,
            company=result.company,
            chunks=result.chunks,
            embedded=result.embedded,
            skipped=result.skipped_unchanged,
            pruned=result.pruned,
        )
        return result

    def _select_pending(
        self, chunks: list[Chunk], hashes: list[str], *, force: bool
    ) -> tuple[list[Chunk], list[str]]:
        """Drop chunks already stored with an identical content hash."""
        if force:
            return chunks, hashes
        stored = self.store.existing_hashes([c.chunk_id for c in chunks])
        pending: list[Chunk] = []
        pending_hashes: list[str] = []
        for chunk, digest in zip(chunks, hashes, strict=True):
            if stored.get(chunk.chunk_id) == digest:
                continue
            pending.append(chunk)
            pending_hashes.append(digest)
        return pending, pending_hashes

    def index_all(
        self, accessions: list[str] | None = None, *, force: bool = False
    ) -> list[IndexingResult]:
        targets = accessions if accessions is not None else self.list_landed_filings()
        return [self.index_filing(a, force=force) for a in targets]

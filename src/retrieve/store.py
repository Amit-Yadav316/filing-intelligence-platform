"""Postgres chunk store: schema management and writes.

Reads live in :mod:`src.retrieve.indexes`; this module owns the table, the
connection and getting chunks into it.

Upserts are keyed on ``chunk_id``, which is derived from the accession, the Item
and an ordinal - so reprocessing a filing after a chunker change overwrites that
filing's chunks rather than duplicating them. The pipeline can therefore be
re-run over the same corpus as many times as the chunker changes, which is the
whole point of an immutable raw layer underneath.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.chunk.structural_chunker import Chunk
from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger

log = get_logger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def fiscal_year_for(form: str, filed: date) -> int:
    """Best-effort fiscal year for a filing, for metadata filtering only.

    A filing's true fiscal year comes from its XBRL period and is resolved
    properly by :class:`~src.evaluate.xbrl_resolver.XBRLResolver`. This is a
    filtering convenience, not ground truth, and is deliberately not used for
    scoring: an annual report filed in the second half of a year reports that
    year, one filed in the first half reports the year before.
    """
    if form.upper().startswith("10-K"):
        return filed.year if filed.month >= 7 else filed.year - 1
    return filed.year


@dataclass(frozen=True, slots=True)
class StoreStats:
    chunks: int
    embedded: int
    filings: int
    newest_filed: date | None

    @property
    def unembedded(self) -> int:
        return self.chunks - self.embedded


class ChunkStore:
    """Owns the ``chunks`` table."""

    def __init__(self, settings: Settings | None = None, *, dsn: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.dsn = dsn or self.settings.postgres_dsn
        self._conn: Any = None

    # --- connection -------------------------------------------------------
    def connect(self) -> Any:
        if self._conn is None or getattr(self._conn, "closed", True):
            import psycopg
            from pgvector.psycopg import register_vector

            self._conn = psycopg.connect(self.dsn, autocommit=True)
            # Without this, a vector comes back as a string and every distance
            # calculation silently becomes text comparison.
            register_vector(self._conn)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not getattr(self._conn, "closed", True):
            self._conn.close()
        self._conn = None

    def __enter__(self) -> ChunkStore:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- schema -----------------------------------------------------------
    def apply_schema(self) -> None:
        """Idempotent. Safe on every start-up, not just the first."""
        conn = self.connect()
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        log.info("schema_applied", dsn=self.dsn.rsplit("@", 1)[-1])

    def drop_all(self) -> None:
        """Only for tests and a deliberate rebuild."""
        self.connect().execute("DROP TABLE IF EXISTS chunks CASCADE")

    # --- writes -----------------------------------------------------------
    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
        company: str | None = None,
        content_hashes: Sequence[str] | None = None,
    ) -> int:
        """Insert or replace chunks. Returns the number written."""
        if not chunks:
            return 0
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError(
                f"embeddings ({len(embeddings)}) must align with chunks ({len(chunks)})"
            )
        if content_hashes is not None and len(content_hashes) != len(chunks):
            raise ValueError("content_hashes must align with chunks")

        rows = []
        for i, c in enumerate(chunks):
            rows.append(
                (
                    c.chunk_id,
                    c.accession,
                    c.cik,
                    company,
                    c.form,
                    c.filed,
                    fiscal_year_for(c.form, c.filed),
                    c.item_number,
                    c.item_title,
                    str(c.section),
                    c.chunk_type,
                    c.char_start,
                    c.char_end,
                    c.token_count,
                    c.text,
                    content_hashes[i] if content_hashes else "",
                    c.parser_version,
                    list(embeddings[i]) if embeddings is not None else None,
                )
            )

        conn = self.connect()
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, accession, cik, company, form, filed, fiscal_year,
                    item_number, item_title, section, chunk_type,
                    char_start, char_end, token_count, text,
                    content_hash, parser_version, embedding
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    accession      = EXCLUDED.accession,
                    cik            = EXCLUDED.cik,
                    company        = COALESCE(EXCLUDED.company, chunks.company),
                    form           = EXCLUDED.form,
                    filed          = EXCLUDED.filed,
                    fiscal_year    = EXCLUDED.fiscal_year,
                    item_number    = EXCLUDED.item_number,
                    item_title     = EXCLUDED.item_title,
                    section        = EXCLUDED.section,
                    chunk_type     = EXCLUDED.chunk_type,
                    char_start     = EXCLUDED.char_start,
                    char_end       = EXCLUDED.char_end,
                    token_count    = EXCLUDED.token_count,
                    text           = EXCLUDED.text,
                    content_hash   = EXCLUDED.content_hash,
                    parser_version = EXCLUDED.parser_version,
                    -- Keep an existing vector when the caller supplies none, so
                    -- a metadata-only rewrite does not silently unindex a chunk.
                    embedding      = COALESCE(EXCLUDED.embedding, chunks.embedding),
                    updated_at     = now()
                """,
                rows,
            )
        log.info("chunks_upserted", count=len(rows), embedded=embeddings is not None)
        return len(rows)

    def set_embeddings(self, pairs: Iterable[tuple[str, Sequence[float]]]) -> int:
        """Attach vectors to chunks already stored."""
        rows = [(list(vec), chunk_id) for chunk_id, vec in pairs]
        if not rows:
            return 0
        with self.connect().cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET embedding = %s, updated_at = now() WHERE chunk_id = %s",
                rows,
            )
        return len(rows)

    def delete_filing(self, accession: str) -> int:
        with self.connect().cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE accession = %s", (accession,))
            return int(cur.rowcount)

    def prune_filing(self, accession: str, keep_chunk_ids: Sequence[str]) -> int:
        """Delete chunks for a filing that the current chunker no longer emits.

        Without this, changing the chunker leaves the previous run's chunks
        behind under their old ids. They keep their embeddings, stay
        retrievable, and quietly compete with the correct chunks - a stale
        result that nothing reports as stale.
        """
        if not keep_chunk_ids:
            return 0
        with self.connect().cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE accession = %s AND NOT (chunk_id = ANY(%s))",
                (accession, list(keep_chunk_ids)),
            )
            return int(cur.rowcount)

    # --- reads ------------------------------------------------------------
    def stats(self) -> StoreStats:
        row = (
            self.connect()
            .execute(
                """
            SELECT count(*),
                   count(embedding),
                   count(DISTINCT accession),
                   max(filed)
            FROM chunks
            """
            )
            .fetchone()
        )
        return StoreStats(chunks=row[0], embedded=row[1], filings=row[2], newest_filed=row[3])

    def existing_hashes(self, chunk_ids: Sequence[str]) -> dict[str, str]:
        """Content hashes already stored, so unchanged chunks can be skipped."""
        if not chunk_ids:
            return {}
        rows = (
            self.connect()
            .execute(
                "SELECT chunk_id, content_hash FROM chunks WHERE chunk_id = ANY(%s)",
                (list(chunk_ids),),
            )
            .fetchall()
        )
        return {r[0]: r[1] for r in rows}

    def index_freshness_seconds(self, now: date | None = None) -> float | None:
        """Now minus the filing date of the newest indexed filing.

        This is the SLO in ``docs/SLOS.md``, computed from the index itself
        rather than from a pipeline run's self-report - a DAG that fails without
        alerting would otherwise keep reporting a freshness it no longer has.
        """
        newest = self.stats().newest_filed
        if newest is None:
            return None
        return ((now or date.today()) - newest).total_seconds()

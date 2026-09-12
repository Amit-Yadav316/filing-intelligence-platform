"""Lexical and dense retrieval over the chunk store.

Both arms read the same table, so a metadata pre-filter is a ``WHERE`` clause
rather than a cross-system join, and both return the same :class:`SearchHit`
shape so fusion has nothing to reconcile.

A note on naming, because the honest version matters more than the familiar one
------------------------------------------------------------------------------
:class:`BM25Index` ranks with Postgres ``ts_rank_cd``, which is **cover density
ranking, not Okapi BM25**. It shares BM25's important properties for this use -
term frequency saturation, rewarding proximity of query terms - but it does not
implement BM25's document-length normalisation or its IDF formulation, and it
will not reproduce BM25 scores.

The class keeps the name because that is what the retrieval arm *is* in the
architecture, and the README says so plainly. Getting true BM25 in Postgres
needs an extension that cannot be assumed present on a stock image; the honest
trade is a built-in ranker plus an accurate description of it, measured in the
ablation, rather than a claim the code does not support.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger
from src.observability.metrics import RETRIEVAL_LATENCY
from src.retrieve.store import ChunkStore

log = get_logger(__name__)

_SELECT_COLUMNS = """
    chunk_id, accession, cik, company, form, filed, fiscal_year,
    item_number, item_title, section, chunk_type,
    char_start, char_end, token_count, text
"""


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One retrieved chunk, carrying the full provenance chain."""

    chunk_id: str
    score: float
    rank: int
    accession: str
    cik: str
    company: str | None
    form: str
    filed: date
    fiscal_year: int | None
    item_number: str | None
    item_title: str | None
    section: str | None
    chunk_type: str
    char_start: int
    char_end: int
    token_count: int
    text: str
    retriever: str = "unknown"

    @property
    def edgar_url(self) -> str:
        bare = self.cik.lstrip("0") or "0"
        return f"https://www.sec.gov/Archives/edgar/data/{bare}/{self.accession.replace('-', '')}/"

    def citation(self) -> dict[str, Any]:
        """The shape the API returns, so a caller can trace any answer."""
        return {
            "chunk_id": self.chunk_id,
            "accession": self.accession,
            "cik": self.cik,
            "company": self.company,
            "form": self.form,
            "filed": self.filed.isoformat(),
            "item": f"Item {self.item_number}" if self.item_number else None,
            "section": self.section,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "edgar_url": self.edgar_url,
        }


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """Narrows the candidate set before ranking.

    Pre-filtering matters more than it looks. "What did Apple say about supply
    chain risk in 2023" should never rank a Microsoft chunk at all, and
    filtering after ranking would spend the top-k budget on documents the user
    already excluded.
    """

    cik: str | None = None
    ciks: Sequence[str] = field(default_factory=tuple)
    form: str | None = None
    fiscal_year: int | None = None
    year_from: int | None = None
    year_to: int | None = None
    section: str | None = None
    item_number: str | None = None
    chunk_type: str | None = None
    accession: str | None = None

    def to_sql(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.cik:
            clauses.append("cik = %s")
            params.append(self.cik.zfill(10))
        if self.ciks:
            clauses.append("cik = ANY(%s)")
            params.append([c.zfill(10) for c in self.ciks])
        if self.form:
            clauses.append("upper(form) = upper(%s)")
            params.append(self.form)
        if self.fiscal_year is not None:
            clauses.append("fiscal_year = %s")
            params.append(self.fiscal_year)
        if self.year_from is not None:
            clauses.append("fiscal_year >= %s")
            params.append(self.year_from)
        if self.year_to is not None:
            clauses.append("fiscal_year <= %s")
            params.append(self.year_to)
        if self.section:
            clauses.append("section = %s")
            params.append(self.section)
        if self.item_number:
            clauses.append("upper(item_number) = upper(%s)")
            params.append(self.item_number)
        if self.chunk_type:
            clauses.append("chunk_type = %s")
            params.append(self.chunk_type)
        if self.accession:
            clauses.append("accession = %s")
            params.append(self.accession)
        return (" AND ".join(clauses) if clauses else "TRUE"), params

    @property
    def is_empty(self) -> bool:
        return self.to_sql()[0] == "TRUE"


def _row_to_hit(row: Sequence[Any], score: float, rank: int, retriever: str) -> SearchHit:
    return SearchHit(
        chunk_id=row[0],
        score=score,
        rank=rank,
        accession=row[1],
        cik=row[2],
        company=row[3],
        form=row[4],
        filed=row[5],
        fiscal_year=row[6],
        item_number=row[7],
        item_title=row[8],
        section=row[9],
        chunk_type=row[10],
        char_start=row[11],
        char_end=row[12],
        token_count=row[13],
        text=row[14],
        retriever=retriever,
    )


class BM25Index:
    """Lexical retrieval via Postgres full-text search.

    See the module docstring: this is ``ts_rank_cd`` cover-density ranking, not
    Okapi BM25.
    """

    name = "bm25"

    def __init__(self, store: ChunkStore, settings: Settings | None = None) -> None:
        self.store = store
        self.settings = settings or get_settings()

    def search(
        self, query: str, *, k: int | None = None, filters: MetadataFilter | None = None
    ) -> list[SearchHit]:
        k = k or self.settings.retrieval_candidate_k
        where, params = (filters or MetadataFilter()).to_sql()

        # websearch_to_tsquery accepts natural input - quotes, OR, leading
        # minus - and, unlike to_tsquery, does not raise on punctuation a user
        # would reasonably type.
        sql = f"""
            SELECT {_SELECT_COLUMNS},
                   ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) AS score
            FROM chunks
            WHERE tsv @@ websearch_to_tsquery('english', %s)
              AND {where}
            ORDER BY score DESC, chunk_id
            LIMIT %s
        """
        with RETRIEVAL_LATENCY.labels(mode=self.name).time():
            rows = self.store.connect().execute(sql, [query, query, *params, k]).fetchall()
        return [_row_to_hit(r, float(r[-1]), i + 1, self.name) for i, r in enumerate(rows)]


class VectorIndex:
    """Dense retrieval via pgvector, cosine distance."""

    name = "dense"

    def __init__(self, store: ChunkStore, settings: Settings | None = None) -> None:
        self.store = store
        self.settings = settings or get_settings()

    def search(
        self,
        embedding: Sequence[float],
        *,
        k: int | None = None,
        filters: MetadataFilter | None = None,
    ) -> list[SearchHit]:
        k = k or self.settings.retrieval_candidate_k
        where, params = (filters or MetadataFilter()).to_sql()

        # `<=>` is cosine distance, so smaller is closer. Reporting similarity
        # keeps "higher is better" true across both arms, which fusion assumes.
        sql = f"""
            SELECT {_SELECT_COLUMNS},
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE embedding IS NOT NULL
              AND {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        vec = list(embedding)
        with RETRIEVAL_LATENCY.labels(mode=self.name).time():
            rows = self.store.connect().execute(sql, [vec, *params, vec, k]).fetchall()
        return [_row_to_hit(r, float(r[-1]), i + 1, self.name) for i, r in enumerate(rows)]

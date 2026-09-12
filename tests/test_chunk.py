"""Chunker and section-tagger tests.

Two properties carry the weight here, because both fail silently:

* every chunk is an exact slice of the parsed text, or citations point at the
  wrong bytes;
* no chunk spans an Item boundary, or a query about litigation risk retrieves
  a chunk that is half Properties.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.chunk.section_tagger import Section, SectionTagger
from src.chunk.structural_chunker import (
    Chunk,
    StructuralChunker,
    approx_token_count,
    item_sort_key,
)
from src.config.settings import Settings
from src.parse.filing_parser import FilingParser

FILED = date(2023, 11, 3)
ACC = "0000320193-23-000106"
CIK = "0000320193"


def build_filing(items: dict[str, str], *, toc: bool = True) -> bytes:
    """A filing shaped like a real one: a table-of-contents table, then body."""
    parts = ["<html><body>"]
    if toc:
        parts.append("<table>")
        for n, title in items.items():
            parts.append(f"<tr><td>Item {n}. {title}</td><td>{len(parts)}</td></tr>")
        parts.append("</table>")
    for n, title in items.items():
        parts.append(f"<div>Item {n}. {title}</div>")
        parts.append(f"<p>{('Prose about ' + title + '. ') * 40}</p>")
    parts.append("</body></html>")
    return "".join(parts).encode()


ITEMS = {
    "1": "Business",
    "1A": "Risk Factors",
    "7": "Management's Discussion and Analysis",
    "8": "Financial Statements",
}


@pytest.fixture
def settings_small() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        edgar_user_agent="Test Harness test@example.org",
        chunk_token_window=120,
        chunk_token_overlap=20,
        chunk_min_tokens=5,
    )


@pytest.fixture
def chunker(settings_small: Settings) -> StructuralChunker:
    return StructuralChunker(settings_small)


def run(chunker: StructuralChunker, raw: bytes) -> tuple[list[Chunk], object, object]:
    parsed = FilingParser().parse(raw)
    chunks, stats = chunker.chunk(parsed, accession=ACC, cik=CIK, form="10-K", filed=FILED)
    return chunks, stats, parsed


# --- token estimation -----------------------------------------------------
def test_token_estimate_scales_with_length() -> None:
    assert approx_token_count("") == 0
    assert approx_token_count("one two three") > 0
    assert approx_token_count("word " * 100) > approx_token_count("word " * 10)


@pytest.mark.parametrize(
    ("items", "expected"),
    [(["1", "1A", "2"], ["1", "1A", "2"]), (["10", "2", "1A"], ["1A", "2", "10"])],
)
def test_items_sort_in_document_order(items: list[str], expected: list[str]) -> None:
    """Item 10 comes after Item 2, which string sorting gets wrong."""
    assert sorted(items, key=item_sort_key) == expected


# --- structure ------------------------------------------------------------
def test_finds_every_item_and_ignores_the_table_of_contents(
    chunker: StructuralChunker,
) -> None:
    """The TOC lists every Item before the body. Because it is a table, the
    parser isolates it and prose-only scanning skips it - no page-number or
    density heuristic needed."""
    _, stats, _ = run(chunker, build_filing(ITEMS, toc=True))

    assert stats.items == ["1", "1A", "7", "8"]  # type: ignore[attr-defined]


def test_item_detection_is_unaffected_by_a_missing_toc(
    chunker: StructuralChunker,
) -> None:
    _, with_toc, _ = run(chunker, build_filing(ITEMS, toc=True))
    _, without, _ = run(chunker, build_filing(ITEMS, toc=False))

    assert with_toc.items == without.items  # type: ignore[attr-defined]


def test_no_chunk_spans_an_item_boundary(chunker: StructuralChunker) -> None:
    chunks, _, parsed = run(chunker, build_filing(ITEMS))
    regions = {r.item_number: r for r in chunker.find_item_regions(parsed)}

    for chunk in chunks:
        if chunk.item_number is None:
            continue
        region = regions[chunk.item_number]
        assert region.char_start <= chunk.char_start
        assert chunk.char_end <= region.char_end


def test_every_chunk_is_an_exact_slice_of_the_text(chunker: StructuralChunker) -> None:
    chunks, _, parsed = run(chunker, build_filing(ITEMS))

    for chunk in chunks:
        assert parsed.text[chunk.char_start : chunk.char_end] == chunk.text


def test_chunks_respect_the_token_budget(
    chunker: StructuralChunker, settings_small: Settings
) -> None:
    """A paragraph-only splitter produced 17,000-token chunks on a real filing,
    because normalised text often has no blank lines inside an Item."""
    chunks, _, _ = run(chunker, build_filing(ITEMS))

    oversized = [
        c
        for c in chunks
        if c.chunk_type == "prose" and c.token_count > settings_small.chunk_token_window * 1.1
    ]
    assert not oversized, f"{len(oversized)} prose chunks exceeded the window"


def test_a_run_on_paragraph_still_gets_split(settings_small: Settings) -> None:
    """No blank lines, no newlines, no sentence ends - the hard split is the
    only thing that stops this becoming one enormous chunk."""
    raw = (
        "<html><body><div>Item 1. Business</div><p>" + ("x" * 40 + " ") * 400 + "</p></body></html>"
    ).encode()

    chunks, _, _ = run(StructuralChunker(settings_small), raw)

    assert len(chunks) > 1
    assert max(c.token_count for c in chunks) <= settings_small.chunk_token_window * 1.5


def test_one_block_spanning_several_items_is_attributed_to_each(
    chunker: StructuralChunker,
) -> None:
    """Regression. The parser splits blocks on markup, not on Item headings, so
    a filing with no tables is ONE prose block containing every Item. Taking the
    region at the block's start and applying it to the whole block labelled
    everything from the cover page through Item 4 as "no item" - seven sections
    of Apple's 10-K vanished into 69 orphaned chunks.
    """
    raw = build_filing(ITEMS, toc=False)
    chunks, _, parsed = run(chunker, raw)

    assert len(parsed.prose_blocks) == 1, "fixture must produce a single block"

    attributed = {c.item_number for c in chunks if c.item_number}
    assert attributed == set(ITEMS), f"items lost during attribution: {set(ITEMS) - attributed}"


def test_only_pre_item_text_is_left_unattributed(chunker: StructuralChunker) -> None:
    """Chunks before the first Item heading are the cover page, and those are
    legitimately item-less. Anything after it must belong to a section."""
    chunks, _, parsed = run(chunker, build_filing(ITEMS, toc=False))
    first_item_start = min(r.char_start for r in chunker.find_item_regions(parsed))

    orphans_after_first_item = [
        c for c in chunks if c.item_number is None and c.char_start >= first_item_start
    ]
    assert not orphans_after_first_item, (
        f"{len(orphans_after_first_item)} chunks inside a section carry no item number"
    )


# --- tables ---------------------------------------------------------------
def test_a_table_becomes_one_chunk_and_is_never_split(
    chunker: StructuralChunker,
) -> None:
    rows = "".join(
        f"<tr><td>Row {i}</td><td>{i * 1000:,}</td><td>{i * 900:,}</td></tr>" for i in range(60)
    )
    raw = (
        "<html><body><div>Item 8. Financial Statements</div>"
        f"<table><tr><td></td><td>2023</td><td>2022</td></tr>{rows}</table>"
        "</body></html>"
    ).encode()

    chunks, _, _ = run(chunker, raw)

    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 1, "a financial statement was split across chunks"
    assert "Row 59" in tables[0].text and "Row 0" in tables[0].text


# --- metadata -------------------------------------------------------------
def test_chunks_carry_the_full_provenance_chain(chunker: StructuralChunker) -> None:
    chunks, _, _ = run(chunker, build_filing(ITEMS))
    chunk = next(c for c in chunks if c.item_number == "7")

    assert chunk.chunk_id.startswith(f"{ACC}::item7::c")
    assert (chunk.accession, chunk.cik, chunk.form, chunk.filed) == (ACC, CIK, "10-K", FILED)
    assert chunk.section == Section.MDA
    assert chunk.char_end > chunk.char_start
    assert chunk.token_count > 0
    assert chunk.parser_version
    assert chunk.edgar_url == ("https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/")


def test_chunk_ids_are_unique(chunker: StructuralChunker) -> None:
    chunks, _, _ = run(chunker, build_filing(ITEMS))
    ids = [c.chunk_id for c in chunks]

    assert len(ids) == len(set(ids))


# --- section tagging ------------------------------------------------------
@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("1", Section.BUSINESS),
        ("1A", Section.RISK_FACTORS),
        ("7", Section.MDA),
        ("7A", Section.MARKET_RISK),
        ("8", Section.FINANCIAL_STATEMENTS),
        ("9A", Section.CONTROLS),
        ("99", Section.UNKNOWN),
    ],
)
def test_annual_item_tagging(item: str, expected: Section) -> None:
    assert SectionTagger().tag(item, "10-K") == expected


def test_quarterly_items_are_not_read_with_the_annual_table() -> None:
    """A 10-Q's Item 1 is the financial statements, not the business
    description. One shared table would mislabel every quarterly filing."""
    tagger = SectionTagger()

    assert tagger.tag("1", "10-K") == Section.BUSINESS
    assert tagger.tag("1", "10-Q") == Section.QUARTERLY_FINANCIALS


def test_missing_item_is_unknown_not_an_error() -> None:
    assert SectionTagger().tag(None) == Section.UNKNOWN


@pytest.mark.parametrize("raw", ["Item 7", "item 7", " ITEM 7. ", "7"])
def test_item_normalisation_is_forgiving(raw: str) -> None:
    assert SectionTagger().tag(raw, "10-K") == Section.MDA

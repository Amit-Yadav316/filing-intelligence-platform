"""Parser and table-extractor tests.

The table cases are built from the real shapes found in EDGAR markup - spacer
cells, a currency symbol in its own cell on only some rows, colspans - because
those are what turn a financial statement into numeric soup, and a synthetic
tidy table would not exercise any of it.
"""

from __future__ import annotations

import warnings

import pytest
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from src.parse.filing_parser import (
    PARSER_VERSION,
    FilingParser,
    decode_filing,
    normalise_whitespace,
)
from src.parse.table_extractor import TableExtractor

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def soup_table(html: str):
    return BeautifulSoup(html, "lxml").find("table")


# --- decoding -------------------------------------------------------------
def test_utf8_is_preferred_over_cp1252() -> None:
    """cp1252 accepts any byte sequence, so trying it first would silently
    mangle a UTF-8 document. UTF-8 must be attempted strictly, first."""
    raw = "Registrant’s “business”".encode()

    text, encoding = decode_filing(raw)

    assert encoding == "utf-8"
    assert "’" in text


def test_falls_back_to_cp1252_for_legacy_filings() -> None:
    raw = b"Registrant\x92s results"  # 0x92 is a cp1252 smart quote, invalid UTF-8

    text, encoding = decode_filing(raw)

    assert encoding == "cp1252"
    assert "’" in text


def test_normalise_collapses_runs_but_keeps_paragraphs() -> None:
    assert normalise_whitespace("a  \t b\n\n\n\nc") == "a b\n\nc"


def test_normalise_converts_unicode_line_separators() -> None:
    """U+2028 appears in real EDGAR HTML and is not whitespace to strip()."""
    assert normalise_whitespace("a\u2028b") == "a\nb"


# --- table extraction -----------------------------------------------------
SEGMENT_TABLE = """
<table>
  <tr><td></td><td>2023</td><td></td><td>2022</td></tr>
  <tr><td>Americas</td><td>$</td><td>162,560</td><td></td><td>$</td><td>169,658</td></tr>
  <tr><td>Europe</td><td>94,294</td><td></td><td>95,118</td></tr>
</table>
"""


def test_currency_symbol_is_reattached_to_its_amount() -> None:
    """The $ sits in its own cell on the first row only. Left alone it shifts
    that row one column right of every other row, breaking year alignment."""
    table = TableExtractor().extract(soup_table(SEGMENT_TABLE))

    assert "$162,560" in table.markdown
    assert "| $ |" not in table.markdown


def test_rows_align_after_symbol_merging() -> None:
    table = TableExtractor().extract(soup_table(SEGMENT_TABLE))
    rows = [r for r in table.markdown.splitlines() if r.startswith("|")]
    widths = {r.count("|") for r in rows}

    assert len(widths) == 1, f"ragged table: differing column counts {widths}"


def test_empty_spacer_columns_are_dropped() -> None:
    table = TableExtractor().extract(soup_table(SEGMENT_TABLE))

    assert "|  |  |" not in table.markdown


def test_percent_is_attached_to_the_preceding_number() -> None:
    html = (
        "<table><tr><td>Growth</td><td>(4)</td><td>%</td></tr>"
        "<tr><td>x</td><td>1</td><td>%</td></tr></table>"
    )

    assert "(4)%" in TableExtractor().extract(soup_table(html)).markdown


def test_colspan_is_expanded_into_real_columns() -> None:
    html = (
        "<table><tr><td colspan='2'>Wide</td><td>B</td></tr>"
        "<tr><td>1</td><td>2</td><td>3</td></tr></table>"
    )
    table = TableExtractor().extract(soup_table(html))

    assert table.n_cols >= 3


def test_pipe_in_a_cell_is_escaped() -> None:
    html = "<table><tr><td>a|b</td><td>1</td></tr><tr><td>c</td><td>2</td></tr></table>"

    assert r"a\|b" in TableExtractor().extract(soup_table(html)).markdown


def test_single_row_table_is_layout_not_data() -> None:
    """Filers wrap page headers and paragraphs in tables constantly. Treating
    those as data tables would bury real statements under one-cell noise."""
    html = "<table><tr><td>Apple Inc. | Form 10-K | 2023</td></tr></table>"

    assert TableExtractor.looks_like_layout(soup_table(html)) is True


def test_numeric_table_is_data_not_layout() -> None:
    assert TableExtractor.looks_like_layout(soup_table(SEGMENT_TABLE)) is False


def test_nbsp_only_cell_counts_as_empty() -> None:
    """&nbsp; is not whitespace to str.strip(), so a spacer cell full of them
    looks populated and defeats empty-column removal."""
    html = (
        "<table><tr><td>A</td><td>&#160;</td><td>1</td></tr>"
        "<tr><td>B</td><td>&#160;</td><td>2</td></tr></table>"
    )
    table = TableExtractor().extract(soup_table(html))

    assert table.n_cols == 2


# --- the parser -----------------------------------------------------------
FILING_HTML = """
<html><body>
  <ix:header><xbrli:context id="c1">METADATA</xbrli:context></ix:header>
  <div>Item 1. Business</div>
  <p>We design and sell devices.</p>
  <table>
    <tr><td></td><td>2023</td><td>2022</td></tr>
    <tr><td>Net sales</td><td>383,285</td><td>394,328</td></tr>
    <tr><td>Cost</td><td>214,137</td><td>223,546</td></tr>
  </table>
  <div>Item 1A. Risk Factors</div>
  <p>Our business is subject to risks.</p>
</body></html>
"""


@pytest.fixture
def parsed():
    return FilingParser().parse(FILING_HTML.encode())


def test_every_block_is_a_slice_of_the_canonical_text(parsed) -> None:
    """The invariant the whole provenance chain rests on."""
    for block in parsed.blocks:
        assert parsed.text[block.char_start : block.char_end] == block.text


def test_inline_xbrl_header_metadata_is_removed(parsed) -> None:
    """ix:header holds hundreds of context and unit definitions. They render as
    nothing and would embed as pure noise."""
    assert "METADATA" not in parsed.text


def test_tables_become_their_own_blocks(parsed) -> None:
    assert len(parsed.table_blocks) == 1
    assert "383,285" in parsed.table_blocks[0].text
    assert parsed.table_blocks[0].text.startswith("|")


def test_table_content_does_not_leak_into_prose_blocks(parsed) -> None:
    assert all("383,285" not in b.text for b in parsed.prose_blocks)


def test_document_order_is_preserved(parsed) -> None:
    positions = [b.char_start for b in parsed.blocks]
    assert positions == sorted(positions)
    assert parsed.text.index("Business") < parsed.text.index("Risk Factors")


def test_parse_is_deterministic() -> None:
    a = FilingParser().parse(FILING_HTML.encode())
    b = FilingParser().parse(FILING_HTML.encode())

    assert a.text == b.text
    assert [x.char_start for x in a.blocks] == [x.char_start for x in b.blocks]


def test_result_carries_provenance(parsed) -> None:
    assert parsed.parser_version == PARSER_VERSION
    assert parsed.source_bytes == len(FILING_HTML.encode())
    assert len(parsed.source_sha256) == 64


def test_empty_document_does_not_crash() -> None:
    result = FilingParser().parse(b"<html><body></body></html>")

    assert result.text == ""
    assert result.blocks == ()

"""HTML filing to canonical clean text, with tables kept whole.

Everything downstream - chunk offsets, citations, the character range in an API
response - is expressed against the ``text`` this parser produces. That makes
the parser part of the provenance chain rather than a preprocessing detail, so
two properties are deliberate:

**Determinism.** The same archived bytes always produce the same text, so an
offset recorded today still resolves next year. ``PARSER_VERSION`` is carried on
every result; when the parser changes in a way that moves offsets, that version
changes and previously stored offsets are known to be stale rather than
silently wrong.

**Interleaving.** Tables are lifted out, serialised to markdown, and put back in
document order as their own blocks. A financial statement is therefore never
split across chunk boundaries and never flattened into the prose around it.

The mechanism is a sentinel substitution: each data table is replaced in the DOM
by a marker, the document is flattened to text once, and the marker positions
tell us exactly where each table belongs in the final string.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from bs4.element import NavigableString

from src.observability.logging import get_logger
from src.parse.table_extractor import ExtractedTable, TableExtractor

log = get_logger(__name__)

# Bump when a change moves character offsets. Stored chunks carry it, so a
# mismatch is detectable instead of producing citations that point nowhere.
PARSER_VERSION = "1.0.0"

BlockKind = Literal["prose", "table"]

_SENTINEL = "\x00TBL{}\x00"
_SENTINEL_RE = re.compile(r"\x00TBL(\d+)\x00")

# Inline XBRL carries a header block of context, unit and fact definitions -
# hundreds of elements of pure machine metadata. It is invisible when rendered
# and must not reach the text, or every filing begins with a wall of
# identifiers that embeds as meaningless noise.
_DROP_TAGS = (
    "script",
    "style",
    "ix:header",
    "ix:references",
    "ix:resources",
    "ix:hidden",
    "xbrli:context",
    "xbrli:unit",
    "link",
    "meta",
)


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A contiguous span of the canonical text.

    Invariant, asserted in the tests: ``parsed.text[char_start:char_end]``
    equals ``text``.
    """

    text: str
    kind: BlockKind
    char_start: int
    char_end: int
    table_index: int | None = None

    def __len__(self) -> int:
        return self.char_end - self.char_start


@dataclass(frozen=True, slots=True)
class ParsedFiling:
    """The parse result: canonical text plus where everything came from."""

    text: str
    blocks: tuple[TextBlock, ...]
    tables: tuple[ExtractedTable, ...]
    encoding: str
    source_bytes: int
    source_sha256: str
    parser_version: str = PARSER_VERSION

    @property
    def prose_blocks(self) -> tuple[TextBlock, ...]:
        return tuple(b for b in self.blocks if b.kind == "prose")

    @property
    def table_blocks(self) -> tuple[TextBlock, ...]:
        return tuple(b for b in self.blocks if b.kind == "table")

    @property
    def n_chars(self) -> int:
        return len(self.text)


def decode_filing(raw: bytes) -> tuple[str, str]:
    """Decode filing bytes, returning the text and the encoding used.

    EDGAR filings frequently declare no charset. Modern inline-XBRL documents
    are UTF-8; older ones are Windows-1252. Trying UTF-8 strictly first is the
    reliable test, because real cp1252 text containing smart quotes is almost
    always invalid UTF-8, while the reverse is not true - cp1252 accepts any
    byte sequence and would silently mangle a UTF-8 document.
    """
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def normalise_whitespace(text: str) -> str:
    """Collapse rendering whitespace without destroying paragraph structure.

    U+2028 and U+2029 are written as escapes deliberately. EDGAR HTML contains
    both, and a literal one in source is invisible in an editor and silently
    splits the line for anything using str.splitlines().
    """
    text = (
        text.replace("\xa0", " ")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n\n")
        .replace("\r\n", "\n")
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    # Three or more newlines carry no more meaning than a paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class FilingParser:
    """Parses one filing document into canonical text and blocks."""

    def __init__(self, table_extractor: TableExtractor | None = None) -> None:
        self.tables = table_extractor or TableExtractor()

    def parse(self, raw: bytes) -> ParsedFiling:
        text_html, encoding = decode_filing(raw)
        digest = hashlib.sha256(raw).hexdigest()

        with warnings.catch_warnings():
            # Inline XBRL declares XML namespaces, so bs4 suggests an XML
            # parser. The HTML parser is the correct choice here - the document
            # is served and rendered as HTML - so the advice does not apply.
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(text_html, "lxml")

        for tag_name in _DROP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        extracted: list[ExtractedTable] = []
        for table in soup.find_all("table"):
            if self.tables.looks_like_layout(table):
                # Leave it in place; its text flows into the prose around it.
                continue
            result = self.tables.extract(table, index=len(extracted))
            if result.is_empty:
                table.decompose()
                continue
            extracted.append(result)
            table.replace_with(NavigableString(_SENTINEL.format(result.index)))

        flat = soup.get_text("\n")
        blocks, text = self._assemble(flat, extracted)

        log.info(
            "filing_parsed",
            chars=len(text),
            blocks=len(blocks),
            tables=len(extracted),
            encoding=encoding,
        )
        return ParsedFiling(
            text=text,
            blocks=tuple(blocks),
            tables=tuple(extracted),
            encoding=encoding,
            source_bytes=len(raw),
            source_sha256=digest,
        )

    def _assemble(self, flat: str, extracted: list[ExtractedTable]) -> tuple[list[TextBlock], str]:
        """Interleave prose and serialised tables, recording exact offsets."""
        by_index = {t.index: t for t in extracted}
        blocks: list[TextBlock] = []
        parts: list[str] = []
        cursor = 0

        def emit(content: str, kind: BlockKind, table_index: int | None = None) -> None:
            nonlocal cursor
            if not content:
                return
            if parts:
                parts.append("\n\n")
                cursor += 2
            start = cursor
            parts.append(content)
            cursor += len(content)
            blocks.append(
                TextBlock(
                    text=content,
                    kind=kind,
                    char_start=start,
                    char_end=cursor,
                    table_index=table_index,
                )
            )

        last = 0
        for match in _SENTINEL_RE.finditer(flat):
            emit(normalise_whitespace(flat[last : match.start()]), "prose")
            table = by_index.get(int(match.group(1)))
            if table is not None:
                emit(table.markdown, "table", table.index)
            last = match.end()

        emit(normalise_whitespace(flat[last:]), "prose")
        return blocks, "".join(parts)

    def parse_document(self, raw: bytes) -> ParsedFiling:
        """Alias kept for readability at call sites."""
        return self.parse(raw)


def is_probably_exhibit(filename: str) -> bool:
    """Exhibits are separate documents and are not part of the main filing body.

    They are excluded at ingest by primary-document selection, so this exists
    for callers handling a whole filing directory.
    """
    low = filename.lower()
    return bool(re.search(r"exhibit|(^|[^a-z0-9])ex-?\d", low))

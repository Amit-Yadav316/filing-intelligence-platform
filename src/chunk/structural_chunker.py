"""Structural chunking: Item boundaries first, token windows second.

A 512-token sliding window over a 10-K is the obvious approach and it destroys
retrieval quality, because it cuts across the legal structure of the document.
Half of Risk Factors ends up in the same vector as the start of Properties, and
a query about litigation risk retrieves a chunk that is mostly about real
estate.

So sections come first. Item boundaries are detected, each Item becomes a
region, and only *within* a region is a token window applied. No chunk ever
spans an Item boundary - that is asserted in the tests, not just intended.

Finding the Item headings
-------------------------
The hard part is not the regex, it is the table of contents. A 10-K lists every
Item with a page number before the body starts, so a naive scan finds each Item
twice and puts every section boundary inside the first page.

The parser solves this for free. A table of contents *is* a table, so it is
lifted into its own block during parsing, and restricting heading detection to
prose blocks excludes it without any heuristic about density or page numbers.
On Apple's FY2023 10-K this yields exactly the 23 real Item headings, in order,
with no false positives and nothing missed.

Offsets
-------
Every chunk is a contiguous slice of the parsed text, so
``parsed.text[chunk.char_start:chunk.char_end] == chunk.text`` holds even for
overlapping chunks. That is what makes a citation resolvable back to a byte
range in a specific filing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from src.chunk.section_tagger import Section, SectionTagger
from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger
from src.parse.filing_parser import PARSER_VERSION, ParsedFiling

log = get_logger(__name__)

ChunkType = Literal["prose", "table"]

# An Item heading sits at the start of its own line. Anchoring to line start is
# what separates a heading from an in-sentence cross-reference such as
# "as discussed in Item 1A above", which would otherwise open a bogus section.
# A real Item heading carries punctuation after the number, a title on the same
# line, or both. A bare "Item 1" alone on a line is a running page header, and
# filers repeat those on every page: Microsoft's FY2023 10-K contains 21 of them
# for Item 1 and 42 for Item 8. Treating those as section starts shattered the
# document into 128 regions instead of 22.
#
# Anchoring to line start additionally excludes in-sentence cross-references
# such as "as discussed in Item 1A above".
ITEM_HEADING = re.compile(
    r"^[ \t]*Item[ \t\u00a0]+([0-9]{1,2}[A-C]?)"
    r"(?:[ \t]*[.:\u2014\u2013-]+[ \t]*(.{0,120})"
    r"|[ \t]+(\S.{0,119}))$",
    re.IGNORECASE | re.MULTILINE,
)

# Part headings reset the Item sequence: a 10-Q runs Items 1-4 in Part I and
# then Items 1-6 again in Part II, so a document-wide "must increase" rule
# would discard the whole of Part II.
PART_HEADING = re.compile(r"^[ \t]*PART[ \t\u00a0]+([IVX]{1,4})\b", re.IGNORECASE | re.MULTILINE)


_ITEM_ORDER_RE = re.compile(r"^(\d+)([A-C]?)$")


def item_sort_key(item: str) -> tuple[int, str]:
    """Order Items the way the document does: 1, 1A, 1B, 2, ... 10, 11."""
    m = _ITEM_ORDER_RE.match(item.upper())
    if not m:
        return (999, item.upper())
    return (int(m.group(1)), m.group(2))


def approx_token_count(text: str) -> int:
    """Estimate tokens without loading a tokenizer.

    Deliberately an approximation. The real ``bge-small-en-v1.5`` tokenizer
    arrives with the embedding stage on day 3 and can be injected here; until
    then this keeps the chunker free of a 2 GB torch dependency. Subword
    tokenizers average close to 0.75 words per token on English prose, and
    financial text runs slightly denser because of numbers and tickers.
    """
    words = text.split()
    return int(len(words) / 0.75) + 1 if words else 0


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit, carrying the provenance to cite it."""

    chunk_id: str
    accession: str
    cik: str
    form: str
    filed: date
    chunk_type: ChunkType
    char_start: int
    char_end: int
    text: str
    token_count: int
    item_number: str | None = None
    item_title: str | None = None
    section: str = Section.UNKNOWN
    table_index: int | None = None
    parser_version: str = PARSER_VERSION

    @property
    def edgar_url(self) -> str:
        bare = self.cik.lstrip("0") or "0"
        return f"https://www.sec.gov/Archives/edgar/data/{bare}/{self.accession.replace('-', '')}/"


@dataclass(frozen=True, slots=True)
class ItemRegion:
    """A span of the document belonging to one Item."""

    item_number: str
    item_title: str
    char_start: int
    char_end: int
    section: Section = Section.UNKNOWN


@dataclass
class ChunkingStats:
    """Reported per filing so chunking quality is observable, not assumed."""

    items_found: int = 0
    chunks: int = 0
    table_chunks: int = 0
    prose_chunks: int = 0
    unassigned_chars: int = 0
    items: list[str] = field(default_factory=list)


class StructuralChunker:
    """Splits a parsed filing on its legal structure, then by token window."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tagger: SectionTagger | None = None,
        token_counter: Callable[[str], int] = approx_token_count,
    ) -> None:
        self.settings = settings or get_settings()
        self.tagger = tagger or SectionTagger()
        self.count_tokens = token_counter

    # --- structure --------------------------------------------------------
    def find_item_regions(self, parsed: ParsedFiling, form: str = "10-K") -> list[ItemRegion]:
        """Locate Item headings, scanning prose only so the TOC is excluded.

        Three filters, each earning its place against a real failure:

        * **prose only** - the table of contents is a table, so it isolates
          itself during parsing and needs no page-number heuristic;
        * **punctuation or a title** - excludes the running page headers that
          repeat an Item number at the top of every page;
        * **strictly increasing within a Part** - a document cannot re-enter a
          section it has left, but a new Part legitimately restarts numbering.
        """
        prose_spans = [(b.char_start, b.char_end) for b in parsed.prose_blocks]

        def in_prose(pos: int) -> bool:
            return any(s <= pos < e for s, e in prose_spans)

        parts = [m.start() for m in PART_HEADING.finditer(parsed.text) if in_prose(m.start())]

        def part_of(pos: int) -> int:
            return sum(1 for p in parts if p <= pos)

        found: list[tuple[int, str, str, int]] = []
        for m in ITEM_HEADING.finditer(parsed.text):
            if not in_prose(m.start()):
                continue
            title = (m.group(2) or m.group(3) or "").strip(" .:-—–")
            found.append((m.start(), m.group(1).upper(), title, part_of(m.start())))

        kept: list[tuple[int, str, str, int]] = []
        for entry in found:
            pos, item, _title, part = entry
            previous = [k for k in kept if k[3] == part]
            if previous and item_sort_key(item) <= item_sort_key(previous[-1][1]):
                continue
            kept.append(entry)

        regions: list[ItemRegion] = []
        for i, (pos, item, title, _part) in enumerate(kept):
            end_pos = kept[i + 1][0] if i + 1 < len(kept) else len(parsed.text)
            regions.append(
                ItemRegion(
                    item_number=item,
                    item_title=title,
                    char_start=pos,
                    char_end=end_pos,
                    section=self.tagger.tag(item, form),
                )
            )
        return regions

    # --- windowing --------------------------------------------------------
    def _split_units(self, text: str, offset: int) -> list[tuple[int, int]]:
        """Split a block into units no larger than the token budget.

        Splitting on blank lines alone is not enough. After whitespace
        normalisation many filings present an entire Item as one block with
        only single newlines in it, so a paragraph-only splitter finds one unit
        and emits a 17,000-token chunk - which is what the first version of this
        did.

        So the separators are tried in descending order of how natural a break
        they make, dropping to the next only for units still over budget:
        blank line, then single newline, then sentence end, then a hard word
        boundary. The hard split is a last resort but it must exist, or a single
        run-on paragraph silently defeats the window.
        """
        budget = self.settings.chunk_token_window
        separators = (
            re.compile(r"\n{2,}"),
            re.compile(r"\n"),
            re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])"),
        )

        def split(span_start: int, span_end: int, depth: int) -> list[tuple[int, int]]:
            body = text[span_start - offset : span_end - offset]
            if not body.strip():
                return []
            if self.count_tokens(body) <= budget:
                return [(span_start, span_end)]

            if depth < len(separators):
                pieces: list[tuple[int, int]] = []
                cursor = span_start
                for m in separators[depth].finditer(body):
                    cut = span_start + m.end()
                    if cut > cursor:
                        pieces.append((cursor, cut))
                    cursor = cut
                if cursor < span_end:
                    pieces.append((cursor, span_end))
                if len(pieces) > 1:
                    out: list[tuple[int, int]] = []
                    for a, b in pieces:
                        out.extend(split(a, b, depth + 1))
                    return out
                return split(span_start, span_end, depth + 1)

            # Last resort: fixed windows on word boundaries.
            return self._hard_split(span_start, span_end, text, offset, budget)

        return split(offset, offset + len(text), 0) or (
            [(offset, offset + len(text))] if text else []
        )

    @staticmethod
    def _hard_split(
        span_start: int, span_end: int, text: str, offset: int, budget: int
    ) -> list[tuple[int, int]]:
        body = text[span_start - offset : span_end - offset]
        # Derived from the same ratio approx_token_count uses, so the hard cut
        # lands near the budget rather than wildly over or under it.
        approx_chars = max(200, int(budget * 0.75 * 6))
        out: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(body):
            end = min(len(body), cursor + approx_chars)
            if end < len(body):
                space = body.rfind(" ", cursor + approx_chars // 2, end)
                if space > cursor:
                    end = space + 1
            out.append((span_start + cursor, span_start + end))
            cursor = end
        return out

    def _window(
        self, parsed: ParsedFiling, block_start: int, block_end: int
    ) -> list[tuple[int, int]]:
        """Contiguous, overlapping windows over one prose block."""
        text = parsed.text[block_start:block_end]
        budget = self.settings.chunk_token_window
        overlap = self.settings.chunk_token_overlap

        if self.count_tokens(text) <= budget:
            return [(block_start, block_end)]

        units = self._split_units(text, block_start)
        windows: list[tuple[int, int]] = []
        i = 0
        while i < len(units):
            start = units[i][0]
            tokens = 0
            j = i
            while j < len(units):
                unit_tokens = self.count_tokens(parsed.text[units[j][0] : units[j][1]])
                if tokens and tokens + unit_tokens > budget:
                    break
                tokens += unit_tokens
                j += 1
            end = units[min(j, len(units)) - 1][1]
            windows.append((start, end))

            if j >= len(units):
                break

            # Step back far enough to carry `overlap` tokens into the next
            # window, so a sentence split across a boundary is still retrievable
            # from at least one whole chunk.
            back = j
            carried = 0
            while back > i + 1 and carried < overlap:
                back -= 1
                carried += self.count_tokens(parsed.text[units[back][0] : units[back][1]])
            i = back
        return windows

    # --- assembly ---------------------------------------------------------
    def chunk(
        self,
        parsed: ParsedFiling,
        *,
        accession: str,
        cik: str,
        form: str,
        filed: date,
    ) -> tuple[list[Chunk], ChunkingStats]:
        regions = self.find_item_regions(parsed, form)
        stats = ChunkingStats(items_found=len(regions), items=[r.item_number for r in regions])
        chunks: list[Chunk] = []
        counters: dict[str, int] = {}

        def region_for(pos: int) -> ItemRegion | None:
            for r in regions:
                if r.char_start <= pos < r.char_end:
                    return r
            return None

        def add(
            start: int,
            end: int,
            kind: ChunkType,
            region: ItemRegion | None,
            table_index: int | None = None,
        ) -> None:
            body = parsed.text[start:end].strip()
            if not body:
                return
            tokens = self.count_tokens(body)
            if kind == "prose" and tokens < self.settings.chunk_min_tokens:
                return
            # Re-derive exact offsets after stripping, so the slice invariant
            # survives the strip.
            lead = len(parsed.text[start:end]) - len(parsed.text[start:end].lstrip())
            real_start = start + lead
            real_end = real_start + len(body)

            item = region.item_number if region else None
            key = f"item{item}" if item else "front"
            counters[key] = counters.get(key, 0) + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{accession}::{key}::c{counters[key]}",
                    accession=accession,
                    cik=cik,
                    form=form,
                    filed=filed,
                    chunk_type=kind,
                    char_start=real_start,
                    char_end=real_end,
                    text=body,
                    token_count=tokens,
                    item_number=item,
                    item_title=region.item_title if region else None,
                    section=str(region.section) if region else str(Section.UNKNOWN),
                    table_index=table_index,
                )
            )

        for block in parsed.blocks:
            region = region_for(block.char_start)
            if block.kind == "table":
                # Never split: a financial statement cut in half is worse than
                # one oversized chunk.
                add(block.char_start, block.char_end, "table", region, block.table_index)
                continue

            # Clip the block to its region so no window crosses an Item boundary.
            start = block.char_start
            end = block.char_end
            if region is not None:
                end = min(end, region.char_end)
            if end <= start:
                continue

            for w_start, w_end in self._window(parsed, start, end):
                add(w_start, w_end, "prose", region)

            # A block straddling a boundary continues in the next region.
            while region is not None and end < block.char_end:
                nxt = region_for(end)
                start, end = end, min(block.char_end, nxt.char_end if nxt else block.char_end)
                if end <= start:
                    break
                for w_start, w_end in self._window(parsed, start, end):
                    add(w_start, w_end, "prose", nxt)
                region = nxt

        stats.chunks = len(chunks)
        stats.table_chunks = sum(1 for c in chunks if c.chunk_type == "table")
        stats.prose_chunks = stats.chunks - stats.table_chunks
        log.info(
            "filing_chunked",
            accession=accession,
            items=stats.items_found,
            chunks=stats.chunks,
            tables=stats.table_chunks,
        )
        return chunks, stats

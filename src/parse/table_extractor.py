"""Turn an EDGAR HTML table into something a language model can read.

Financial statements are tables, and tables extracted as flat text are the
single biggest source of extraction error. The reason is visible the moment you
look at real markup: in Apple's FY2023 segment table, **59% of the cells are
empty spacers** used only for visual alignment, row widths vary from 10 to 30
cells because of colspans, and a currency symbol lives in its own cell, one
column to the left of the number it belongs to.

``get_text()`` on that produces:

    Americas $ 162,560 (4) % $ 169,658 11 %

which is numeric soup: no column headings, no association between a figure and
its year, and a dollar sign floating free of its amount.

The approach here is to rebuild an actual grid before serialising:

1. **Expand** ``colspan`` and ``rowspan`` into a dense matrix, so every row has
   the same width and cells sit in their true column.
2. **Drop empty columns**, not empty cells. Dropping cells is the obvious move
   and it is wrong - it shifts every value left by a different amount per row
   and destroys the alignment that makes a column mean one year.
3. **Merge orphaned symbols** - a column holding only ``$``, ``%`` or ``(`` is
   layout, so it is folded into the neighbour it qualifies.
4. **Serialise to markdown**, which models read reliably and which survives
   being embedded as text.

The result is chunked as one unit and never split mid-table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Cells holding only these are typographic, not data.
_SYMBOL_ONLY = re.compile(r"^[$€£¥%()\s.\-–—]*$")
_CURRENCY = re.compile(r"^[$€£¥]$")
_NUMERIC = re.compile(r"^\(?[\d,]+(\.\d+)?\)?$")


@dataclass(frozen=True, slots=True)
class ExtractedTable:
    """One table, serialised and described."""

    index: int
    markdown: str
    n_rows: int
    n_cols: int
    caption: str | None = None

    @property
    def is_empty(self) -> bool:
        return self.n_rows == 0 or self.n_cols == 0


def _cell_text(cell: Any) -> str:
    text = cell.get_text(" ", strip=True)
    # Non-breaking spaces are pervasive in EDGAR markup and are not whitespace
    # to str.strip(), so an "empty" spacer cell looks non-empty without this.
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _build_grid(table: Any) -> list[list[str]]:
    """Expand colspan and rowspan into a dense, rectangular matrix."""
    grid: list[list[str]] = []
    # (row_index, col_index) -> text, filled as spans are resolved.
    pending: dict[tuple[int, int], str] = {}

    rows = table.find_all("tr")
    for r, row in enumerate(rows):
        line: list[str] = []
        col = 0
        for cell in row.find_all(["td", "th"]):
            # Step past any column already occupied by a rowspan from above.
            while (r, col) in pending:
                line.append(pending.pop((r, col)))
                col += 1

            text = _cell_text(cell)
            try:
                colspan = max(1, int(cell.get("colspan", 1)))
                rowspan = max(1, int(cell.get("rowspan", 1)))
            except (TypeError, ValueError):
                colspan = rowspan = 1
            # A runaway span in malformed markup would otherwise allocate an
            # enormous matrix.
            colspan, rowspan = min(colspan, 64), min(rowspan, 64)

            for c in range(colspan):
                # Only the first cell of a span carries the text; the rest are
                # blank, which is what makes the empty-column test meaningful.
                line.append(text if c == 0 else "")
                for extra_row in range(1, rowspan):
                    pending[(r + extra_row, col + c)] = text if c == 0 else ""
                col += 1

        while (r, col) in pending:
            line.append(pending.pop((r, col)))
            col += 1
        grid.append(line)

    width = max((len(line) for line in grid), default=0)
    return [line + [""] * (width - len(line)) for line in grid]


def _drop_empty_columns(grid: list[list[str]]) -> list[list[str]]:
    """Remove columns that are empty in every row.

    Column-wise, never cell-wise: removing empty cells row by row shifts values
    by a different offset in each row, which is exactly how a table turns into
    numeric soup.
    """
    if not grid:
        return grid
    width = len(grid[0])
    keep = [c for c in range(width) if any(row[c] for row in grid)]
    return [[row[c] for c in keep] for row in grid]


def _merge_symbols_in_row(row: list[str]) -> list[str]:
    """Reattach orphaned typography to the figure it belongs to.

    EDGAR filers put the currency symbol in its own cell, and only on the first
    and total rows of a statement - so "Americas" carries a ``$`` cell that
    "Europe" does not. Left alone, that single extra cell shifts those rows one
    column right of every other row, which is what breaks the association
    between a figure and its year.

    Merging ``$`` rightwards into its amount and ``%`` leftwards into its
    number removes the discrepancy at source: every row then carries the same
    number of values, and compaction aligns them.
    """
    out: list[str] = []
    for cell in row:
        if _CURRENCY.match(cell):
            out.append(cell)  # provisional; resolved on the next non-empty cell
            continue
        if cell == "%" and out:
            for i in range(len(out) - 1, -1, -1):
                if out[i]:
                    out[i] = f"{out[i]}%"
                    break
            continue
        if out and _CURRENCY.match(out[-1]) and cell:
            out[-1] = f"{out[-1]}{cell}"
            continue
        out.append(cell)
    return out


def _compact_rows(grid: list[list[str]]) -> list[list[str]]:
    """Drop empty cells row by row, then re-pad to a common width.

    Cell-wise compaction is unsafe in general - it is the classic way to
    scramble a table. It is safe *here*, and only here, because it runs after
    symbol merging has removed the structural difference between rows, so every
    data row holds the same count of real values.

    A short row is padded on the left when it is the header, because a header
    omits the row-label column that the data rows carry.
    """
    compacted = [[c for c in _merge_symbols_in_row(row) if c.strip()] for row in grid]
    compacted = [row for row in compacted if row]
    if not compacted:
        return []

    width = max(len(row) for row in compacted)
    aligned: list[list[str]] = []
    for i, row in enumerate(compacted):
        missing = width - len(row)
        if missing and i == 0:
            aligned.append([""] * missing + row)
        else:
            aligned.append(row + [""] * missing)
    return aligned


def _drop_empty_rows(grid: list[list[str]]) -> list[list[str]]:
    return [row for row in grid if any(cell.strip() for cell in row)]


def _escape(cell: str) -> str:
    return cell.replace("|", r"\|")


def _to_markdown(grid: list[list[str]]) -> str:
    if not grid:
        return ""
    width = len(grid[0])
    header, body = grid[0], grid[1:]

    lines = ["| " + " | ".join(_escape(c) for c in header) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    lines.extend("| " + " | ".join(_escape(c) for c in row) + " |" for row in body)
    return "\n".join(lines)


class TableExtractor:
    """Serialises HTML tables to markdown, preserving column structure."""

    def __init__(self, *, max_rows: int = 400, max_cols: int = 40) -> None:
        self.max_rows = max_rows
        self.max_cols = max_cols

    def extract(self, table: Any, index: int = 0) -> ExtractedTable:
        grid = _build_grid(table)
        grid = _drop_empty_rows(grid)
        grid = _drop_empty_columns(grid)
        grid = _compact_rows(grid)

        if grid:
            grid = [row[: self.max_cols] for row in grid[: self.max_rows]]

        return ExtractedTable(
            index=index,
            markdown=_to_markdown(grid),
            n_rows=len(grid),
            n_cols=len(grid[0]) if grid else 0,
        )

    @staticmethod
    def looks_like_layout(table: Any) -> bool:
        """True for tables used purely to position text on the page.

        EDGAR filers routinely wrap a paragraph, a page header or a table of
        contents in a single-column or two-cell table. Serialising those as
        data tables would litter the chunk stream with one-cell "tables" and
        bury the real financial statements.
        """
        rows = table.find_all("tr")
        if len(rows) <= 1:
            return True
        cells = table.find_all(["td", "th"])
        if len(cells) <= 3:
            return True
        numeric = sum(1 for c in cells if _NUMERIC.match(_cell_text(c)))
        filled = sum(1 for c in cells if _cell_text(c))
        # A data table in a filing is mostly figures. A layout table is prose.
        return filled > 0 and numeric / filled < 0.10 and len(rows) < 4

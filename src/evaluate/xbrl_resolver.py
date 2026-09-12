"""Resolve the answer key: an XBRL fact for a business concept and fiscal year.

This is the half of the project that makes extraction accuracy measurable. If
it does not work, nothing downstream can be scored, which is why it is built
before the extractor rather than after it.

Three problems have to be handled, and all three are real rather than
theoretical:

**Tag variance.** ``us-gaap:Revenues``,
``us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax`` and
``us-gaap:SalesRevenueNet`` all mean revenue depending on filer and year. The
mapping lives in ``config/xbrl_tag_map.yaml``, ordered by preference.

**Period shape.** Revenue and cash flow are *duration* facts spanning a year;
total assets is an *instant* fact at a balance-sheet date. Accepting a quarterly
duration as an annual figure silently produces an answer key that is wrong by
roughly 4x, and the extraction it scores would then be marked wrong for being
right.

**Restatement.** The same fiscal year appears in ``companyfacts`` many times -
once as originally reported, then again as a comparative in each later filing,
sometimes restated. Scoring an extraction taken from a specific 10-K against a
figure restated two years later would penalise the model for reproducing what
the document actually said. So the resolver prefers the value *as originally
reported*, and accepts an ``accession`` to pin the fact to one filing exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from src.config.settings import get_settings


class UnresolvableError(LookupError):
    """No XBRL fact could be found for this field and fiscal year."""


@dataclass(frozen=True, slots=True)
class GroundTruthFact:
    """One resolved answer-key value, with the provenance to defend it."""

    field: str
    value: Decimal
    tag: str
    unit: str
    period_start: date | None
    period_end: date
    fiscal_year: int
    accession: str
    form: str
    filed: date
    restated: bool
    """True if a later filing reported a different value for the same period."""

    @property
    def qualified_tag(self) -> str:
        return f"us-gaap:{self.tag}"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    period: str  # "duration" | "instant"
    units: tuple[str, ...]
    tags: tuple[str, ...]


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


class XBRLResolver:
    """Turns a ``companyfacts`` payload into ground-truth values."""

    def __init__(
        self,
        fields: dict[str, FieldSpec],
        *,
        min_annual_days: int = 340,
        max_annual_days: int = 400,
        taxonomy: str = "us-gaap",
    ) -> None:
        self.fields = fields
        self.min_annual_days = min_annual_days
        self.max_annual_days = max_annual_days
        self.taxonomy = taxonomy

    @classmethod
    def from_config(cls, path: Path | None = None) -> XBRLResolver:
        path = path or get_settings().xbrl_tag_map_path
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        window = raw.get("annual_duration_days", {})
        fields = {
            name: FieldSpec(
                name=name,
                period=spec["period"],
                units=tuple(spec.get("units", ["USD"])),
                tags=tuple(spec["tags"]),
            )
            for name, spec in raw["fields"].items()
        }
        return cls(
            fields,
            min_annual_days=int(window.get("min", 340)),
            max_annual_days=int(window.get("max", 400)),
        )

    # --- fact selection ---------------------------------------------------
    def _is_annual(self, start: date | None, end: date) -> bool:
        if start is None:
            return False
        span = (end - start).days
        return self.min_annual_days <= span <= self.max_annual_days

    def _candidates(
        self, facts: dict[str, Any], spec: FieldSpec, tag: str, fiscal_year: int
    ) -> list[dict[str, Any]]:
        """Every reported observation of one tag that belongs to this fiscal year."""
        entry = facts.get("facts", {}).get(self.taxonomy, {}).get(tag)
        if not entry:
            return []

        out: list[dict[str, Any]] = []
        for unit in spec.units:
            for obs in entry.get("units", {}).get(unit, []):
                end = _parse_date(obs.get("end"))
                if end is None or end.year != fiscal_year:
                    # A fiscal year is identified by the calendar year its period
                    # ENDS in, not by the `fy` field. `fy` describes the filing
                    # the fact appeared in, so a FY2022 comparative sitting in a
                    # FY2024 10-K carries fy=2024 and would be misfiled by it.
                    continue

                start = _parse_date(obs.get("start"))
                if spec.period == "duration" and not self._is_annual(start, end):
                    continue
                if spec.period == "instant" and start is not None:
                    continue

                out.append({**obs, "_unit": unit, "_start": start, "_end": end})
        return out

    def resolve_field(
        self,
        facts: dict[str, Any],
        field: str,
        fiscal_year: int,
        *,
        accession: str | None = None,
    ) -> GroundTruthFact:
        """Resolve one field. Raises :class:`UnresolvableError` if no tag matched.

        ``accession`` pins the answer to the figure as reported in that specific
        filing, which is the correct comparison when scoring an extraction taken
        from it.
        """
        spec = self.fields.get(field)
        if spec is None:
            raise KeyError(f"no tag mapping configured for field {field!r}")

        for tag in spec.tags:  # order encodes preference; first match wins
            candidates = self._candidates(facts, spec, tag, fiscal_year)
            if not candidates:
                continue

            if accession is not None:
                pinned = [c for c in candidates if c.get("accn") == accession]
                if pinned:
                    candidates = pinned

            annual_reports = [c for c in candidates if c.get("form") == "10-K"]
            pool = annual_reports or candidates

            # Earliest filing = as originally reported. The extraction being
            # scored was read out of the document, so the document's own number
            # is the fair comparison.
            pool.sort(key=lambda c: (c.get("filed") or "9999-99-99", c.get("accn") or ""))
            chosen = pool[0]

            distinct_values = {str(c["val"]) for c in pool}
            return GroundTruthFact(
                field=field,
                value=Decimal(str(chosen["val"])),
                tag=tag,
                unit=chosen["_unit"],
                period_start=chosen["_start"],
                period_end=chosen["_end"],
                fiscal_year=fiscal_year,
                accession=str(chosen.get("accn", "")),
                form=str(chosen.get("form", "")),
                filed=_parse_date(chosen.get("filed")) or chosen["_end"],
                restated=len(distinct_values) > 1,
            )

        raise UnresolvableError(
            f"no us-gaap tag among {spec.tags} produced an annual {field} "
            f"for fiscal year {fiscal_year}"
        )

    def resolve(
        self,
        facts: dict[str, Any],
        fiscal_year: int,
        fields: list[str] | None = None,
        *,
        accession: str | None = None,
    ) -> dict[str, GroundTruthFact | None]:
        """Resolve every configured field. ``None`` means unresolvable.

        Unresolvable is returned rather than raised because the count of
        unresolvable filings is itself a reported result - it bounds the
        denominator of every accuracy figure in the README.
        """
        wanted = fields if fields is not None else list(self.fields)
        resolved: dict[str, GroundTruthFact | None] = {}
        for field in wanted:
            try:
                resolved[field] = self.resolve_field(facts, field, fiscal_year, accession=accession)
            except UnresolvableError:
                resolved[field] = None
        return resolved

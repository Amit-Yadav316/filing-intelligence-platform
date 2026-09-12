"""The resolver produces the answer key, so its failure modes are silent by
nature: a wrong ground-truth value does not crash anything, it just marks a
correct extraction as wrong. These tests pin the selection rules.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from src.evaluate.xbrl_resolver import (
    FieldSpec,
    UnresolvableError,
    XBRLResolver,
)


def obs(
    val: float,
    *,
    start: str | None = None,
    end: str,
    form: str = "10-K",
    filed: str = "2024-02-01",
    accn: str = "0000000000-24-000001",
    fy: int = 2024,
) -> dict[str, Any]:
    """One observation as it appears in a companyfacts units array."""
    entry = {"val": val, "end": end, "form": form, "filed": filed, "accn": accn, "fy": fy}
    if start is not None:
        entry["start"] = start
    return entry


def facts(tag_units: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "cik": 1,
        "entityName": "Test Co",
        "facts": {"us-gaap": {tag: {"units": {"USD": units}} for tag, units in tag_units.items()}},
    }


@pytest.fixture
def resolver() -> XBRLResolver:
    return XBRLResolver(
        {
            "total_revenue": FieldSpec(
                name="total_revenue",
                period="duration",
                units=("USD",),
                tags=("Revenues", "SalesRevenueNet"),
            ),
            "total_assets": FieldSpec(
                name="total_assets", period="instant", units=("USD",), tags=("Assets",)
            ),
        }
    )


# --- period shape ---------------------------------------------------------
def test_resolves_an_annual_duration_fact(resolver: XBRLResolver) -> None:
    f = facts({"Revenues": [obs(1000, start="2023-01-01", end="2023-12-31")]})

    got = resolver.resolve_field(f, "total_revenue", 2023)

    assert got.value == Decimal("1000")
    assert got.tag == "Revenues"
    assert got.period_start == date(2023, 1, 1)
    assert got.period_end == date(2023, 12, 31)


def test_rejects_a_quarterly_duration_as_an_annual_figure(resolver: XBRLResolver) -> None:
    """The 4x bug. A Q4 revenue accepted as FY revenue would mark every correct
    extraction wrong, and nothing would crash to reveal it."""
    f = facts({"Revenues": [obs(250, start="2023-10-01", end="2023-12-31")]})

    with pytest.raises(UnresolvableError):
        resolver.resolve_field(f, "total_revenue", 2023)


def test_accepts_a_52_53_week_fiscal_year(resolver: XBRLResolver) -> None:
    """Retailers report 52/53-week years that do not align to the calendar.
    52 weeks is 364 days, 53 weeks is 371 - both inside the accepted window."""
    f = facts({"Revenues": [obs(900, start="2022-12-31", end="2023-12-30")]})

    assert resolver.resolve_field(f, "total_revenue", 2023).value == Decimal("900")


def test_instant_field_rejects_a_duration_observation(resolver: XBRLResolver) -> None:
    f = facts({"Assets": [obs(500, start="2023-01-01", end="2023-12-31")]})

    with pytest.raises(UnresolvableError):
        resolver.resolve_field(f, "total_assets", 2023)


def test_instant_field_resolves_a_point_in_time_observation(resolver: XBRLResolver) -> None:
    f = facts({"Assets": [obs(500, end="2023-12-31")]})

    got = resolver.resolve_field(f, "total_assets", 2023)

    assert got.value == Decimal("500")
    assert got.period_start is None


# --- fiscal year identification -------------------------------------------
def test_fiscal_year_comes_from_the_period_end_not_the_fy_field(
    resolver: XBRLResolver,
) -> None:
    """A FY2022 comparative restated inside a FY2024 10-K carries fy=2024.
    Keying on `fy` would file it under the wrong year."""
    f = facts(
        {
            "Revenues": [
                obs(100, start="2022-01-01", end="2022-12-31", fy=2024, filed="2024-02-01"),
                obs(300, start="2023-01-01", end="2023-12-31", fy=2024, filed="2024-02-01"),
            ]
        }
    )

    assert resolver.resolve_field(f, "total_revenue", 2022).value == Decimal("100")
    assert resolver.resolve_field(f, "total_revenue", 2023).value == Decimal("300")


def test_non_calendar_fiscal_year_keys_on_the_year_it_ends_in(
    resolver: XBRLResolver,
) -> None:
    """Apple's FY2023 runs Oct 2022 to Sep 2023 and is FY2023, not FY2022."""
    f = facts({"Revenues": [obs(383, start="2022-09-25", end="2023-09-30")]})

    assert resolver.resolve_field(f, "total_revenue", 2023).value == Decimal("383")
    with pytest.raises(UnresolvableError):
        resolver.resolve_field(f, "total_revenue", 2022)


# --- tag preference -------------------------------------------------------
def test_first_configured_tag_wins(resolver: XBRLResolver) -> None:
    """Order in the tag map encodes preference, not merely membership."""
    f = facts(
        {
            "SalesRevenueNet": [obs(111, start="2023-01-01", end="2023-12-31")],
            "Revenues": [obs(999, start="2023-01-01", end="2023-12-31")],
        }
    )

    got = resolver.resolve_field(f, "total_revenue", 2023)

    assert (got.tag, got.value) == ("Revenues", Decimal("999"))


def test_falls_through_to_a_later_tag_when_the_preferred_one_is_absent(
    resolver: XBRLResolver,
) -> None:
    f = facts({"SalesRevenueNet": [obs(111, start="2023-01-01", end="2023-12-31")]})

    assert resolver.resolve_field(f, "total_revenue", 2023).tag == "SalesRevenueNet"


def test_a_tag_present_but_empty_for_the_year_falls_through(resolver: XBRLResolver) -> None:
    """Revenues exists but only for 2022, so 2023 must fall through to the next tag."""
    f = facts(
        {
            "Revenues": [obs(500, start="2022-01-01", end="2022-12-31")],
            "SalesRevenueNet": [obs(600, start="2023-01-01", end="2023-12-31")],
        }
    )

    assert resolver.resolve_field(f, "total_revenue", 2023).tag == "SalesRevenueNet"


def test_unknown_field_is_a_configuration_error_not_a_lookup_failure(
    resolver: XBRLResolver,
) -> None:
    with pytest.raises(KeyError, match="no tag mapping configured"):
        resolver.resolve_field(facts({}), "ebitda", 2023)


# --- restatement ----------------------------------------------------------
def test_prefers_the_value_as_originally_reported(resolver: XBRLResolver) -> None:
    """The extraction being scored was read out of the original document, so
    the original figure is the fair comparison - not a restatement published
    two years later."""
    f = facts(
        {
            "Revenues": [
                obs(1000, start="2023-01-01", end="2023-12-31", filed="2024-02-01"),
                obs(1050, start="2023-01-01", end="2023-12-31", filed="2026-02-01"),
            ]
        }
    )

    got = resolver.resolve_field(f, "total_revenue", 2023)

    assert got.value == Decimal("1000")
    assert got.filed == date(2024, 2, 1)
    assert got.restated is True, "a differing later value must be flagged"


def test_not_flagged_as_restated_when_every_report_agrees(resolver: XBRLResolver) -> None:
    f = facts(
        {
            "Revenues": [
                obs(1000, start="2023-01-01", end="2023-12-31", filed="2024-02-01"),
                obs(1000, start="2023-01-01", end="2023-12-31", filed="2026-02-01"),
            ]
        }
    )

    assert resolver.resolve_field(f, "total_revenue", 2023).restated is False


def test_prefers_the_annual_report_over_a_quarterly_one(resolver: XBRLResolver) -> None:
    f = facts(
        {
            "Revenues": [
                obs(1000, start="2023-01-01", end="2023-12-31", form="10-Q", filed="2024-01-05"),
                obs(1000, start="2023-01-01", end="2023-12-31", form="10-K", filed="2024-02-01"),
            ]
        }
    )

    assert resolver.resolve_field(f, "total_revenue", 2023).form == "10-K"


def test_accession_pins_the_answer_to_one_filing(resolver: XBRLResolver) -> None:
    """Scoring an extraction from a specific filing should compare against the
    number that filing actually printed."""
    f = facts(
        {
            "Revenues": [
                obs(1000, start="2023-01-01", end="2023-12-31", filed="2024-02-01", accn="A"),
                obs(1050, start="2023-01-01", end="2023-12-31", filed="2026-02-01", accn="B"),
            ]
        }
    )

    assert resolver.resolve_field(f, "total_revenue", 2023, accession="B").value == Decimal("1050")


# --- bulk interface -------------------------------------------------------
def test_resolve_returns_none_for_unresolvable_fields(resolver: XBRLResolver) -> None:
    """Unresolvable is a reported result, not an exception: it bounds the
    denominator of every accuracy figure."""
    f = facts({"Revenues": [obs(1000, start="2023-01-01", end="2023-12-31")]})

    got = resolver.resolve(f, 2023)

    assert got["total_revenue"] is not None
    assert got["total_assets"] is None


def test_empty_companyfacts_resolves_nothing_without_crashing(resolver: XBRLResolver) -> None:
    assert resolver.resolve({}, 2023) == {"total_revenue": None, "total_assets": None}


# --- configuration --------------------------------------------------------
def test_loads_the_shipped_tag_map() -> None:
    """The committed config must stay loadable and cover the four scored fields."""
    loaded = XBRLResolver.from_config()

    assert set(loaded.fields) == {
        "total_revenue",
        "net_income",
        "total_assets",
        "operating_cash_flow",
    }
    assert loaded.fields["total_assets"].period == "instant"
    assert loaded.fields["total_revenue"].period == "duration"
    # Revenue needs several tags; one alone resolves about half the corpus.
    assert len(loaded.fields["total_revenue"].tags) >= 4
    assert loaded.min_annual_days < 365 < loaded.max_annual_days

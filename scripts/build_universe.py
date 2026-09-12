"""Build ``config/universe.json``: the corpus of companies to ingest.

The ticker list is hand-curated rather than screened, for two reasons. EDGAR
publishes no market capitalisation and no GICS sector, so an automated
large-cap screen would need a second data source the project does not
otherwise require; and a fixed list makes the corpus reproducible, which the
accuracy denominator depends on.

What is *not* hand-asserted is the industry classification. Each company's SIC
code and description are read from EDGAR's own submissions endpoint, so the
sector breakdown in the README is sourced, not claimed. The GICS-style
``sector`` field below is the curation axis; ``sic_description`` is EDGAR's.

Usage:  python -m scripts.build_universe
"""

from __future__ import annotations

import json
from typing import Any

from src.config.settings import get_settings
from src.ingest.edgar_client import EdgarClient, EdgarError, pad_cik
from src.observability.logging import configure_logging, get_logger

log = get_logger(__name__)

# 55 US large caps across 11 sectors. Breadth matters more than count: the
# interesting question is whether XBRL tag usage varies by industry, and a
# corpus of 55 technology companies could not answer it.
UNIVERSE: dict[str, list[str]] = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "VZ", "T"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
    "Consumer Staples": ["WMT", "PG", "KO", "PEP", "COST"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP"],
    "Health Care": ["JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Industrials": ["BA", "CAT", "GE", "UPS", "HON", "LMT"],
    "Materials": ["LIN", "SHW"],
    "Utilities": ["NEE", "DUK"],
    "Real Estate": ["AMT", "PLD"],
}

# Ticker -> CIK is not stable across corporate reorganisations, and EDGAR's
# ticker map always points at the *current* registrant. When a company
# reorganises, the ticker migrates to a newly-registered entity whose XBRL
# history begins at the reorganisation, while the filings for the target period
# stay under the predecessor CIK. A naive lookup then silently resolves to an
# entity with no data for the years being studied.
#
# This was not hypothetical: the resolution probe initially failed all four
# fields for XOM, because the ticker now maps to "ExxonMobil Holdings Corp"
# (CIK 2115436), whose facts start in 2025. The 2022-2024 filings belong to
# EXXON MOBIL CORP (CIK 34088), which no longer carries the ticker at all.
#
# The coverage check below catches this class of problem generally; these
# overrides record the answer once it has been checked by hand.
# Only one CIK per company is carried, to keep the pipeline simple. Where a
# reorganisation splits the corpus window across two entities, the CIK covering
# the most of it is chosen and the shortfall is reported honestly by the
# resolution probe rather than hidden.
CIK_OVERRIDES: dict[str, str] = {
    # Ticker now points at "ExxonMobil Holdings Corp" (CIK 2115436), whose XBRL
    # facts begin in 2025. All 2022-2024 filings are under the predecessor.
    "XOM": "0000034088",  # EXXON MOBIL CORP
    # Ticker now points at the post-reorganisation "BlackRock, Inc."
    # (CIK 2012383), first filing 2024-02-20. FY2022 and FY2023 belong to the
    # predecessor, renamed BlackRock Finance, Inc. FY2024 straddles the two and
    # is expected to show as partially unresolvable.
    "BLK": "0001364742",  # BlackRock Finance, Inc.
}


def _history_span(sub: dict[str, Any]) -> tuple[str | None, str | None]:
    """The full date range of an entity's EDGAR filing history.

    Counting target forms inside ``filings.recent`` is the obvious check and it
    is wrong. ``recent`` is capped at roughly the latest 1000 filings, and a
    large bank issuing structured notes files tens of thousands of 424B2s a
    year - JPMorgan's ``recent`` block holds 26,065 filings spanning a single
    twelve-month window, with its 10-Ks paged out into ``filings.files``.
    A form-count check therefore reports "no 10-K in 2022-2024" for exactly the
    companies that file the most.

    The date ranges of those older pages are already present in the payload, so
    the span of an entity's history can be established without downloading any
    of them. That is the signal we actually want: a predecessor CIK whose
    history begins after the corpus window is the failure mode worth catching.
    """
    dates: list[str] = []
    filings = sub.get("filings", {})
    recent = filings.get("filingDate", []) or filings.get("recent", {}).get("filingDate", [])
    if recent:
        dates += [min(recent), max(recent)]
    for page in filings.get("files", []):
        for key in ("filingFrom", "filingTo"):
            if page.get(key):
                dates.append(str(page[key]))
    if not dates:
        return None, None
    return min(dates), max(dates)


def build() -> dict[str, Any]:
    settings = get_settings()
    wanted = {t: sector for sector, tickers in UNIVERSE.items() for t in tickers}

    with EdgarClient(settings) as client:
        log.info("fetching_ticker_map", url=settings.edgar_company_tickers_url)
        raw = client.get_company_tickers()

        # company_tickers.json is a dict keyed by row index, not by ticker.
        by_ticker = {str(row["ticker"]).upper(): row for row in raw.values() if row.get("ticker")}

        companies: list[dict[str, Any]] = []
        missing: list[str] = []
        no_coverage: list[str] = []
        target_years = set(settings.fiscal_years)

        for ticker, sector in sorted(wanted.items()):
            row = by_ticker.get(ticker)
            if row is None:
                missing.append(ticker)
                log.warning("ticker_not_in_edgar_map", ticker=ticker)
                continue

            cik = CIK_OVERRIDES.get(ticker) or pad_cik(row["cik_str"])
            entry: dict[str, Any] = {
                "ticker": ticker,
                "cik": cik,
                "name": row["title"],
                "sector": sector,
                "sic": None,
                "sic_description": None,
                "exchange": None,
                "fiscal_year_end": None,
            }

            # Enrich from EDGAR itself so the classification is sourced.
            try:
                sub = client.get_submissions(cik)
            except EdgarError as exc:
                log.warning("submissions_unavailable", ticker=ticker, cik=cik, error=str(exc))
            else:
                entry["sic"] = sub.get("sic") or None
                entry["sic_description"] = sub.get("sicDescription") or None
                exchanges = sub.get("exchanges") or []
                entry["exchange"] = exchanges[0] if exchanges else None
                entry["fiscal_year_end"] = sub.get("fiscalYearEnd") or None

                first_filing, last_filing = _history_span(sub)
                entry["history_from"] = first_filing
                entry["history_to"] = last_filing
                window_end = f"{max(target_years)}-12-31"
                if first_filing is not None and first_filing > window_end:
                    no_coverage.append(
                        f"{ticker} (CIK {cik}, {sub.get('name')}, history starts {first_filing})"
                    )
                    log.warning(
                        "entity_history_postdates_corpus_window",
                        ticker=ticker,
                        cik=cik,
                        entity=sub.get("name"),
                        history_from=first_filing,
                        window_end=window_end,
                    )

            companies.append(entry)

    return {
        "description": (
            "Corpus universe for the Filing Intelligence Platform. "
            "sector is a hand-assigned GICS-style label; sic and sic_description "
            "come from EDGAR submissions."
        ),
        "forms": list(get_settings().target_forms),
        "fiscal_years": list(get_settings().fiscal_years),
        "company_count": len(companies),
        "missing_tickers": missing,
        "no_filing_coverage": no_coverage,
        "companies": companies,
    }


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)

    universe = build()
    settings.universe_path.parent.mkdir(parents=True, exist_ok=True)
    settings.universe_path.write_text(json.dumps(universe, indent=2) + "\n", encoding="utf-8")

    sectors: dict[str, int] = {}
    for c in universe["companies"]:
        sectors[c["sector"]] = sectors.get(c["sector"], 0) + 1

    print(f"\nWrote {settings.universe_path}")
    print(f"  companies : {universe['company_count']}")
    print(f"  sectors   : {len(sectors)}")
    resolved_sic = sum(1 for c in universe["companies"] if c["sic"])
    print(f"  SIC resolved from EDGAR: {resolved_sic}/{universe['company_count']}")
    if universe["missing_tickers"]:
        print(f"  NOT FOUND in EDGAR ticker map: {universe['missing_tickers']}")
    if universe["no_filing_coverage"]:
        print("\n  WARNING - no 10-K/10-Q filings in the corpus years:")
        for item in universe["no_filing_coverage"]:
            print(f"    {item}")
        print("  Add a CIK_OVERRIDES entry for each, then re-run.")


if __name__ == "__main__":
    main()

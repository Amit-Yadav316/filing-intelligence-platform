"""EdgarClient tests run against recorded fixtures and mocked transport.

Not the live API: a test suite that depends on sec.gov being up, and that adds
load to a public service on every CI run, is the wrong kind of test. The one
live check is marked ``network`` and deselected by default.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import httpx
import pytest
import respx

from src.config.settings import Settings
from src.ingest.circuit_breaker import CircuitOpenError
from src.ingest.edgar_client import EdgarClient, EdgarError, FilingRef, bare_cik, pad_cik

INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/2024/QTR1/master.20240102.idx"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"


@pytest.fixture
def client(settings: Settings, no_sleep: Callable[[float], None]) -> EdgarClient:
    return EdgarClient(settings, sleep=no_sleep)


# --- CIK normalisation ----------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "padded", "bare"),
    [
        (320193, "0000320193", "320193"),
        ("320193", "0000320193", "320193"),
        ("0000320193", "0000320193", "320193"),
        ("CIK0000320193", "0000320193", "320193"),
        (1288750, "0001288750", "1288750"),
    ],
)
def test_cik_normalisation(raw: str | int, padded: str, bare: str) -> None:
    """data.sec.gov wants 10 digits zero-padded; Archives paths want neither."""
    assert pad_cik(raw) == padded
    assert bare_cik(raw) == bare


def test_filing_ref_derives_accession() -> None:
    ref = FilingRef(
        cik="320193",
        company="Apple Inc.",
        form="10-K",
        filed=date(2023, 11, 3),
        path="edgar/data/320193/0000320193-23-000106.txt",
    )
    assert ref.accession == "0000320193-23-000106"
    assert ref.accession_nodash == "000032019323000106"


# --- Responsible consumption ----------------------------------------------
@respx.mock
def test_sends_contact_user_agent_on_every_request(client: EdgarClient) -> None:
    """The SEC returns 403 without this. It is the reason the setting is required."""
    route = respx.get(FACTS_URL).mock(return_value=httpx.Response(200, json={"cik": 320193}))

    client.get_company_facts(320193)

    assert route.call_count == 1
    assert route.calls[0].request.headers["User-Agent"] == "Test Harness test@example.org"


@respx.mock
def test_rate_limiter_is_consulted_before_each_request(settings: Settings) -> None:
    calls: list[float] = []

    class SpyLimiter:
        rate_per_sec = 10.0

        def acquire(self, tokens: float = 1.0) -> float:
            calls.append(tokens)
            return 0.0

    respx.get(FACTS_URL).mock(return_value=httpx.Response(200, json={}))
    EdgarClient(settings, rate_limiter=SpyLimiter()).get_company_facts(320193)  # type: ignore[arg-type]

    assert calls == [1.0]


# --- Retry policy ---------------------------------------------------------
@respx.mock
def test_retries_5xx_then_succeeds(client: EdgarClient) -> None:
    route = respx.get(FACTS_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json={"cik": 320193}),
        ]
    )

    assert client.get_company_facts(320193) == {"cik": 320193}
    assert route.call_count == 3


@respx.mock
def test_retries_transport_errors(client: EdgarClient) -> None:
    route = respx.get(FACTS_URL).mock(
        side_effect=[httpx.ConnectTimeout("timed out"), httpx.Response(200, json={"ok": True})]
    )

    assert client.get_company_facts(320193) == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_honours_retry_after_on_429(settings: Settings) -> None:
    """EDGAR throttling is answered by waiting the time it asked for, not by
    a backoff we invented."""
    slept: list[float] = []
    respx.get(FACTS_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json={}),
        ]
    )

    EdgarClient(settings, sleep=slept.append).get_company_facts(320193)

    assert slept == [7.0]


@respx.mock
def test_malformed_retry_after_falls_back_to_backoff(settings: Settings) -> None:
    slept: list[float] = []
    respx.get(FACTS_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
            httpx.Response(200, json={}),
        ]
    )

    EdgarClient(settings, sleep=slept.append).get_company_facts(320193)

    assert len(slept) == 1
    assert 0.0 <= slept[0] <= settings.edgar_backoff_max_seconds


@respx.mock
def test_does_not_retry_client_errors(client: EdgarClient) -> None:
    """A 404 is an answer, not an outage. Retrying it wastes the rate budget."""
    route = respx.get(FACTS_URL).mock(return_value=httpx.Response(404))

    with pytest.raises(EdgarError) as err:
        client.get_company_facts(320193)

    assert err.value.status == 404
    assert route.call_count == 1


@respx.mock
def test_gives_up_after_max_retries(client: EdgarClient, settings: Settings) -> None:
    route = respx.get(FACTS_URL).mock(return_value=httpx.Response(503))

    with pytest.raises(EdgarError, match="failed after 3 attempts"):
        client.get_company_facts(320193)

    assert route.call_count == settings.edgar_max_retries + 1


# --- Circuit breaker ------------------------------------------------------
@respx.mock
def test_circuit_opens_after_consecutive_failures(settings: Settings) -> None:
    """Threshold is 3 and max_retries is 2, so the first call burns all three
    attempts and trips the breaker; the second call must not reach the network."""
    route = respx.get(FACTS_URL).mock(return_value=httpx.Response(503))
    client = EdgarClient(settings, sleep=lambda _s: None)

    with pytest.raises(EdgarError):
        client.get_company_facts(320193)
    calls_before = route.call_count

    with pytest.raises(CircuitOpenError):
        client.get_company_facts(320193)

    assert route.call_count == calls_before, "breaker let a request through while open"


# --- Daily index ----------------------------------------------------------
@respx.mock
def test_parses_daily_index(client: EdgarClient, master_idx: str) -> None:
    respx.get(INDEX_URL).mock(return_value=httpx.Response(200, text=master_idx))

    rows = client.get_daily_index(date(2024, 1, 2))

    assert len(rows) == 7, "header and preamble lines must not survive parsing"
    assert all(isinstance(r, FilingRef) for r in rows)
    apple = next(r for r in rows if r.cik == "320193")
    assert (apple.company, apple.form, apple.filed) == ("Apple Inc.", "10-Q", date(2024, 1, 2))
    assert apple.accession == "0000320193-24-000005"


@respx.mock
def test_daily_index_filters_by_form(client: EdgarClient, master_idx: str) -> None:
    respx.get(INDEX_URL).mock(return_value=httpx.Response(200, text=master_idx))

    rows = client.get_daily_index(date(2024, 1, 2), forms=("10-K", "10-Q"))

    assert {r.form for r in rows} == {"10-K", "10-Q"}
    assert len(rows) == 4


@respx.mock
def test_daily_index_on_a_non_publishing_day_is_empty_not_an_error(client: EdgarClient) -> None:
    """EDGAR publishes no index at weekends. That is normal, so a backfill over
    a date range must not fail on a Saturday."""
    respx.get(INDEX_URL).mock(return_value=httpx.Response(404))

    assert client.get_daily_index(date(2024, 1, 2)) == []


@respx.mock
def test_daily_index_quarter_is_derived_from_the_month(client: EdgarClient) -> None:
    url = "https://www.sec.gov/Archives/edgar/daily-index/2023/QTR4/master.20231103.idx"
    route = respx.get(url).mock(return_value=httpx.Response(200, text=""))

    client.get_daily_index(date(2023, 11, 3))

    assert route.call_count == 1


# --- URL construction -----------------------------------------------------
@respx.mock
def test_builds_archive_document_url_without_padding(client: EdgarClient) -> None:
    url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    respx.get(url).mock(return_value=httpx.Response(200, content=b"<html/>"))

    body = client.fetch_filing_document("0000320193", "0000320193-23-000106", "aapl-20230930.htm")

    assert body == b"<html/>"


@respx.mock
def test_submissions_url_uses_padded_cik(client: EdgarClient) -> None:
    route = respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json={"cik": "320193"})
    )

    client.get_submissions(320193)

    assert route.call_count == 1


# --- One live check, deselected by default --------------------------------
@pytest.mark.network
def test_live_companyfacts_smoke() -> None:
    """Proves the contract against the real service. Run with -m network."""
    with EdgarClient() as live:
        facts = live.get_company_facts(320193)
    assert facts["entityName"].startswith("Apple")
    assert "Revenues" in facts["facts"]["us-gaap"] or "Assets" in facts["facts"]["us-gaap"]

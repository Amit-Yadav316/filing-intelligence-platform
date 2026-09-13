"""A rate-limited, retrying, circuit-broken client for SEC EDGAR.

Every outbound SEC call in this project goes through this class. Nothing else
is permitted to construct a URL against sec.gov.

Responsible-consumption contract
--------------------------------
The SEC asks automated consumers to (a) identify themselves with a descriptive
``User-Agent`` carrying a reachable contact address and (b) stay under 10
requests per second. Both are enforced here rather than left to the caller:
the ``User-Agent`` is a required, validated setting, and a single shared
:class:`RateLimiter` caps the whole process below the published ceiling.

Failure handling
----------------
* Transport errors and 429/5xx are retried with exponential backoff plus full
  jitter, honouring ``Retry-After`` when the server sends one.
* Other 4xx are *not* retried - a 404 is an answer, not an outage.
* Consecutive server-side failures trip a :class:`CircuitBreaker`, converting a
  degraded dependency into a fast bounded failure instead of a retry storm that
  would earn an IP block.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from types import TracebackType
from typing import Any

import httpx

from src.config.settings import Settings, get_settings
from src.ingest.circuit_breaker import STATE_CODE, CircuitBreaker
from src.ingest.rate_limiter import RateLimiter
from src.observability.logging import get_logger
from src.observability.metrics import (
    EDGAR_CIRCUIT_STATE,
    EDGAR_RATE_LIMIT_WAIT,
    EDGAR_REQUEST_LATENCY,
    EDGAR_REQUESTS,
)

log = get_logger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class EdgarError(RuntimeError):
    """A request to EDGAR failed permanently, after retries."""

    def __init__(self, message: str, *, url: str, status: int | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.status = status


@dataclass(frozen=True, slots=True)
class FilingRef:
    """One row of an EDGAR daily index."""

    cik: str
    company: str
    form: str
    filed: date
    path: str

    @property
    def accession(self) -> str:
        """Dashed accession number, e.g. 0000320193-23-000106."""
        return self.path.rsplit("/", 1)[-1].removesuffix(".txt")

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")


def pad_cik(cik: str | int) -> str:
    """EDGAR's JSON APIs key on a zero-padded 10-digit CIK; Archives does not."""
    digits = str(cik).upper().removeprefix("CIK").lstrip("0")
    return (digits or "0").zfill(10)


def bare_cik(cik: str | int) -> str:
    """The Archives path form: no padding, no prefix."""
    digits = str(cik).upper().removeprefix("CIK").lstrip("0")
    return digits or "0"


class EdgarClient:
    """The single doorway to sec.gov."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        rate_limiter: RateLimiter | None = None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={
                "User-Agent": self.settings.edgar_user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=httpx.Timeout(self.settings.edgar_timeout_seconds),
            follow_redirects=True,
        )
        self._limiter = rate_limiter or RateLimiter(self.settings.edgar_rate_limit_per_sec)
        self._breaker = breaker or CircuitBreaker(
            self.settings.edgar_breaker_fail_threshold,
            self.settings.edgar_breaker_reset_seconds,
            name="edgar",
        )

    # --- lifecycle --------------------------------------------------------
    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # --- transport --------------------------------------------------------
    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.settings.edgar_retry_after_max_seconds)
        ceiling = min(
            self.settings.edgar_backoff_base_seconds * (2**attempt),
            self.settings.edgar_backoff_max_seconds,
        )
        # Full jitter. Fixed backoff synchronises concurrent workers into a
        # thundering herd at exactly the moment the dependency recovers.
        return random.uniform(0, ceiling)

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None  # HTTP-date form; fall back to computed backoff

    def _get(self, url: str, *, endpoint: str) -> httpx.Response:
        """Fetch url, retrying transient failures. Raises EdgarError on give-up."""
        last_error = "no attempt made"
        last_status: int | None = None
        retry_after: float | None = None

        for attempt in range(self.settings.edgar_max_retries + 1):
            self._breaker.before_call()
            EDGAR_CIRCUIT_STATE.set(STATE_CODE[self._breaker.state])

            waited = self._limiter.acquire()
            EDGAR_RATE_LIMIT_WAIT.observe(waited)

            started = time.perf_counter()
            try:
                response = self._client.get(url)
            except httpx.HTTPError as exc:
                self._breaker.record_failure()
                EDGAR_REQUESTS.labels(endpoint=endpoint, status="transport_error").inc()
                last_error, last_status, retry_after = f"transport error: {exc}", None, None
            else:
                EDGAR_REQUEST_LATENCY.labels(endpoint=endpoint).observe(
                    time.perf_counter() - started
                )
                EDGAR_REQUESTS.labels(endpoint=endpoint, status=str(response.status_code)).inc()

                if response.is_success:
                    self._breaker.record_success()
                    EDGAR_CIRCUIT_STATE.set(STATE_CODE[self._breaker.state])
                    return response

                last_status = response.status_code
                if response.status_code not in RETRYABLE_STATUS:
                    # A client error is a definitive answer. Do not retry it and
                    # do not count it against the breaker - EDGAR is healthy.
                    raise EdgarError(
                        f"EDGAR returned {response.status_code} for {url}",
                        url=url,
                        status=response.status_code,
                    )

                self._breaker.record_failure()
                last_error = f"HTTP {response.status_code}"
                retry_after = self._retry_after(response)

            EDGAR_CIRCUIT_STATE.set(STATE_CODE[self._breaker.state])

            if attempt < self.settings.edgar_max_retries:
                delay = self._backoff(attempt, retry_after)
                log.warning(
                    "edgar_retry",
                    url=url,
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    of=self.settings.edgar_max_retries,
                    reason=last_error,
                    sleeping=round(delay, 2),
                )
                self._sleep(delay)

        raise EdgarError(
            f"EDGAR request failed after {self.settings.edgar_max_retries + 1} attempts "
            f"({last_error}) for {url}",
            url=url,
            status=last_status,
        )

    # --- endpoints --------------------------------------------------------
    def get_daily_index(self, on: date, forms: tuple[str, ...] | None = None) -> list[FilingRef]:
        """Every filing disseminated on a date, optionally filtered by form type.

        Uses the pipe-delimited ``master`` index rather than the fixed-width
        ``form`` index; the delimiter removes a class of column-alignment bugs.
        EDGAR publishes no index on weekends and market holidays, which surfaces
        as a 404 and is returned here as an empty list, not an error.
        """
        quarter = (on.month - 1) // 3 + 1
        url = (
            f"{self.settings.edgar_daily_index_base}/{on.year}/QTR{quarter}/master.{on:%Y%m%d}.idx"
        )
        try:
            response = self._get(url, endpoint="daily_index")
        except EdgarError as exc:
            # EDGAR answers 403, not 404, for a daily index that does not exist -
            # every weekend and every market holiday. A backfill with
            # catchup=True over a year crosses about 114 such dates, so treating
            # only 404 as "no index" fails the DAG on all of them.
            #
            # Conflating this with a genuine auth 403 is the risk, and it is
            # bounded: a rejected User-Agent fails on EVERY date rather than on
            # non-publishing ones, and Settings refuses to construct without a
            # contact address in the first place.
            if exc.status in (403, 404):
                log.info(
                    "edgar_no_index_for_date",
                    date=on.isoformat(),
                    status=exc.status,
                    url=url,
                    note="weekend or market holiday; a 403 on every date means a bad User-Agent",
                )
                return []
            raise

        wanted = {f.upper() for f in forms} if forms else None
        rows: list[FilingRef] = []
        for line in response.text.splitlines():
            parts = line.split("|")
            if len(parts) != 5:
                continue
            cik, company, form, filed, path = (p.strip() for p in parts)
            if not cik.isdigit() or len(filed) != 8 or not filed.isdigit():
                continue  # skips the header row and the preamble
            if wanted and form.upper() not in wanted:
                continue
            try:
                filed_date = date(int(filed[0:4]), int(filed[4:6]), int(filed[6:8]))
            except ValueError:
                continue
            rows.append(FilingRef(cik=cik, company=company, form=form, filed=filed_date, path=path))

        log.info("edgar_daily_index", date=on.isoformat(), matched=len(rows), forms=forms)
        return rows

    def get_filing_index(self, cik: str | int, accession: str) -> dict[str, Any]:
        """The per-filing directory listing, used to locate the primary document."""
        url = (
            f"{self.settings.edgar_archives_base}/edgar/data/"
            f"{bare_cik(cik)}/{accession.replace('-', '')}/index.json"
        )
        return dict(self._get(url, endpoint="filing_index").json())

    def fetch_filing_document(self, cik: str | int, accession: str, filename: str) -> bytes:
        """Raw bytes of one document inside a filing.

        Bytes, not text: the archive is immutable, so we store exactly what
        EDGAR served and defer every decoding decision to the parser.
        """
        url = (
            f"{self.settings.edgar_archives_base}/edgar/data/"
            f"{bare_cik(cik)}/{accession.replace('-', '')}/{filename}"
        )
        return self._get(url, endpoint="filing_document").content

    def get_company_facts(self, cik: str | int) -> dict[str, Any]:
        """Every XBRL fact the company has ever tagged. This is the answer key."""
        url = f"{self.settings.edgar_data_api_base}/companyfacts/CIK{pad_cik(cik)}.json"
        return dict(self._get(url, endpoint="companyfacts").json())

    def get_submissions(self, cik: str | int) -> dict[str, Any]:
        """Filing history for one company, including the primary document name."""
        url = f"{self.settings.edgar_submissions_base}/CIK{pad_cik(cik)}.json"
        return dict(self._get(url, endpoint="submissions").json())

    def get_company_tickers(self) -> dict[str, Any]:
        """The ticker-to-CIK map used to build the corpus universe."""
        return dict(
            self._get(self.settings.edgar_company_tickers_url, endpoint="company_tickers").json()
        )

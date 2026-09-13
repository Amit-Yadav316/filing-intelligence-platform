"""Daily EDGAR ingest.

``logical_date`` selects which daily index to process, which is what makes a
backfill a replay of history rather than a re-download of "today". Re-running a
date converges on the same partition because :class:`ArchiveWriter` compares
content hashes before writing - so ``catchup=True`` over a year is safe to run
twice.

Airflow parses this directory on a loop, so there is no logic here beyond
wiring. Everything it calls lives in ``src/`` and runs from a shell without an
orchestrator, which is how it was debugged.
"""

from __future__ import annotations

import pendulum
from airflow.datasets import Dataset
from airflow.decorators import dag, task

RAW_FILINGS = Dataset("s3://filings/raw")


@dag(
    dag_id="ingest_edgar_filings",
    schedule="@daily",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=True,
    # EDGAR asks consumers to stay under 10 requests/second. Parallel DAG runs
    # would each hold their own rate limiter and blow through that ceiling
    # collectively, so history is replayed one day at a time.
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=5)},
    tags=["ingest", "edgar"],
    doc_md=__doc__,
)
def ingest_edgar_filings() -> None:
    @task
    def discover(logical_date: pendulum.DateTime) -> list[dict]:
        """Corpus filings disseminated on this date."""
        from src.ingest.ingest_service import FilingIngestService

        with FilingIngestService() as service:
            refs = service.discover(logical_date.date(), ciks=service.universe_ciks())
            # XCom carries identifiers, never documents.
            return [
                {
                    "cik": r.cik,
                    "company": r.company,
                    "form": r.form,
                    "filed": r.filed.isoformat(),
                    "path": r.path,
                }
                for r in refs
            ]

    @task
    def land(refs: list[dict]) -> dict:
        """Fetch each primary document into immutable object storage."""
        from datetime import date

        from src.ingest.edgar_client import FilingRef
        from src.ingest.ingest_service import FilingIngestService

        landed = failed = 0
        with FilingIngestService() as service:
            for raw in refs:
                result = service.ingest_filing(
                    FilingRef(
                        cik=raw["cik"],
                        company=raw["company"],
                        form=raw["form"],
                        filed=date.fromisoformat(raw["filed"]),
                        path=raw["path"],
                    )
                )
                landed += 1 if result.ok else 0
                failed += 0 if result.ok else 1
        return {"landed": landed, "failed": failed}

    @task(outlets=[RAW_FILINGS])
    def land_company_facts(refs: list[dict]) -> dict:
        """Refresh XBRL ground truth once per company, not once per filing.

        Publishing the dataset here rather than after `land` means downstream
        processing only wakes once the answer key is present too - otherwise
        extraction can run against a filing whose facts have not landed.
        """
        from src.ingest.ingest_service import FilingIngestService

        refreshed = 0
        with FilingIngestService() as service:
            for cik in sorted({r["cik"] for r in refs}):
                _, status = service.ingest_company_facts(cik)
                refreshed += 1 if status in {"landed", "overwritten"} else 0
        return {"companies_refreshed": refreshed}

    found = discover()
    land(found) >> land_company_facts(found)


ingest_edgar_filings()

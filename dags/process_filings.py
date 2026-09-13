"""Parse, chunk, embed and index whatever ingest landed.

Dataset-scheduled rather than time-scheduled: this runs because new filings
exist, not because the clock struck. That matters on a weekend, when EDGAR
publishes nothing and a ``@daily`` schedule would produce a run with no work.

Reprocessing is cheap by design. Content hashes in Postgres skip chunks whose
text has not changed, and the Redis embedding cache catches the rest - measured
at a 98.9% hit rate across a full re-index after a chunker change that altered
every chunk id. That is what makes "change the chunker and replay the corpus"
an afternoon rather than a budget decision.
"""

from __future__ import annotations

import pendulum
from airflow.datasets import Dataset
from airflow.decorators import dag, task, task_group

RAW_FILINGS = Dataset("s3://filings/raw")
INDEXED_CHUNKS = Dataset("postgres://filings/chunks")


@dag(
    dag_id="process_filings",
    schedule=[RAW_FILINGS],
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=2)},
    tags=["process", "index"],
    doc_md=__doc__,
)
def process_filings() -> None:
    @task
    def ensure_schema() -> str:
        """Idempotent DDL, so a fresh database needs no manual step."""
        from src.retrieve.store import ChunkStore

        with ChunkStore() as store:
            store.apply_schema()
        return "ready"

    @task
    def list_landed(_ready: str) -> list[str]:
        from src.retrieve.pipeline import IndexingPipeline

        return IndexingPipeline().list_landed_filings()

    @task_group(group_id="index")
    def index_group(accessions: list[str]) -> dict:
        @task
        def index_all(targets: list[str]) -> dict:
            """Parse, chunk, embed and upsert.

            One task rather than one per filing: the embedding model costs
            several seconds to load, and paying that per filing in a separate
            worker process would dominate the runtime.
            """
            from src.retrieve.pipeline import IndexingPipeline

            pipeline = IndexingPipeline()
            results = pipeline.index_all(targets)
            cache = pipeline.embedder.stats
            return {
                "filings": len(results),
                "failed": sum(1 for r in results if not r.ok),
                "chunks": sum(r.chunks for r in results),
                "embedded": sum(r.embedded for r in results),
                "skipped_unchanged": sum(r.skipped_unchanged for r in results),
                "pruned": sum(r.pruned for r in results),
                "cache_hit_rate": round(cache.hit_rate, 4),
            }

        return index_all(accessions)

    @task(outlets=[INDEXED_CHUNKS])
    def publish_index_freshness(summary: dict) -> dict:
        """Record how far behind EDGAR the index is.

        Computed from the index itself rather than from this run's self-report:
        a DAG that fails silently would otherwise keep publishing a freshness
        it no longer has.
        """
        from src.observability.metrics import INDEX_FRESHNESS
        from src.retrieve.store import ChunkStore

        with ChunkStore() as store:
            seconds = store.index_freshness_seconds()
            stats = store.stats()
        if seconds is not None:
            INDEX_FRESHNESS.set(seconds)
        return {**summary, "index_freshness_seconds": seconds, "chunks_total": stats.chunks}

    ready = ensure_schema()
    publish_index_freshness(index_group(list_landed(ready)))


process_filings()

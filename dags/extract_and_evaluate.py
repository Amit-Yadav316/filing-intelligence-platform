"""Extract financial facts, score them against XBRL, and gate on the result.

The quality gate is the point of this DAG. A ``ShortCircuitOperator`` sits
between scoring and publication: if batch accuracy on ``total_revenue`` falls
below the SLO floor, the extractions are **not** written to the authoritative
store, and the downstream publish tasks are skipped rather than failed.

That distinction is deliberate. A degraded model is not a broken pipeline - the
DAG did exactly what it should, which is notice and refuse. Failing the run
would page someone at 3am for a model regression that is safely contained;
skipping it leaves the previous good extractions in place, records the
accuracy metric that triggered it, and lets the alert fire on the metric rather
than on the task.
"""

from __future__ import annotations

import pendulum
from airflow.datasets import Dataset
from airflow.decorators import dag, task
from airflow.operators.python import ShortCircuitOperator

INDEXED_CHUNKS = Dataset("postgres://filings/chunks")
SCORED_EXTRACTIONS = Dataset("mongo://filings/extractions")


@dag(
    dag_id="extract_and_evaluate",
    schedule=[INDEXED_CHUNKS],
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=3)},
    tags=["extract", "evaluate", "quality-gate"],
    doc_md=__doc__,
)
def extract_and_evaluate() -> None:
    @task
    def extract_and_score() -> dict:
        """Run extraction over the indexed corpus and score every field."""
        from scripts.run_extraction import indexed_filings, load_facts
        from src.config.settings import get_settings
        from src.embed.embedding_service import EmbeddingService
        from src.evaluate.evaluator import ExtractionEvaluator
        from src.evaluate.xbrl_resolver import XBRLResolver
        from src.extract.extraction_service import ExtractionService
        from src.ingest.edgar_client import EdgarClient
        from src.retrieve.hybrid import HybridRetriever
        from src.retrieve.indexes import BM25Index, VectorIndex
        from src.retrieve.store import ChunkStore, fiscal_year_for

        settings = get_settings()
        resolver = XBRLResolver.from_config()
        evaluator = ExtractionEvaluator(resolver, settings)
        cards = []

        with ChunkStore(settings) as store, EdgarClient(settings) as edgar:
            retriever = HybridRetriever(
                BM25Index(store, settings),
                VectorIndex(store, settings),
                EmbeddingService(settings),
                settings,
            )
            service = ExtractionService(retriever, settings)
            for filing in indexed_filings(store):
                facts = load_facts(filing["cik"], edgar)
                fiscal_year = resolver.fiscal_year_for_accession(
                    facts,
                    filing["accession"],
                    fiscal_year_for(filing["form"], filing["filed"]),
                )
                outcome = service.extract(
                    filing["accession"],
                    cik=filing["cik"],
                    form=filing["form"],
                    fiscal_year=fiscal_year,
                    company=filing["company"],
                )
                if outcome.ok and outcome.extraction is not None:
                    cards.append(evaluator.score(outcome.extraction, facts).as_dict())

        return {"scorecards": cards, "filings": len(cards)}

    @task
    def publish_accuracy_metrics(batch: dict) -> dict:
        """Push per-field accuracy to Prometheus before the gate decides.

        Order matters: the metric must exist whether or not the gate passes, or
        the dashboard goes blank exactly when something is wrong.
        """
        from src.evaluate.evaluator import CORRECT, SCOREABLE
        from src.observability.metrics import EXTRACTION_ACCURACY

        per_field: dict[str, dict[str, int]] = {}
        for card in batch["scorecards"]:
            for score in card["scores"]:
                bucket = per_field.setdefault(score["field"], {"correct": 0, "scoreable": 0})
                if score["verdict"] in {str(v) for v in SCOREABLE}:
                    bucket["scoreable"] += 1
                if score["verdict"] in {str(v) for v in CORRECT}:
                    bucket["correct"] += 1

        accuracies = {}
        for name, bucket in per_field.items():
            ratio = bucket["correct"] / bucket["scoreable"] if bucket["scoreable"] else 0.0
            EXTRACTION_ACCURACY.labels(field=name).set(ratio)
            accuracies[name] = round(ratio, 4)
        return accuracies

    def _above_slo_floor(**context) -> bool:
        """The gate. False skips publication rather than failing the run."""
        from src.config.settings import get_settings

        accuracies = context["ti"].xcom_pull(task_ids="publish_accuracy_metrics")
        floor = get_settings().slo_revenue_accuracy_floor
        measured = (accuracies or {}).get("total_revenue", 0.0)
        passed = measured >= floor
        print(
            f"quality gate: total_revenue accuracy {measured:.1%} against floor "
            f"{floor:.0%} -> {'PASS, publishing' if passed else 'BLOCK, not publishing'}"
        )
        return passed

    quality_gate = ShortCircuitOperator(
        task_id="quality_gate",
        python_callable=_above_slo_floor,
        # Skip only what follows, not sibling branches.
        ignore_downstream_trigger_rules=False,
    )

    @task(outlets=[SCORED_EXTRACTIONS])
    def publish_extractions(batch: dict) -> dict:
        """Write extractions to the document store as authoritative.

        Only reached when the gate passed, which is the whole point.
        """
        from src.evaluate.store import ScorecardStore

        with ScorecardStore() as store:
            written = store.upsert_many(batch["scorecards"])
        return {"published": written}

    batch = extract_and_score()
    accuracies = publish_accuracy_metrics(batch)
    accuracies >> quality_gate >> publish_extractions(batch)


extract_and_evaluate()

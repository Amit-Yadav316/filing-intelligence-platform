"""MongoDB store for extractions and scorecards.

Why Mongo and not another Postgres table
----------------------------------------
A scorecard is a nested, evolving document: a filing, four fields, each with a
verdict, an extracted value, a ground-truth value, a tag, a relative error and
a list of supporting chunk ids. It is read whole and written whole, and its
shape changes whenever the schema version changes - a new scored field adds a
key rather than a migration.

Modelling that relationally means either a join across three tables to read one
scorecard, or a JSONB column that is a document store wearing a Postgres badge.
The chunks table is genuinely relational and lives in Postgres; scorecards are
genuinely documents and live here. That split is the justification, not the
JD checkbox.

Every write is an upsert keyed on (accession, schema_version, model), so
re-running the evaluation replaces a filing's scorecard rather than
accumulating one per run - and a different model's results never overwrite
another's, which would silently mix two models into one accuracy table.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger

log = get_logger(__name__)

SCORECARDS = "scorecards"
EXTRACTIONS = "extractions"


class ScorecardStore:
    """Upserts and reads evaluation results."""

    def __init__(self, settings: Settings | None = None, *, client: Any = None) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._owns = client is None

    # --- lifecycle --------------------------------------------------------
    def connect(self) -> Any:
        if self._client is None:
            from pymongo import MongoClient

            self._client = MongoClient(
                self.settings.mongo_uri, serverSelectionTimeoutMS=5000
            )
        return self._client

    @property
    def db(self) -> Any:
        return self.connect()[self.settings.mongo_db]

    def close(self) -> None:
        if self._client is not None and self._owns:
            self._client.close()
        self._client = None

    def __enter__(self) -> ScorecardStore:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ensure_indexes(self) -> None:
        """Idempotent. The unique index is what makes upsert semantics real."""
        self.db[SCORECARDS].create_index(
            [("accession", 1), ("schema_version", 1), ("model", 1)],
            unique=True,
            name="filing_schema_model",
        )
        self.db[SCORECARDS].create_index([("evaluated_at", -1)], name="recent")
        self.db[SCORECARDS].create_index([("cik", 1), ("fiscal_year", -1)], name="by_company")

    # --- writes -----------------------------------------------------------
    def upsert_many(self, scorecards: list[dict[str, Any]]) -> int:
        if not scorecards:
            return 0
        from pymongo import ReplaceOne

        now = datetime.now(UTC)
        operations = []
        for card in scorecards:
            document = {
                **card,
                "schema_version": card.get(
                    "schema_version", self.settings.extraction_schema_version
                ),
                "evaluated_at": now,
            }
            operations.append(
                ReplaceOne(
                    {
                        "accession": document["accession"],
                        "schema_version": document["schema_version"],
                        "model": document.get("model", ""),
                    },
                    document,
                    upsert=True,
                )
            )
        result = self.db[SCORECARDS].bulk_write(operations, ordered=False)
        written = result.upserted_count + result.modified_count
        log.info("scorecards_upserted", count=written, submitted=len(scorecards))
        return int(written)

    def upsert_extraction(self, extraction: dict[str, Any]) -> None:
        self.db[EXTRACTIONS].replace_one(
            {
                "accession": extraction.get("accession"),
                "schema_version": extraction.get("schema_version"),
                "model": extraction.get("model"),
            },
            {**extraction, "stored_at": datetime.now(UTC)},
            upsert=True,
        )

    # --- reads ------------------------------------------------------------
    def get(self, accession: str) -> dict[str, Any] | None:
        return self.db[SCORECARDS].find_one(
            {"accession": accession}, sort=[("evaluated_at", -1)]
        )

    def latest(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(
            self.db[SCORECARDS]
            .find({}, {"_id": 0})
            .sort("evaluated_at", -1)
            .limit(limit)
        )

    def accuracy_by_field(self) -> dict[str, float]:
        """Per-field accuracy across everything stored.

        Aggregated in the database rather than in Python so the API can serve
        it without loading every scorecard.
        """
        pipeline = [
            {"$unwind": "$scores"},
            {
                "$match": {
                    "scores.verdict": {
                        "$in": [
                            "exact",
                            "within_tolerance",
                            "scale_error",
                            "wrong",
                            "hallucinated",
                            "abstained",
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": "$scores.field",
                    "scoreable": {"$sum": 1},
                    "correct": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$in": [
                                        "$scores.verdict",
                                        ["exact", "within_tolerance"],
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                }
            },
        ]
        return {
            row["_id"]: (row["correct"] / row["scoreable"] if row["scoreable"] else 0.0)
            for row in self.db[SCORECARDS].aggregate(pipeline)
        }

    def count(self) -> int:
        return int(self.db[SCORECARDS].count_documents({}))

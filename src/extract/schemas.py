"""The extraction contract.

The model never returns free text. It fills this schema or it abstains, and
both outcomes are scored - which is the only reason the accuracy table in the
README can exist.

**Abstention is a first-class outcome.** A model that says "I could not find
operating cash flow in these chunks" is more useful than one that guesses, and
the evaluator scores abstention in its own bucket rather than counting it as a
miss. Rewarding a confident wrong answer over an honest "not found" is how
extraction systems end up unusable in practice.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator

SCORED_FIELDS: tuple[str, ...] = (
    "total_revenue",
    "net_income",
    "total_assets",
    "operating_cash_flow",
)


class FilingExtraction(BaseModel):
    """Structured financial facts extracted from one filing."""

    fiscal_year: int = Field(description="Fiscal year the figures cover.")

    total_revenue: Decimal | None = Field(
        default=None, description="Total revenue or net sales, in full units."
    )
    net_income: Decimal | None = Field(default=None)
    total_assets: Decimal | None = Field(default=None)
    operating_cash_flow: Decimal | None = Field(default=None)

    reported_currency: str = Field(default="USD")
    top_risk_categories: list[str] = Field(default_factory=list)

    # Provenance. `source_chunks` is every chunk the answer drew on;
    # `field_sources` narrows it per field, so a citation points at the chunk
    # that actually carried the number rather than at the whole context window.
    source_chunks: list[str] = Field(default_factory=list)
    field_sources: dict[str, list[str]] = Field(default_factory=dict)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    abstained_fields: list[str] = Field(default_factory=list)

    # Bookkeeping, set by the service rather than the model.
    accession: str = ""
    cik: str = ""
    company: str | None = None
    form: str = ""
    model: str = ""
    schema_version: str = "v1"

    @field_validator("abstained_fields")
    @classmethod
    def _known_fields_only(cls, v: list[str]) -> list[str]:
        return [f for f in v if f in SCORED_FIELDS]

    def value_for(self, field: str) -> Decimal | None:
        return getattr(self, field, None)

    def abstained(self, field: str) -> bool:
        """True when the model declined this field.

        A null value is treated as abstention even if the model forgot to list
        it: refusing to answer and returning nothing are the same claim, and
        penalising a formatting lapse would distort the abstention rate.
        """
        return field in self.abstained_fields or self.value_for(field) is None

    def sources_for(self, field: str) -> list[str]:
        return self.field_sources.get(field) or self.source_chunks


# The JSON Schema handed to the model. Written out rather than generated from
# the pydantic model because Gemini's responseSchema accepts a restricted
# dialect - no $ref, no anyOf - and a generated schema trips over both.
GEMINI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "fiscal_year": {"type": "INTEGER"},
        "total_revenue": {"type": "NUMBER", "nullable": True},
        "net_income": {"type": "NUMBER", "nullable": True},
        "total_assets": {"type": "NUMBER", "nullable": True},
        "operating_cash_flow": {"type": "NUMBER", "nullable": True},
        "reported_currency": {"type": "STRING"},
        "top_risk_categories": {"type": "ARRAY", "items": {"type": "STRING"}},
        "abstained_fields": {"type": "ARRAY", "items": {"type": "STRING"}},
        "confidence": {"type": "NUMBER"},
        "field_sources": {
            "type": "OBJECT",
            "properties": {
                f: {"type": "ARRAY", "items": {"type": "STRING"}} for f in SCORED_FIELDS
            },
        },
    },
    # Every scored field is required, though still nullable. Leaving them
    # optional let the model omit a key entirely, which reads downstream as
    # an abstention without the model having decided to abstain. Required
    # plus nullable forces an explicit null.
    "required": [
        "fiscal_year",
        "reported_currency",
        "abstained_fields",
        "confidence",
        *SCORED_FIELDS,
    ],
}

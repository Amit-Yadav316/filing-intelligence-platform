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


# --- the response contract, in one place ---------------------------------
#
# Standard JSON Schema is the source of truth, because it is what every
# OpenAI-compatible provider speaks. Gemini accepts a restricted dialect of its
# own - uppercase type names, a `nullable` flag instead of union types, and no
# $ref or anyOf - so it is DERIVED from this rather than maintained separately.
# Two hand-written schemas would drift the moment a field is added, and the
# symptom would be one provider silently omitting a field the other requires.

EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fiscal_year": {"type": "integer"},
        "total_revenue": {"type": ["number", "null"]},
        "net_income": {"type": ["number", "null"]},
        "total_assets": {"type": ["number", "null"]},
        "operating_cash_flow": {"type": ["number", "null"]},
        "reported_currency": {"type": "string"},
        "top_risk_categories": {"type": "array", "items": {"type": "string"}},
        "abstained_fields": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "field_sources": {
            "type": "object",
            "properties": {
                f: {"type": "array", "items": {"type": "string"}} for f in SCORED_FIELDS
            },
        },
    },
    # Every scored field is required, though still nullable. Leaving them
    # optional let the model omit a key entirely, which reads downstream as an
    # abstention without the model having decided to abstain. Required plus
    # nullable forces an explicit null.
    "required": [
        "fiscal_year",
        "reported_currency",
        "abstained_fields",
        "confidence",
        *SCORED_FIELDS,
    ],
    "additionalProperties": False,
}


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Translate standard JSON Schema into Gemini's responseSchema dialect."""
    node: dict[str, Any] = {}
    raw_type = schema.get("type")

    if isinstance(raw_type, list):
        # ["number", "null"] becomes a nullable NUMBER.
        concrete = [x for x in raw_type if x != "null"]
        node["type"] = concrete[0].upper() if concrete else "STRING"
        if "null" in raw_type:
            node["nullable"] = True
    elif isinstance(raw_type, str):
        node["type"] = raw_type.upper()

    if "properties" in schema:
        node["properties"] = {k: to_gemini_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        node["items"] = to_gemini_schema(schema["items"])
    if "required" in schema:
        node["required"] = list(schema["required"])
    # additionalProperties is not part of the accepted dialect and is dropped.
    return node


GEMINI_RESPONSE_SCHEMA: dict[str, Any] = to_gemini_schema(EXTRACTION_JSON_SCHEMA)

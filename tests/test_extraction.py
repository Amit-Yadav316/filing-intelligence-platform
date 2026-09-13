"""Extraction service tests.

The LLM is injected, so these run offline and deterministically. What is being
tested is the machinery around the model: does a schema violation retry once
and then abstain, is the context built from retrieval rather than the whole
filing, does the cache key change when the model does.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from src.extract.extraction_service import ExtractionService
from src.extract.llm_client import LLMClient, LLMError, LLMResponse
from src.extract.schemas import SCORED_FIELDS, FilingExtraction
from src.retrieve.indexes import SearchHit

GOOD_PAYLOAD = """{
  "fiscal_year": 2023,
  "total_revenue": 383285000000,
  "net_income": 96995000000,
  "total_assets": null,
  "operating_cash_flow": 110543000000,
  "reported_currency": "USD",
  "top_risk_categories": ["supply chain"],
  "abstained_fields": ["total_assets"],
  "confidence": 0.9,
  "field_sources": {"total_revenue": ["c1"]}
}"""


def hit(chunk_id: str, *, kind: str = "table", char_start: int = 0) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=1.0,
        rank=1,
        accession="0000320193-23-000106",
        cik="0000320193",
        company="Apple Inc.",
        form="10-K",
        filed=date(2023, 11, 3),
        fiscal_year=2023,
        item_number="8",
        item_title="Financial Statements",
        section="Financial Statements",
        chunk_type=kind,
        char_start=char_start,
        char_end=char_start + 50,
        token_count=40,
        text=f"body of {chunk_id}",
    )


class FakeLLM(LLMClient):
    """Returns queued payloads and counts calls."""

    def __init__(self, settings, payloads: list[str]) -> None:
        super().__init__(settings, sleep=lambda _s: None)
        self.payloads = list(payloads)
        self.calls: list[str] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def _post(self, prompt: str, schema: dict[str, Any], system: str | None) -> LLMResponse:
        self.calls.append(prompt)
        if not self.payloads:
            raise LLMError("no payloads queued")
        return LLMResponse(self.payloads.pop(0), 100, 20, "fake-model", 0.0001)


class FakeRetriever:
    """Records the searches made and returns fixed hits."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits
        self.searches: list[tuple[str, Any]] = []
        self.bm25 = self

    def search(self, query: str, *, top_k: int = 10, k: int = 10, filters=None, **_: Any):
        self.searches.append((query, filters))
        limit = top_k or k
        if self is getattr(self, "bm25", None):
            pass
        return [_Fused(h) for h in self._hits[:limit]]


class _Fused:
    def __init__(self, hit_: SearchHit) -> None:
        self.hit = hit_


class BM25Only(FakeRetriever):
    """bm25.search returns bare hits, not fused ones."""

    def search(self, query: str, *, top_k: int = 10, k: int = 10, filters=None, **_: Any):
        self.searches.append((query, filters))
        return [_Fused(h) for h in self._hits[: (top_k or k)]]


@pytest.fixture
def retriever() -> FakeRetriever:
    r = FakeRetriever([hit(f"c{i}", char_start=i * 100) for i in range(6)])

    class _BM:
        def __init__(self, outer: FakeRetriever) -> None:
            self.outer = outer

        def search(self, query: str, *, k: int = 2, filters=None, **_: Any):
            self.outer.searches.append((query, filters))
            return self.outer._hits[:k]

    r.bm25 = _BM(r)  # type: ignore[assignment]
    return r


def service(settings, retriever: FakeRetriever, payloads: list[str]) -> ExtractionService:
    return ExtractionService(
        retriever,  # type: ignore[arg-type]
        settings,
        llm=FakeLLM(settings, payloads),
        redis_client=None,
    )


def extract(svc: ExtractionService) -> Any:
    return svc.extract(
        "0000320193-23-000106",
        cik="0000320193",
        form="10-K",
        fiscal_year=2023,
        company="Apple Inc.",
        use_cache=False,
    )


# --- happy path -----------------------------------------------------------
def test_parses_a_valid_extraction(settings, retriever) -> None:
    outcome = extract(service(settings, retriever, [GOOD_PAYLOAD]))

    assert outcome.ok
    e = outcome.extraction
    assert e.total_revenue == Decimal("383285000000")
    assert e.fiscal_year == 2023
    assert e.confidence == 0.9


def test_records_token_usage_and_cost(settings, retriever) -> None:
    outcome = extract(service(settings, retriever, [GOOD_PAYLOAD]))

    assert outcome.input_tokens == 100
    assert outcome.output_tokens == 20
    assert outcome.cost_usd > 0


def test_bookkeeping_fields_come_from_the_caller_not_the_model(settings, retriever) -> None:
    """The model must not be trusted to echo the accession it was given."""
    e = extract(service(settings, retriever, [GOOD_PAYLOAD])).extraction

    assert e.accession == "0000320193-23-000106"
    assert e.cik == "0000320193"
    assert e.company == "Apple Inc."


# --- abstention -----------------------------------------------------------
def test_a_null_field_is_treated_as_abstention(settings, retriever) -> None:
    e = extract(service(settings, retriever, [GOOD_PAYLOAD])).extraction

    assert e.abstained("total_assets")
    assert not e.abstained("total_revenue")


def test_unknown_field_names_are_dropped_from_abstentions(settings, retriever) -> None:
    payload = GOOD_PAYLOAD.replace('["total_assets"]', '["total_assets", "ebitda"]')

    e = extract(service(settings, retriever, [payload])).extraction

    assert e.abstained_fields == ["total_assets"]


# --- schema violations ----------------------------------------------------
def test_retries_once_on_malformed_json(settings, retriever) -> None:
    svc = service(settings, retriever, ["this is not json", GOOD_PAYLOAD])

    outcome = extract(svc)

    assert outcome.ok
    assert outcome.retried
    assert len(svc.llm.calls) == 2  # type: ignore[attr-defined]


def test_abstains_on_everything_after_a_second_failure(settings, retriever) -> None:
    """Better to return nothing than to invent a shape that validates."""
    svc = service(settings, retriever, ["not json", "still not json"])

    outcome = extract(svc)

    assert outcome.ok, "a schema failure must still produce a scoreable result"
    assert set(outcome.extraction.abstained_fields) == set(SCORED_FIELDS)
    assert outcome.extraction.confidence == 0.0
    assert "schema violation" in (outcome.error or "")


def test_fenced_json_is_tolerated(settings, retriever) -> None:
    payload = f"```json\n{GOOD_PAYLOAD}\n```"

    assert extract(service(settings, retriever, [payload])).ok


def test_a_non_finite_number_becomes_an_abstention(settings, retriever) -> None:
    payload = GOOD_PAYLOAD.replace("383285000000", '"NaN"')

    e = extract(service(settings, retriever, [payload])).extraction

    assert e.total_revenue is None
    assert e.abstained("total_revenue")


# --- context construction -------------------------------------------------
def test_context_is_retrieved_not_the_whole_filing(settings, retriever) -> None:
    """If this ever sends the full document, retrieval becomes decorative and
    per-field provenance becomes impossible."""
    svc = service(settings, retriever, [GOOD_PAYLOAD])

    extract(svc)

    assert retriever.searches, "no retrieval was performed"
    prompt = svc.llm.calls[0]  # type: ignore[attr-defined]
    assert "chunk_id" in prompt
    assert len(prompt) < 50_000


def test_every_context_search_is_filtered_to_the_target_filing(settings, retriever) -> None:
    """Without the accession filter, one company's balance sheet can be used to
    answer about another."""
    extract(service(settings, retriever, [GOOD_PAYLOAD]))

    assert retriever.searches
    for _query, filters in retriever.searches:
        assert filters is not None
        assert filters.accession == "0000320193-23-000106"


def test_context_chunks_are_deduplicated(settings, retriever) -> None:
    svc = service(settings, retriever, [GOOD_PAYLOAD])

    hits = svc.build_context("0000320193-23-000106")

    assert len({h.chunk_id for h in hits}) == len(hits)


def test_context_respects_the_budget(settings, retriever) -> None:
    svc = service(settings, retriever, [GOOD_PAYLOAD])

    hits = svc.build_context("0000320193-23-000106")

    assert len(hits) <= settings.extraction_context_chunks


def test_no_chunks_retrieved_is_an_error_not_a_hallucination(settings) -> None:
    empty = FakeRetriever([])

    class _BM:
        def search(self, *_a: Any, **_k: Any):
            return []

    empty.bm25 = _BM()  # type: ignore[assignment]
    outcome = extract(service(settings, empty, [GOOD_PAYLOAD]))

    assert not outcome.ok
    assert "no chunks" in (outcome.error or "")


# --- caching --------------------------------------------------------------
def test_cache_key_includes_model_and_schema_version(settings, retriever) -> None:
    """Re-running after a model swap must re-extract, or the accuracy table
    reports one model's numbers under another's name."""
    svc = service(settings, retriever, [GOOD_PAYLOAD])
    key = svc.cache_key("acc-1")

    assert settings.llm_model in key
    assert settings.extraction_schema_version in key
    assert "acc-1" in key


def test_llm_failure_is_reported_not_silently_abstained(settings, retriever) -> None:
    """A provider outage must not look like a model that declined to answer."""
    svc = service(settings, retriever, [])

    outcome = extract(svc)

    assert not outcome.ok
    assert outcome.extraction is None
    assert outcome.error


# --- schema helpers -------------------------------------------------------
def test_sources_for_falls_back_to_all_chunks() -> None:
    e = FilingExtraction(
        fiscal_year=2023,
        source_chunks=["a", "b"],
        field_sources={"total_revenue": ["a"]},
    )

    assert e.sources_for("total_revenue") == ["a"]
    assert e.sources_for("net_income") == ["a", "b"]

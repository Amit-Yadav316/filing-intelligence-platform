"""Retrieval tests that need no database and no model.

Fusion, filtering and cache-key construction are pure logic, and they are where
the subtle mistakes live: an off-by-one in RRF quietly reorders results, and a
cache key that omits the model name serves 384-dimension vectors for a
768-dimension model.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.embed.embedding_service import CacheKeyBuilder, EmbeddingService
from src.retrieve.hybrid import reciprocal_rank_fusion
from src.retrieve.indexes import MetadataFilter, SearchHit
from src.retrieve.store import fiscal_year_for


def hit(chunk_id: str, *, rank: int = 1, cik: str = "0000320193", item: str = "1A") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=1.0 / rank,
        rank=rank,
        accession="0000320193-23-000106",
        cik=cik,
        company="Apple Inc.",
        form="10-K",
        filed=date(2023, 11, 3),
        fiscal_year=2023,
        item_number=item,
        item_title="Risk Factors",
        section="Risk Factors",
        chunk_type="prose",
        char_start=0,
        char_end=100,
        token_count=50,
        text="text",
    )


# --- reciprocal rank fusion ----------------------------------------------
def test_a_chunk_found_by_both_arms_outranks_one_found_by_either() -> None:
    """The whole point of fusion: agreement between independent retrievers is
    evidence, even when neither ranked it first."""
    runs = {
        "bm25": [hit("solo-lex"), hit("both"), hit("x")],
        "dense": [hit("solo-dense"), hit("both"), hit("y")],
    }

    fused = reciprocal_rank_fusion(runs, k=60)

    assert fused[0].hit.chunk_id == "both"
    assert fused[0].in_both_arms is True
    assert fused[0].found_by == ("bm25", "dense")


def test_rrf_score_matches_the_definition() -> None:
    runs = {"bm25": [hit("a"), hit("b")], "dense": [hit("b"), hit("a")]}

    fused = {f.hit.chunk_id: f.rrf_score for f in reciprocal_rank_fusion(runs, k=60)}

    expected = 1 / 61 + 1 / 62  # rank 1 in one arm, rank 2 in the other
    assert fused["a"] == pytest.approx(expected)
    assert fused["b"] == pytest.approx(expected)


def test_larger_k_flattens_the_advantage_of_rank_one() -> None:
    runs = {"bm25": [hit("first"), hit("second")]}

    tight = reciprocal_rank_fusion(runs, k=1)
    loose = reciprocal_rank_fusion(runs, k=1000)

    tight_gap = tight[0].rrf_score - tight[1].rrf_score
    loose_gap = loose[0].rrf_score - loose[1].rrf_score
    assert loose_gap < tight_gap


def test_fusion_works_when_one_arm_returns_nothing() -> None:
    """Degrading to a single arm must still produce results - a query that
    matches densely but not lexically is common on financial jargon."""
    fused = reciprocal_rank_fusion({"bm25": [], "dense": [hit("a"), hit("b")]}, k=60)

    assert [f.hit.chunk_id for f in fused] == ["a", "b"]
    assert all(not f.in_both_arms for f in fused)


def test_fusion_of_nothing_is_empty() -> None:
    assert reciprocal_rank_fusion({"bm25": [], "dense": []}, k=60) == []


def test_ordering_is_stable_for_tied_scores() -> None:
    """An unstable sort makes an ablation impossible to reproduce."""
    runs = {"bm25": [hit("b"), hit("a")], "dense": [hit("a"), hit("b")]}

    first = [f.hit.chunk_id for f in reciprocal_rank_fusion(runs, k=60)]
    second = [f.hit.chunk_id for f in reciprocal_rank_fusion(runs, k=60)]

    assert first == second == sorted(first)


def test_top_k_truncates_after_fusion_not_before() -> None:
    runs = {"bm25": [hit(f"c{i}") for i in range(10)], "dense": [hit("c9"), hit("c8")]}

    fused = reciprocal_rank_fusion(runs, k=60, top_k=3)

    assert len(fused) == 3
    # c9 was rank 10 lexically but rank 1 densely, so fusion must promote it.
    assert "c9" in [f.hit.chunk_id for f in fused]


# --- metadata filtering ---------------------------------------------------
def test_empty_filter_matches_everything() -> None:
    where, params = MetadataFilter().to_sql()

    assert where == "TRUE"
    assert params == []
    assert MetadataFilter().is_empty


def test_cik_is_zero_padded_to_match_storage() -> None:
    """The daily index gives unpadded CIKs and the store holds padded ones.
    Without normalising, a filter on a real company silently matches nothing."""
    where, params = MetadataFilter(cik="320193").to_sql()

    assert "cik = %s" in where
    assert params == ["0000320193"]


def test_filters_combine_with_and() -> None:
    where, params = MetadataFilter(cik="320193", form="10-K", fiscal_year=2023).to_sql()

    assert where.count(" AND ") == 2
    assert params == ["0000320193", "10-K", 2023]


def test_year_range_filter() -> None:
    where, params = MetadataFilter(year_from=2022, year_to=2024).to_sql()

    assert "fiscal_year >= %s" in where and "fiscal_year <= %s" in where
    assert params == [2022, 2024]


def test_multi_company_filter_uses_any() -> None:
    where, params = MetadataFilter(ciks=["320193", "789019"]).to_sql()

    assert "= ANY(%s)" in where
    assert params == [["0000320193", "0000789019"]]


# --- fiscal year derivation ----------------------------------------------
@pytest.mark.parametrize(
    ("form", "filed", "expected"),
    [
        ("10-K", date(2023, 11, 3), 2023),  # Apple, FY ends September
        ("10-K", date(2024, 2, 16), 2023),  # calendar-year filer reporting FY2023
        ("10-K", date(2023, 7, 27), 2023),  # Microsoft, FY ends June
        ("10-Q", date(2024, 8, 2), 2024),
    ],
)
def test_fiscal_year_heuristic(form: str, filed: date, expected: int) -> None:
    """A filtering convenience, not ground truth - scoring uses the XBRL period."""
    assert fiscal_year_for(form, filed) == expected


# --- cache keys -----------------------------------------------------------
def test_cache_key_changes_with_the_model() -> None:
    """Swapping models must invalidate every entry. A shared key would return
    384-dimension vectors for a 768-dimension model and fail deep inside
    pgvector with an error naming neither cause."""
    a = CacheKeyBuilder.content_hash("text", "bge-small-en-v1.5", 384)
    b = CacheKeyBuilder.content_hash("text", "bge-base-en-v1.5", 768)

    assert a != b


def test_cache_key_changes_with_the_text() -> None:
    a = CacheKeyBuilder.content_hash("alpha", "m", 384)
    b = CacheKeyBuilder.content_hash("beta", "m", 384)

    assert a != b


def test_cache_key_is_stable_across_calls() -> None:
    assert CacheKeyBuilder.content_hash("t", "m", 384) == CacheKeyBuilder.content_hash(
        "t", "m", 384
    )


def test_fields_cannot_be_confused_by_concatenation() -> None:
    """Without a separator, ("ab","c") and ("a","bc") would hash identically."""
    assert CacheKeyBuilder.content_hash("x", "ab", 384) != CacheKeyBuilder.content_hash(
        "x", "a", 384
    )


# --- embedding service cache behaviour ------------------------------------
class FakeModel:
    """Counts how many texts were actually encoded."""

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, texts, **_kwargs):
        import numpy as np

        self.encoded.extend(texts)
        return np.asarray([[float(len(t)), 0.0, 0.0] for t in texts], dtype=np.float32)


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def mget(self, keys):
        return [self.store.get(k) for k in keys]

    def pipeline(self):
        return _FakePipe(self)

    def ping(self):
        return True


class _FakePipe:
    def __init__(self, parent: FakeRedis) -> None:
        self.parent = parent
        self.ops: list[tuple[str, bytes]] = []

    def setex(self, key, _ttl, value):
        self.ops.append((key, value))

    def execute(self):
        for key, value in self.ops:
            self.parent.store[key] = value
        self.ops.clear()


@pytest.fixture
def service(settings):
    return EmbeddingService(settings, model=FakeModel(), redis_client=FakeRedis())


def test_second_call_embeds_nothing(service: EmbeddingService) -> None:
    """The day-3 gate: reprocessing an unchanged corpus must do no work."""
    texts = ["alpha", "beta", "gamma"]

    first = service.embed_texts(texts)
    encoded_after_first = len(service._model.encoded)
    second = service.embed_texts(texts)

    assert len(service._model.encoded) == encoded_after_first, "re-embedded cached text"
    assert first == second
    assert service.stats.hits == 3


def test_repeats_within_one_batch_are_embedded_once(service: EmbeddingService) -> None:
    """Filings share a lot of boilerplate, so this is not theoretical."""
    result = service.embed_texts(["same", "same", "same", "other"])

    assert len(service._model.encoded) == 2
    assert service.stats.duplicates == 2
    assert result[0] == result[1] == result[2]


def test_vectors_are_returned_in_input_order(service: EmbeddingService) -> None:
    out = service.embed_texts(["a", "bbb", "cc"])

    # FakeModel encodes length into the first component.
    assert [v[0] for v in out] == [1.0, 3.0, 2.0]


def test_a_changed_chunk_misses_the_cache(service: EmbeddingService) -> None:
    service.embed_texts(["original"])
    hits_before = service.stats.hits

    service.embed_texts(["original", "edited"])

    assert service.stats.hits == hits_before + 1
    assert service.stats.misses == 2  # one cold, one genuinely new


def test_an_unavailable_cache_degrades_to_recomputation(settings) -> None:
    """Redis is an optimisation, not a dependency. Losing it must not fail the
    pipeline, only make it slower."""
    service = EmbeddingService(settings, model=FakeModel(), use_cache=False)

    first = service.embed_texts(["alpha"])
    second = service.embed_texts(["alpha"])

    assert first == second
    assert len(service._model.encoded) == 2


def test_empty_input_does_nothing(service: EmbeddingService) -> None:
    assert service.embed_texts([]) == []
    assert service.stats.lookups == 0


def test_query_embedding_gets_the_bge_instruction_prefix(service: EmbeddingService) -> None:
    """BGE is trained asymmetrically. Omitting the prefix is a silent quality
    loss - nothing errors, retrieval is just worse."""
    service.embed_query("supply chain risk")

    assert service._model.encoded[-1].startswith(
        "Represent this sentence for searching relevant passages:"
    )


def test_passages_are_embedded_without_the_query_prefix(service: EmbeddingService) -> None:
    service.embed_texts(["supply chain risk"])

    assert service._model.encoded == ["supply chain risk"]

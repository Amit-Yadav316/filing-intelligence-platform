"""Evaluator tests.

The evaluator produces the numbers in the README, so a bug here does not crash
anything - it publishes a wrong accuracy figure. Each verdict is pinned
individually, and the boundaries between them are tested from both sides.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.evaluate.evaluator import CORRECT, ExtractionEvaluator, ScoreCard, Verdict
from src.evaluate.xbrl_resolver import GroundTruthFact
from src.extract.schemas import FilingExtraction

APPLE_REVENUE = Decimal("383285000000")


def truth(value: Decimal = APPLE_REVENUE, tag: str = "Revenues") -> GroundTruthFact:
    return GroundTruthFact(
        field="total_revenue",
        value=value,
        tag=tag,
        unit="USD",
        period_start=date(2022, 9, 25),
        period_end=date(2023, 9, 30),
        fiscal_year=2023,
        accession="0000320193-23-000106",
        form="10-K",
        filed=date(2023, 11, 3),
        restated=False,
    )


@pytest.fixture
def evaluator(settings) -> ExtractionEvaluator:
    return ExtractionEvaluator(settings=settings)


# --- the two gate cases from TASKS.md -------------------------------------
def test_scores_a_correct_extraction_as_exact(evaluator: ExtractionEvaluator) -> None:
    score = evaluator.score_field("total_revenue", APPLE_REVENUE, truth())

    assert score.verdict is Verdict.EXACT
    assert score.is_correct
    assert score.relative_error == 0.0


def test_scores_a_wrong_extraction_as_wrong(evaluator: ExtractionEvaluator) -> None:
    score = evaluator.score_field("total_revenue", Decimal("294135000000"), truth())

    assert score.verdict is Verdict.WRONG
    assert not score.is_correct
    assert score.relative_error == pytest.approx(0.2326, abs=1e-3)


# --- tolerance boundary ---------------------------------------------------
def test_rounding_in_the_prose_is_within_tolerance(evaluator: ExtractionEvaluator) -> None:
    """Filings round in narrative text. 383.3 billion against 383.285 billion is
    the document being readable, not the model being wrong."""
    score = evaluator.score_field("total_revenue", Decimal("383300000000"), truth())

    assert score.verdict is Verdict.WITHIN_TOLERANCE
    assert score.is_correct


def test_just_outside_tolerance_is_wrong(evaluator: ExtractionEvaluator) -> None:
    just_over = APPLE_REVENUE * Decimal("1.006")  # 0.6%, tolerance is 0.5%

    assert evaluator.score_field("total_revenue", just_over, truth()).verdict is Verdict.WRONG


# --- scale errors ---------------------------------------------------------
@pytest.mark.parametrize("factor", [1_000, 1_000_000, 1_000_000_000])
def test_reporting_in_the_tables_units_is_a_scale_error(
    evaluator: ExtractionEvaluator, factor: int
) -> None:
    """Statements are headed "in millions". A model that copies 383,285 straight
    off the page has made a systematic error, not a reading error, and the fix
    is the prompt rather than the model."""
    score = evaluator.score_field("total_revenue", APPLE_REVENUE / factor, truth())

    assert score.verdict is Verdict.SCALE_ERROR
    assert score.scale_factor == factor
    assert not score.is_correct


def test_scale_error_in_the_other_direction_is_also_caught(
    evaluator: ExtractionEvaluator,
) -> None:
    score = evaluator.score_field("total_revenue", APPLE_REVENUE * 1000, truth())

    assert score.verdict is Verdict.SCALE_ERROR


def test_a_near_miss_is_not_mistaken_for_a_scale_error(
    evaluator: ExtractionEvaluator,
) -> None:
    """Only a clean power of ten counts. 2x out is a wrong line item."""
    score = evaluator.score_field("total_revenue", APPLE_REVENUE * 2, truth())

    assert score.verdict is Verdict.WRONG


# --- hallucination --------------------------------------------------------
def test_an_unrelated_number_is_hallucinated(evaluator: ExtractionEvaluator) -> None:
    score = evaluator.score_field("total_revenue", Decimal("42"), truth())

    assert score.verdict is Verdict.HALLUCINATED
    assert not score.is_correct


def test_an_unparseable_value_is_hallucinated(evaluator: ExtractionEvaluator) -> None:
    score = evaluator.score_field("total_revenue", Decimal("NaN"), truth())

    assert score.verdict is Verdict.HALLUCINATED


# --- abstention -----------------------------------------------------------
def test_abstention_is_its_own_verdict_not_a_wrong_answer(
    evaluator: ExtractionEvaluator,
) -> None:
    """The feature the schema exists for: declining is more useful than
    guessing, and must not be scored as a failure."""
    score = evaluator.score_field("total_revenue", None, truth(), abstained=True)

    assert score.verdict is Verdict.ABSTAINED
    assert score.verdict not in CORRECT
    assert score.extracted is None


def test_a_null_value_counts_as_abstention_even_if_unflagged(
    evaluator: ExtractionEvaluator,
) -> None:
    """Returning nothing and refusing to answer are the same claim; penalising
    the formatting lapse would distort the abstention rate."""
    assert evaluator.score_field("total_revenue", None, truth()).verdict is Verdict.ABSTAINED


# --- unresolvable ---------------------------------------------------------
def test_missing_ground_truth_is_unresolvable_not_wrong(
    evaluator: ExtractionEvaluator,
) -> None:
    """A gap in the answer key is our problem, not the model's."""
    score = evaluator.score_field("total_revenue", APPLE_REVENUE, None)

    assert score.verdict is Verdict.UNRESOLVABLE
    assert score.truth is None


def test_unresolvable_is_excluded_from_the_accuracy_denominator(
    evaluator: ExtractionEvaluator,
) -> None:
    """A denominator quietly reduced by unscoreable cases is how accuracy
    figures get inflated, so this is asserted explicitly."""
    card = ScoreCard("acc", "cik", "Co", "10-K", 2023)
    card.scores = [
        evaluator.score_field("total_revenue", APPLE_REVENUE, truth()),
        evaluator.score_field("net_income", APPLE_REVENUE, None),
    ]

    assert len(card.scoreable) == 1
    assert card.accuracy == 1.0


def test_abstention_stays_in_the_denominator(evaluator: ExtractionEvaluator) -> None:
    """A model that abstains on everything is not 100% accurate."""
    card = ScoreCard("acc", "cik", "Co", "10-K", 2023)
    card.scores = [
        evaluator.score_field("total_revenue", APPLE_REVENUE, truth()),
        evaluator.score_field("net_income", None, truth(), abstained=True),
    ]

    assert len(card.scoreable) == 2
    assert card.accuracy == 0.5


# --- whole-filing scoring -------------------------------------------------
def facts_payload() -> dict:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 383285000000,
                                "start": "2022-09-25",
                                "end": "2023-09-30",
                                "form": "10-K",
                                "filed": "2023-11-03",
                                "accn": "0000320193-23-000106",
                            }
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 96995000000,
                                "start": "2022-09-25",
                                "end": "2023-09-30",
                                "form": "10-K",
                                "filed": "2023-11-03",
                                "accn": "0000320193-23-000106",
                            }
                        ]
                    }
                },
            }
        }
    }


def test_scores_a_whole_extraction(evaluator: ExtractionEvaluator) -> None:
    extraction = FilingExtraction(
        fiscal_year=2023,
        total_revenue=APPLE_REVENUE,
        net_income=None,
        abstained_fields=["net_income"],
        accession="0000320193-23-000106",
        cik="0000320193",
        company="Apple Inc.",
        form="10-K",
    )

    card = evaluator.score(extraction, facts_payload())
    verdicts = {s.field: s.verdict for s in card.scores}

    assert verdicts["total_revenue"] is Verdict.EXACT
    # Abstention is only scoreable when an answer key exists to abstain from.
    assert verdicts["net_income"] is Verdict.ABSTAINED
    # Not configured in the payload, so no answer key exists.
    assert verdicts["total_assets"] is Verdict.UNRESOLVABLE


def test_aggregate_counts_verdicts_per_field(evaluator: ExtractionEvaluator) -> None:
    extraction = FilingExtraction(
        fiscal_year=2023, total_revenue=APPLE_REVENUE, accession="0000320193-23-000106"
    )
    cards = [evaluator.score(extraction, facts_payload()) for _ in range(3)]

    agg = ExtractionEvaluator.aggregate(cards)

    assert agg["total_revenue"].counts["exact"] == 3
    assert agg["total_revenue"].accuracy == 1.0
    assert agg["total_assets"].scoreable == 0

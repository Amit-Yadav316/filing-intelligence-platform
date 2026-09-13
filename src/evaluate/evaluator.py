"""Score extractions against XBRL ground truth.

This is the module the whole project exists to make possible. Everything else -
the archive, the chunker, the retrieval arms - is machinery for producing an
extraction that this can mark right or wrong against a fact the SEC published
in the same filing.

The verdict taxonomy is the interesting part
--------------------------------------------
"Accuracy" as a single number hides the thing worth knowing, which is *how* an
extraction fails. Six verdicts, each actionable:

``exact``            the value matches the XBRL fact.
``within_tolerance`` inside 0.5%, which absorbs rounding in the prose.
``scale_error``      the right digits at the wrong magnitude - a clean power of
                     ten out. Financial statements are headed "in millions",
                     and a model that reads 383,285 and reports 383,285 rather
                     than 383,285,000,000 has made a *systematic* error, not a
                     reading error. Separating it says "fix the prompt", where
                     lumping it into `wrong` says only "the model is bad".
``wrong``            a different number of the same magnitude - the wrong line
                     item, or the wrong fiscal year's column.
``hallucinated``     a value with no plausible relationship to the truth and no
                     scale explanation. The failure that actually damages trust.
``abstained``        the model declined. Not a failure, and deliberately not
                     counted as one.
``unresolvable``     no XBRL fact exists to compare against, so nothing can be
                     concluded. Excluded from the accuracy denominator and
                     reported separately, because a denominator quietly reduced
                     by unscoreable cases is how accuracy figures get inflated.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from src.config.settings import Settings, get_settings
from src.evaluate.xbrl_resolver import GroundTruthFact, XBRLResolver
from src.extract.schemas import SCORED_FIELDS, FilingExtraction
from src.observability.logging import get_logger

log = get_logger(__name__)


class Verdict(StrEnum):
    EXACT = "exact"
    WITHIN_TOLERANCE = "within_tolerance"
    SCALE_ERROR = "scale_error"
    WRONG = "wrong"
    HALLUCINATED = "hallucinated"
    ABSTAINED = "abstained"
    UNRESOLVABLE = "unresolvable"


#: Verdicts that count as a correct answer.
CORRECT = frozenset({Verdict.EXACT, Verdict.WITHIN_TOLERANCE})

#: Verdicts that are scoreable at all - the accuracy denominator.
SCOREABLE = frozenset(
    {
        Verdict.EXACT,
        Verdict.WITHIN_TOLERANCE,
        Verdict.SCALE_ERROR,
        Verdict.WRONG,
        Verdict.HALLUCINATED,
        Verdict.ABSTAINED,
    }
)

#: Powers of ten a filer plausibly reports in: thousands, millions, billions.
_SCALE_FACTORS = (Decimal(10) ** n for n in (3, 6, 9))


@dataclass(frozen=True, slots=True)
class FieldScore:
    """The verdict for one field of one filing, with the evidence for it."""

    field: str
    verdict: Verdict
    extracted: Decimal | None
    truth: Decimal | None
    relative_error: float | None
    tag: str | None = None
    scale_factor: int | None = None
    source_chunks: tuple[str, ...] = ()
    note: str = ""
    matched_alternative: bool = False
    """True when the answer matched a different concept the tag map also
    declares as meaning this field - a defensible answer, not the preferred one."""

    @property
    def is_correct(self) -> bool:
        return self.verdict in CORRECT

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "verdict": str(self.verdict),
            "extracted": str(self.extracted) if self.extracted is not None else None,
            "truth": str(self.truth) if self.truth is not None else None,
            "relative_error": self.relative_error,
            "tag": self.tag,
            "scale_factor": self.scale_factor,
            "source_chunks": list(self.source_chunks),
            "note": self.note,
            "matched_alternative": self.matched_alternative,
        }


@dataclass
class ScoreCard:
    """Every field verdict for one filing."""

    accession: str
    cik: str
    company: str | None
    form: str
    fiscal_year: int
    scores: list[FieldScore] = dataclass_field(default_factory=list)
    model: str = ""

    def by_field(self) -> dict[str, FieldScore]:
        return {s.field: s for s in self.scores}

    @property
    def scoreable(self) -> list[FieldScore]:
        return [s for s in self.scores if s.verdict in SCOREABLE]

    @property
    def accuracy(self) -> float:
        pool = self.scoreable
        return sum(1 for s in pool if s.is_correct) / len(pool) if pool else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "accession": self.accession,
            "cik": self.cik,
            "company": self.company,
            "form": self.form,
            "fiscal_year": self.fiscal_year,
            "model": self.model,
            "accuracy": round(self.accuracy, 4),
            "scores": [s.as_dict() for s in self.scores],
        }


@dataclass
class FieldAggregate:
    """Counts across the corpus for one field."""

    field: str
    counts: Counter[str] = dataclass_field(default_factory=Counter)

    def add(self, verdict: Verdict) -> None:
        self.counts[str(verdict)] += 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def scoreable(self) -> int:
        return sum(self.counts[str(v)] for v in SCOREABLE)

    @property
    def correct(self) -> int:
        return sum(self.counts[str(v)] for v in CORRECT)

    @property
    def accuracy(self) -> float:
        """Share of scoreable cases answered correctly.

        Abstentions sit in the denominator on purpose. A model that abstains on
        everything is not 100% accurate; it is 0% useful, and an accuracy figure
        that hid that would be worthless.
        """
        return self.correct / self.scoreable if self.scoreable else 0.0

    @property
    def hallucination_rate(self) -> float:
        return self.counts[str(Verdict.HALLUCINATED)] / self.scoreable if self.scoreable else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.counts[str(Verdict.ABSTAINED)] / self.scoreable if self.scoreable else 0.0

    @property
    def resolvable_rate(self) -> float:
        return self.scoreable / self.total if self.total else 0.0


class ExtractionEvaluator:
    """Scores :class:`FilingExtraction` objects against XBRL companyfacts."""

    def __init__(
        self,
        resolver: XBRLResolver | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.resolver = resolver or XBRLResolver.from_config()
        self.tolerance = Decimal(str(self.settings.extraction_tolerance))

    # --- one field --------------------------------------------------------
    def score_field(
        self,
        field_name: str,
        extracted: Decimal | None,
        truth: GroundTruthFact | None,
        *,
        abstained: bool = False,
        source_chunks: tuple[str, ...] = (),
        alternatives: Sequence[GroundTruthFact] = (),
    ) -> FieldScore:
        if truth is None:
            # Nothing to compare against. Saying "wrong" here would punish the
            # model for a gap in the answer key.
            return FieldScore(
                field=field_name,
                verdict=Verdict.UNRESOLVABLE,
                extracted=extracted,
                truth=None,
                relative_error=None,
                source_chunks=source_chunks,
                note="no XBRL fact resolved for this field and fiscal year",
            )

        if abstained or extracted is None:
            return FieldScore(
                field=field_name,
                verdict=Verdict.ABSTAINED,
                extracted=None,
                truth=truth.value,
                relative_error=None,
                tag=truth.tag,
                source_chunks=source_chunks,
                note="model declined to answer",
            )

        try:
            value = Decimal(extracted)
        except (InvalidOperation, TypeError, ValueError):
            return FieldScore(
                field=field_name,
                verdict=Verdict.HALLUCINATED,
                extracted=None,
                truth=truth.value,
                relative_error=None,
                tag=truth.tag,
                source_chunks=source_chunks,
                note=f"unparseable value {extracted!r}",
            )

        if not value.is_finite():
            return FieldScore(
                field=field_name,
                verdict=Verdict.HALLUCINATED,
                extracted=None,
                truth=truth.value,
                relative_error=None,
                tag=truth.tag,
                source_chunks=source_chunks,
                note=f"non-finite value {extracted!r}",
            )

        actual = truth.value
        if value == actual:
            return FieldScore(
                field_name, Verdict.EXACT, value, actual, 0.0, truth.tag, None, source_chunks
            )

        relative = self._relative_error(value, actual)

        if relative is not None and relative <= float(self.tolerance):
            return FieldScore(
                field_name,
                Verdict.WITHIN_TOLERANCE,
                value,
                actual,
                relative,
                truth.tag,
                None,
                source_chunks,
                f"within {float(self.tolerance):.2%}",
            )

        # Before calling this wrong, check whether the model reported a
        # different concept the tag map already declares as meaning this field.
        # "Net income" is both NetIncomeLoss (attributable to the parent) and
        # ProfitLoss (including non-controlling interests); a model choosing the
        # second has answered correctly under a different, equally defensible
        # reading. Scoring that as an error measures the preference order, not
        # the extractor.
        for candidate in alternatives:
            if candidate.tag == truth.tag:
                continue
            alt_error = self._relative_error(value, candidate.value)
            if alt_error is not None and alt_error <= float(self.tolerance):
                return FieldScore(
                    field=field_name,
                    verdict=Verdict.EXACT if value == candidate.value else Verdict.WITHIN_TOLERANCE,
                    extracted=value,
                    truth=actual,
                    relative_error=relative,
                    tag=candidate.tag,
                    source_chunks=source_chunks,
                    note=(
                        f"matched us-gaap:{candidate.tag}, a concept the tag map also "
                        f"maps to {field_name}; preferred tag was {truth.tag}"
                    ),
                    matched_alternative=True,
                )

        scale = self._scale_factor(value, actual)
        if scale is not None:
            return FieldScore(
                field_name,
                Verdict.SCALE_ERROR,
                value,
                actual,
                relative,
                truth.tag,
                scale,
                source_chunks,
                f"correct digits, off by a factor of {scale:,}",
            )

        # No scale explanation left. Magnitude RATIO decides, not relative
        # error: relative error is asymmetric, so an under-estimate asymptotes
        # to 1.0 no matter how absurd it is - reporting 42 against 383 billion
        # scores 0.9999 and would slip through as merely "wrong". The ratio is
        # symmetric, so an order of magnitude out in either direction is caught.
        ratio = self._magnitude_ratio(value, actual)
        if ratio is None or ratio > 10.0:
            return FieldScore(
                field_name,
                Verdict.HALLUCINATED,
                value,
                actual,
                relative,
                truth.tag,
                None,
                source_chunks,
                "no plausible relationship to the reported fact",
            )

        return FieldScore(
            field_name,
            Verdict.WRONG,
            value,
            actual,
            relative,
            truth.tag,
            None,
            source_chunks,
            "wrong line item or wrong period",
        )

    @staticmethod
    def _relative_error(value: Decimal, actual: Decimal) -> float | None:
        if actual == 0:
            return None if value != 0 else 0.0
        try:
            return float(abs(value - actual) / abs(actual))
        except (InvalidOperation, ZeroDivisionError, OverflowError):
            return None

    @staticmethod
    def _magnitude_ratio(value: Decimal, actual: Decimal) -> float | None:
        """Larger magnitude over smaller, so the measure is symmetric."""
        a, b = abs(value), abs(actual)
        if a == 0 and b == 0:
            return 1.0
        if a == 0 or b == 0:
            return None
        try:
            return float(max(a, b) / min(a, b))
        except (InvalidOperation, ZeroDivisionError, OverflowError):
            return None

    @staticmethod
    def _scale_factor(value: Decimal, actual: Decimal) -> int | None:
        """Detect a clean power-of-ten mismatch in either direction."""
        if value == 0 or actual == 0:
            return None
        for factor in (Decimal(10) ** n for n in (3, 6, 9)):
            for candidate, sign in ((value * factor, 1), (value / factor, -1)):
                if candidate == 0:
                    continue
                try:
                    if abs(candidate - actual) / abs(actual) <= 0.005:
                        return int(factor) * sign
                except (InvalidOperation, ZeroDivisionError, OverflowError):
                    continue
        return None

    # --- one filing -------------------------------------------------------
    def score(
        self,
        extraction: FilingExtraction,
        facts: dict[str, Any],
        *,
        fields: tuple[str, ...] = SCORED_FIELDS,
    ) -> ScoreCard:
        """Score every field of one extraction against one companyfacts payload."""
        truth = self.resolver.resolve(
            facts, extraction.fiscal_year, list(fields), accession=extraction.accession or None
        )
        card = ScoreCard(
            accession=extraction.accession,
            cik=extraction.cik,
            company=extraction.company,
            form=extraction.form,
            fiscal_year=extraction.fiscal_year,
            model=extraction.model,
        )
        for name in fields:
            card.scores.append(
                self.score_field(
                    name,
                    extraction.value_for(name),
                    truth.get(name),
                    abstained=extraction.abstained(name),
                    source_chunks=tuple(extraction.sources_for(name)),
                    alternatives=self.resolver.resolve_candidates(
                        facts, name, extraction.fiscal_year
                    ),
                )
            )
        return card

    # --- the corpus -------------------------------------------------------
    @staticmethod
    def aggregate(cards: list[ScoreCard]) -> dict[str, FieldAggregate]:
        out: dict[str, FieldAggregate] = {}
        for card in cards:
            for score in card.scores:
                out.setdefault(score.field, FieldAggregate(score.field)).add(score.verdict)
        return out

"""Retrieval-grounded, schema-constrained extraction.

Two rules shape this module.

**Never send the whole filing.** A 10-K is 200,000+ characters. Pasting it into
the context would work - the model has a million-token window - and it would
make the retrieval half of the project decorative, cost 20x more per filing, and
remove any ability to say *which chunk* a number came from. So extraction runs
over retrieved chunks, and every field carries the chunk ids that supported it.

**Abstention beats guessing.** The prompt says so explicitly, the schema has a
field for it, and the evaluator scores it in its own bucket. A model that
answers "not found" for operating cash flow when the cash flow statement was not
retrieved is behaving correctly, and the pipeline must not train it out.

On a schema violation the call is retried once; a second failure abstains on
every field rather than inventing a shape that fits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from src.config.settings import Settings, get_settings
from src.extract.llm_client import LLMClient, LLMError, build_llm_client
from src.extract.schemas import GEMINI_RESPONSE_SCHEMA, SCORED_FIELDS, FilingExtraction
from src.ingest.circuit_breaker import CircuitOpenError
from src.observability.logging import get_logger
from src.observability.metrics import STAGE_LATENCY
from src.retrieve.hybrid import HybridRetriever
from src.retrieve.indexes import MetadataFilter, SearchHit

log = get_logger(__name__)

SYSTEM_PROMPT = """\
You extract financial facts from SEC filings. You are precise and you do not guess.

Rules, in order of importance:

1. Report every monetary amount in FULL UNITS. Financial statements are headed
   "in millions" or "in thousands"; a table showing 383,285 under such a heading
   means 383,285,000,000. Read the units heading before reporting any figure.
2. Report the figure for the fiscal year the filing is ABOUT, not a comparative
   prior-year column shown beside it.
3. If a value is not present in the excerpts provided, add the field name to
   abstained_fields and leave it null. Do NOT infer, estimate, or calculate it
   from other figures. An honest "not found" is more valuable than a guess.
4. For every field you do answer, list the chunk ids you took it from in
   field_sources. Only cite chunks you actually used.
5. total_revenue means total revenue or net sales for the whole company.
   operating_cash_flow means net cash provided by operating activities.
   total_assets is the balance-sheet total at the fiscal year end, current
   year column rather than the prior-year comparative.

Prompt note: an earlier, far more prescriptive version of rule 5 - spelling out
that operating revenue excludes non-operating other income - cut accuracy from
62.5% to 18.8% on the same four filings, because the model responded to the
extra constraint by abstaining rather than by answering more precisely. The
terse version is kept for that measured reason.
"""


@dataclass
class ExtractionOutcome:
    accession: str
    extraction: FilingExtraction | None
    context_chunks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    retried: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.extraction is not None


class ExtractionService:
    """Turns one indexed filing into a scored-ready :class:`FilingExtraction`."""

    #: One query per financial statement, rather than one query for everything.
    #: A single broad query is what the first version did, and it failed: asking
    #: for "revenue net income total assets cash flow" retrieves narrative
    #: *about* revenue - revenue-recognition policy, segment commentary - because
    #: that prose is semantically closer to the query than a table of numbers is.
    #: The model then abstained on total_assets for every filing, despite the
    #: balance sheet sitting in the index, correctly parsed, three chunks away.
    #:
    #: Each statement is asked for by its own name, which is how filings label
    #: them, and the table arm is queried separately because the figures live in
    #: tables while the query embeds like prose.
    STATEMENT_QUERIES: tuple[tuple[str, str], ...] = (
        (
            "income_statement",
            "consolidated statements of income operations net sales "
            "total revenues net operating revenues net income",
        ),
        (
            "balance_sheet",
            "consolidated balance sheets total assets total current assets "
            "total liabilities and equity",
        ),
        (
            "cash_flow",
            "consolidated statements of cash flows net cash provided by operating activities",
        ),
    )

    #: Exact line-item phrases, one per scored field, looked up LEXICALLY.
    #: This is the job BM25 is actually good at and dense retrieval is bad at:
    #: "total assets" is a precise string that appears in the balance sheet and
    #: almost nowhere else, whereas its embedding sits near every paragraph that
    #: mentions assets. With statement-level queries alone the model abstained on
    #: total_assets for every filing while the balance sheet sat in the index.
    FIELD_ANCHORS: tuple[tuple[str, str], ...] = (
        ("total_assets", "total assets"),
        ("operating_cash_flow", "net cash provided by operating activities"),
        ("total_revenue", "total net sales total revenues net operating revenues"),
        ("net_income", "net income attributable"),
    )

    #: Narrative context, for figures discussed in MD&A rather than tabulated.
    NARRATIVE_QUERY = "total revenue increased net income for fiscal year results of operations"

    def __init__(
        self,
        retriever: HybridRetriever,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        redis_client: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever
        self._llm = llm
        self._redis = redis_client
        self._redis_ready = redis_client is not None

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = build_llm_client(self.settings)
        return self._llm

    @property
    def redis(self) -> Any:
        if not self._redis_ready:
            try:
                import redis as redis_lib

                self._redis = redis_lib.Redis.from_url(
                    self.settings.redis_url, socket_connect_timeout=2
                )
                self._redis.ping()
            except Exception as exc:
                log.warning("redis_unavailable", error=str(exc), effect="extraction uncached")
                self._redis = None
            self._redis_ready = True
        return self._redis

    # --- caching ----------------------------------------------------------
    def cache_key(self, accession: str) -> str:
        """Keyed on the filing, the schema and the model.

        All three must be in the key: re-running after a schema change or a
        model swap has to re-extract, or the accuracy table reports one model's
        numbers under another's name.
        """
        return (
            f"extract:{accession}:{self.settings.extraction_schema_version}:"
            f"{self.settings.llm_model}"
        )

    def _cached(self, accession: str) -> FilingExtraction | None:
        client = self.redis
        if client is None:
            return None
        try:
            blob = client.get(self.cache_key(accession))
            return FilingExtraction.model_validate_json(blob) if blob else None
        except Exception as exc:
            log.warning("extraction_cache_read_failed", error=str(exc))
            return None

    def _store(self, accession: str, extraction: FilingExtraction) -> None:
        client = self.redis
        if client is None:
            return
        try:
            client.set(
                self.cache_key(accession),
                extraction.model_dump_json(),
                ex=self.settings.redis_cache_ttl_seconds,
            )
        except Exception as exc:
            log.warning("extraction_cache_write_failed", error=str(exc))

    # --- context ----------------------------------------------------------
    def build_context(self, accession: str) -> list[SearchHit]:
        """Retrieve the chunks that actually carry the four figures.

        Tables first and per statement, then narrative to fill the remainder.
        Deduplicated by chunk id, capped by ``extraction_context_chunks``.
        """
        budget = self.settings.extraction_context_chunks
        seen: dict[str, SearchHit] = {}

        def take(query: str, limit: int, chunk_type: str | None) -> None:
            if len(seen) >= budget:
                return
            fused = self.retriever.search(
                query,
                top_k=limit,
                filters=MetadataFilter(accession=accession, chunk_type=chunk_type),
            )
            for f in fused:
                if len(seen) >= budget:
                    return
                seen.setdefault(f.hit.chunk_id, f.hit)

        # Exact line-item phrases first, lexically. These are the highest-value
        # chunks in the whole context and must not be crowded out.
        for _field, phrase in self.FIELD_ANCHORS:
            if len(seen) >= budget:
                break
            for hit in self.retriever.bm25.search(
                phrase,
                k=2,
                filters=MetadataFilter(accession=accession, chunk_type="table"),
            ):
                seen.setdefault(hit.chunk_id, hit)

        # Then the statements as a whole.
        for _name, query in self.STATEMENT_QUERIES:
            take(query, 2, "table")

        # Then the prose around each statement, which carries the units heading
        # and any figure given only in narrative.
        for _name, query in self.STATEMENT_QUERIES:
            take(query, 1, "prose")

        take(self.NARRATIVE_QUERY, 4, "prose")

        # Document order reads more naturally than relevance order, and keeps a
        # statement next to the text that qualifies it.
        return sorted(seen.values(), key=lambda h: h.char_start)

    @staticmethod
    def render_context(hits: list[SearchHit]) -> str:
        blocks = []
        for hit in hits:
            header = (
                f"[chunk_id: {hit.chunk_id}] "
                f"[Item {hit.item_number or '?'}: {hit.section or 'unknown'}] "
                f"[{hit.chunk_type}]"
            )
            blocks.append(f"{header}\n{hit.text}")
        return "\n\n---\n\n".join(blocks)

    def build_prompt(self, hits: list[SearchHit], fiscal_year: int, company: str | None) -> str:
        return (
            f"Company: {company or 'unknown'}\n"
            f"Fiscal year to extract: {fiscal_year}\n\n"
            "Excerpts retrieved from the filing follow. Each is labelled with its "
            "chunk_id, which you must cite in field_sources.\n\n"
            f"{self.render_context(hits)}\n\n"
            f"Extract the financial facts for fiscal year {fiscal_year}. "
            "Report all amounts in full units. Abstain on anything not present above."
        )

    # --- extraction -------------------------------------------------------
    def extract(
        self,
        accession: str,
        *,
        cik: str,
        form: str,
        fiscal_year: int,
        company: str | None = None,
        use_cache: bool = True,
    ) -> ExtractionOutcome:
        if use_cache:
            cached = self._cached(accession)
            if cached is not None:
                log.info("extraction_cache_hit", accession=accession)
                return ExtractionOutcome(accession, cached, cached=True)

        hits = self.build_context(accession)
        if not hits:
            return ExtractionOutcome(accession, None, error="no chunks retrieved for this filing")

        prompt = self.build_prompt(hits, fiscal_year, company)
        outcome = ExtractionOutcome(accession, None, context_chunks=len(hits))

        for attempt in (1, 2):
            try:
                with STAGE_LATENCY.labels(stage="extract").time():
                    response = self.llm.generate(
                        prompt, schema=GEMINI_RESPONSE_SCHEMA, system=SYSTEM_PROMPT
                    )
            except (LLMError, CircuitOpenError) as exc:
                # Includes a spent quota and an open breaker. Both end this
                # filing cleanly rather than killing the whole run.
                outcome.error = str(exc)
                log.warning("extraction_llm_failed", accession=accession, error=str(exc))
                return outcome

            outcome.input_tokens += response.input_tokens
            outcome.output_tokens += response.output_tokens
            outcome.cost_usd += response.cost_usd

            try:
                payload = response.parsed()
                extraction = self._to_extraction(
                    payload,
                    accession=accession,
                    cik=cik,
                    form=form,
                    company=company,
                    hits=hits,
                    fiscal_year=fiscal_year,
                )
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                if attempt == 1:
                    outcome.retried = True
                    log.warning("extraction_schema_violation", accession=accession, error=str(exc))
                    continue
                # Second failure: abstain on everything rather than invent a
                # shape that happens to validate.
                log.warning("extraction_abstaining", accession=accession, error=str(exc))
                outcome.extraction = self._abstain_all(
                    accession, cik, form, company, fiscal_year, hits
                )
                outcome.error = f"schema violation, abstained: {exc}"
                return outcome

            outcome.extraction = extraction
            self._store(accession, extraction)
            log.info(
                "extraction_complete",
                accession=accession,
                abstained=len(extraction.abstained_fields),
                confidence=extraction.confidence,
                cost_usd=round(outcome.cost_usd, 6),
            )
            return outcome

        return outcome

    def _to_extraction(
        self,
        payload: dict[str, Any],
        *,
        accession: str,
        cik: str,
        form: str,
        company: str | None,
        hits: list[SearchHit],
        fiscal_year: int,
    ) -> FilingExtraction:
        values: dict[str, Any] = {
            "fiscal_year": int(payload.get("fiscal_year") or fiscal_year),
            "reported_currency": payload.get("reported_currency") or "USD",
            "top_risk_categories": payload.get("top_risk_categories") or [],
            "abstained_fields": payload.get("abstained_fields") or [],
            "confidence": float(payload.get("confidence") or 0.0),
            "field_sources": payload.get("field_sources") or {},
            "source_chunks": [h.chunk_id for h in hits],
            "accession": accession,
            "cik": cik,
            "form": form,
            "company": company,
            "model": self.settings.llm_model,
            "schema_version": self.settings.extraction_schema_version,
        }
        for name in SCORED_FIELDS:
            values[name] = self._decimal(payload.get(name))
        return FilingExtraction(**values)

    @staticmethod
    def _decimal(raw: Any) -> Decimal | None:
        if raw is None or raw == "":
            return None
        try:
            value = Decimal(str(raw))
        except (InvalidOperation, TypeError, ValueError):
            return None
        return value if value.is_finite() else None

    def _abstain_all(
        self,
        accession: str,
        cik: str,
        form: str,
        company: str | None,
        fiscal_year: int,
        hits: list[SearchHit],
    ) -> FilingExtraction:
        return FilingExtraction(
            fiscal_year=fiscal_year,
            abstained_fields=list(SCORED_FIELDS),
            source_chunks=[h.chunk_id for h in hits],
            confidence=0.0,
            accession=accession,
            cik=cik,
            form=form,
            company=company,
            model=self.settings.llm_model,
            schema_version=self.settings.extraction_schema_version,
        )

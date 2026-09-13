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

import hashlib
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
3. The excerpts below normally DO contain the financial statements. Search all of
   them - income statement, balance sheet, cash flow statement, statements of
   equity, and MD&A tables - before concluding a figure is missing. Reading a
   number printed in any excerpt is NOT inference; it is the task.
4. Abstain ONLY when a figure genuinely does not appear anywhere in the excerpts.
   Then add the field name to abstained_fields and leave it null. Do not abstain
   merely because two similar lines could both match - report the one that most
   directly answers the field and cite its chunk. Never compute a figure by
   arithmetic on other figures.
5. total_revenue means total revenue or net sales for the whole company.
   operating_cash_flow means net cash provided by operating activities.
   total_assets is the balance-sheet total at fiscal year end, current-year
   column rather than the prior-year comparative.

Prompt note: two earlier versions were measured and rejected. Adding explicit
definitional guidance - spelling out that operating revenue excludes
non-operating other income - cut accuracy from 62.5% to 18.8%, because the model
answered the extra constraint by abstaining. A version asserting that "an honest
not-found is more valuable than a guess" produced a 30% abstention rate, and a
no-LLM diagnostic showed the figure was present in the context for 67% of those
abstentions: the model had the number and declined anyway. Rules 3 and 4 are the
correction - they narrow abstention to genuinely absent figures without adding
definitional constraints.
"""


def estimate_tokens(text: str) -> int:
    """Approximate token count for a rendered context.

    Calibrated against a provider's own count rather than assumed. The usual
    characters-over-four rule underestimated a real extraction prompt by about
    1.6x - Groq counted 10,087 tokens where that rule predicted 6,339 - because
    serialised financial tables are dense with digits, separators and pipes,
    all of which tokenize far more finely than English prose.

    Overestimating is the safe direction: it costs a little context, where
    underestimating costs the whole request with a 413.
    """
    return int(len(text) / 2.5)


def is_index_table(text: str) -> bool:
    """True for a table of contents dressed as a financial statement.

    "Index to Consolidated Financial Statements" lists every statement by name
    and holds no figures, so it is the single best lexical match for a query
    naming a statement - and the single most useless chunk to spend context on.
    Apple's extraction context contained it twice while the actual income
    statement, balance sheet and cash flow statement went unretrieved.
    """
    head = text[:400].lower()
    if "index to" in head and "page" in head:
        return True
    # A table of statement names with almost no digits is a contents listing.
    digits = sum(c.isdigit() for c in text)
    return "consolidated financial statements" in head and digits < len(text) * 0.02


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

    #: Line-item phrases per scored field, looked up LEXICALLY with OR.
    #:
    #: Two things make this the shape it is, both measured rather than guessed.
    #:
    #: First, ``websearch_to_tsquery`` ANDs bare terms, so a long anchor like
    #: "net cash provided by operating activities" matches nothing when Apple
    #: writes "Cash generated by operating activities". Three of the four
    #: original anchors returned ZERO rows for Apple while the statements sat in
    #: the index - which is why Apple abstained on three of four fields. Short
    #: quoted phrases joined by OR survive the wording differences between
    #: filers; that is the whole reason the lexical arm exists.
    #:
    #: Second, these phrases are line ITEMS, not statement titles. Querying by
    #: title retrieves the "Index to Consolidated Financial Statements" table -
    #: which lists every statement name and contains no figures - because the
    #: statements themselves carry no title inside the <table> element.
    FIELD_ANCHORS: tuple[tuple[str, str], ...] = (
        ("total_assets", '"total assets"'),
        ("operating_cash_flow", '"operating activities"'),
        ("total_revenue", '"net sales" OR "total revenues" OR "total revenue"'),
        ("net_income", '"net income" OR "net earnings"'),
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
    def cache_key(self, accession: str, prompt: str = "") -> str:
        """Keyed on everything that changes what the model is asked.

        This has been wrong twice, in the same way each time, so it is now keyed
        on the full request rather than on a proxy for it.

        The first version keyed on filing, schema version and model. Editing the
        system prompt then changed nothing on a re-run - the cache happily
        served answers from the previous prompt, which would have made an A/B
        test compare a prompt against itself.

        Adding a hash of the system prompt fixed that and missed the other half:
        the RETRIEVED CONTEXT is also part of the request, and changing the
        context budget from 18 chunks to a 6,000-token cap produced cache hits
        from a configuration that no longer existed.

        Hashing the assembled prompt covers both, and anything similar in
        future. Building the context to compute it costs a retrieval and no LLM
        call, which is the cheap half of the work.
        """
        fingerprint = hashlib.sha256((SYSTEM_PROMPT + prompt).encode("utf-8")).hexdigest()[:16]
        return (
            f"extract:{accession}:{self.settings.extraction_schema_version}:"
            f"{self.settings.llm_model}:{fingerprint}"
        )

    def _cached(self, accession: str, prompt: str = "") -> FilingExtraction | None:
        client = self.redis
        if client is None:
            return None
        try:
            blob = client.get(self.cache_key(accession, prompt))
            return FilingExtraction.model_validate_json(blob) if blob else None
        except Exception as exc:
            log.warning("extraction_cache_read_failed", error=str(exc))
            return None

    def _store(self, accession: str, extraction: FilingExtraction, prompt: str = "") -> None:
        client = self.redis
        if client is None:
            return
        try:
            client.set(
                self.cache_key(accession, prompt),
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
        token_cap = self.settings.extraction_context_max_tokens
        seen: dict[str, SearchHit] = {}

        def would_exceed(hit: SearchHit) -> bool:
            """True when adding this chunk would break the token ceiling."""
            if not token_cap:
                return False
            used = sum(estimate_tokens(h.text) for h in seen.values())
            return used + estimate_tokens(hit.text) > token_cap

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
                if is_index_table(f.hit.text) or would_exceed(f.hit):
                    continue
                seen.setdefault(f.hit.chunk_id, f.hit)

        # Exact line-item phrases first, lexically. These are the highest-value
        # chunks in the whole context and must not be crowded out.
        for _field, phrase in self.FIELD_ANCHORS:
            if len(seen) >= budget:
                break
            for hit in self.retriever.bm25.search(
                phrase,
                k=3,
                filters=MetadataFilter(accession=accession, chunk_type="table"),
            ):
                if is_index_table(hit.text):
                    continue
                # Anchors are the highest-value chunks, so they are admitted
                # before anything else and only the cap can turn one away.
                if would_exceed(hit):
                    continue
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
        # Context first, then the cache: the prompt is part of the key, so it
        # has to exist before the key can be computed. Retrieval is cheap and
        # involves no LLM call, which is what makes that ordering affordable.
        hits = self.build_context(accession)
        if not hits:
            return ExtractionOutcome(accession, None, error="no chunks retrieved for this filing")

        prompt = self.build_prompt(hits, fiscal_year, company)

        if use_cache:
            cached = self._cached(accession, prompt)
            if cached is not None:
                log.info("extraction_cache_hit", accession=accession)
                return ExtractionOutcome(accession, cached, cached=True)

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
            self._store(accession, extraction, prompt)
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

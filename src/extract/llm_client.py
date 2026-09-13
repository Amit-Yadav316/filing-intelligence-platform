"""Provider-agnostic LLM client with schema-constrained output.

The provider is a configuration choice, not an architectural one. Extraction
depends on a narrow contract - send a prompt and a JSON schema, get back parsed
JSON plus a token count - so swapping Gemini for Claude or GPT is a new subclass
rather than a change to the extraction logic.

Every call goes through retry, timeout and a circuit breaker, for the same
reason the EDGAR client does: a provider returning 429 under load should
degrade into a bounded, visible failure rather than a retry storm.

Cost is counted, not estimated after the fact. Token counts come from the
provider's own usage metadata and are multiplied by configured per-million
prices, so ``llm_cost_usd_total`` is a real series on the dashboard from the
first run.
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import SecretStr

from src.config.settings import Settings, get_settings
from src.ingest.circuit_breaker import CircuitBreaker
from src.observability.logging import get_logger
from src.observability.metrics import LLM_COST_USD, LLM_TOKENS, STAGE_LATENCY

log = get_logger(__name__)

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """A generation request failed permanently."""


class NonRetryableError(LLMError):
    """The request itself is wrong, so sending it again cannot help.

    A malformed request, an unknown model, a rejected schema: every one returns
    a 4xx that is identical on every attempt. The generic retry path treated
    these like a transient failure and sent the same doomed request four times,
    which on a quota-limited free tier spends a fifth of the daily budget to
    learn nothing.
    """


class QuotaExhaustedError(LLMError):
    """The provider's quota is spent, and waiting will not help.

    Separated from a transient rate limit because the two need opposite
    responses. A per-minute limit should be waited out; a per-DAY limit
    must abort immediately, since every retry spends another unit of the
    very quota that has run out. With a free-tier cap of 20 requests per
    day, a 4-attempt retry turns one filing into a fifth of the budget.
    """

    def __init__(
        self, message: str, *, quota_id: str = "", retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.quota_id = quota_id
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One completion, with what it cost."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float

    def parsed(self) -> dict[str, Any]:
        """Parse the JSON body, tolerating a fenced code block.

        Structured-output mode normally returns bare JSON, but a model that
        falls back to prose under load wraps it in ```json fences, and failing
        the whole extraction over three backticks would be careless.
        """
        body = self.text.strip()
        if body.startswith("```"):
            body = body.split("\n", 1)[-1]
            body = body.rsplit("```", 1)[0]
        return dict(json.loads(body))


class LLMClient(ABC):
    """The contract extraction depends on."""

    def __init__(self, settings: Settings | None = None, *, sleep: Any = time.sleep) -> None:
        self.settings = settings or get_settings()
        self._sleep = sleep
        self._breaker = CircuitBreaker(
            fail_threshold=5, reset_seconds=60.0, name=f"llm:{self.settings.llm_provider}"
        )

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def _post(self, prompt: str, schema: dict[str, Any], system: str | None) -> LLMResponse: ...

    def generate(
        self, prompt: str, *, schema: dict[str, Any], system: str | None = None
    ) -> LLMResponse:
        """Generate with retries. Raises :class:`LLMError` after giving up."""
        last = "no attempt made"
        for attempt in range(self.settings.llm_max_retries + 1):
            self._breaker.before_call()
            try:
                with STAGE_LATENCY.labels(stage="llm").time():
                    response = self._post(prompt, schema, system)
            except (QuotaExhaustedError, NonRetryableError):
                # Neither is worth a retry, and neither means the provider is
                # unhealthy: one is our budget, the other is our request. The
                # breaker stays closed so a bad request cannot open it for
                # every other caller.
                raise
            except LLMError as exc:
                self._breaker.record_failure()
                last = str(exc)
                if attempt < self.settings.llm_max_retries:
                    delay = random.uniform(0, min(2**attempt, 30))
                    log.warning(
                        "llm_retry", attempt=attempt + 1, reason=last, sleeping=round(delay, 1)
                    )
                    self._sleep(delay)
                continue
            else:
                self._breaker.record_success()
                self._record(response)
                return response
        raise LLMError(
            f"LLM request failed after {self.settings.llm_max_retries + 1} attempts: {last}"
        )

    def _record(self, response: LLMResponse) -> None:
        LLM_TOKENS.labels(model=response.model, direction="input").inc(response.input_tokens)
        LLM_TOKENS.labels(model=response.model, direction="output").inc(response.output_tokens)
        LLM_COST_USD.labels(model=response.model).inc(response.cost_usd)

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.settings.llm_input_cost_per_mtok
            + output_tokens * self.settings.llm_output_cost_per_mtok
        ) / 1_000_000


class GeminiClient(LLMClient):
    """Google Gemini via the generativelanguage REST API.

    REST rather than the SDK deliberately: the dependency is one already in the
    tree (httpx), the auth is a single header, and the structured-output
    contract is visible in the code instead of behind a client abstraction.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        super().__init__(settings, sleep=sleep)
        if self.settings.gemini_api_key is None:
            raise LLMError("GEMINI_API_KEY is not set")
        self._api_key = self.settings.gemini_api_key.get_secret_value()
        self._owns = client is None
        self._http = client or httpx.Client(timeout=self.settings.llm_timeout_seconds)

    @property
    def model(self) -> str:
        return self.settings.llm_model

    def close(self) -> None:
        if self._owns:
            self._http.close()

    @staticmethod
    def _classify_429(response: httpx.Response) -> LLMError:
        """Tell a spent daily quota apart from a momentary rate limit."""
        try:
            error = response.json().get("error", {})
        except ValueError:
            return LLMError(f"HTTP 429: {response.text[:200]}")

        quota_ids: list[str] = []
        retry_after: float | None = None
        for detail in error.get("details", []):
            kind = detail.get("@type", "")
            if "QuotaFailure" in kind:
                quota_ids += [v.get("quotaId", "") for v in detail.get("violations", [])]
            elif "RetryInfo" in kind:
                raw = str(detail.get("retryDelay", "")).rstrip("s")
                retry_after = float(raw) if raw.replace(".", "", 1).isdigit() else None

        if any("PerDay" in q for q in quota_ids):
            return QuotaExhaustedError(
                f"daily quota exhausted ({', '.join(quota_ids)})",
                quota_id=";".join(quota_ids),
                retry_after=retry_after,
            )
        return LLMError(f"HTTP 429 rate limited (retry in {retry_after or '?'}s)")

    def _post(self, prompt: str, schema: dict[str, Any], system: str | None) -> LLMResponse:
        url = f"{self.settings.gemini_api_base}/models/{self.model}:generateContent"
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.settings.llm_temperature,
                "maxOutputTokens": self.settings.llm_max_tokens,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        try:
            response = self._http.post(
                url,
                json=payload,
                headers={
                    "x-goog-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"transport error: {exc}") from exc

        if response.status_code == 429:
            raise self._classify_429(response)
        if response.status_code in RETRYABLE_STATUS:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
        if not response.is_success:
            raise NonRetryableError(
                f"HTTP {response.status_code} (not retryable): {response.text[:300]}"
            )

        body = response.json()
        candidates = body.get("candidates") or []
        if not candidates:
            raise LLMError(f"no candidates returned: {json.dumps(body)[:300]}")

        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text.strip():
            # A truncated or filtered response is empty rather than malformed,
            # and must not be mistaken for an abstention by the model.
            reason = candidates[0].get("finishReason", "unknown")
            raise LLMError(f"empty completion (finishReason={reason})")

        usage = body.get("usageMetadata", {})
        input_tokens = int(usage.get("promptTokenCount", 0))
        output_tokens = int(usage.get("candidatesTokenCount", 0))
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            cost_usd=self._cost(input_tokens, output_tokens),
        )


class OpenAICompatibleClient(LLMClient):
    """Any provider speaking the OpenAI chat-completions format.

    Groq, OpenRouter, Cerebras, Together and a local vLLM server are the same
    wire protocol with a different base URL, so they are one class rather than
    five. The differences that do matter - whether the provider enforces a JSON
    schema or merely guarantees valid JSON - are configuration.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        base_url: str,
        api_key: str,
        model: str,
        provider: str = "openai-compatible",
        client: httpx.Client | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        super().__init__(settings, sleep=sleep)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._provider = provider
        self._owns = client is None
        self._http = client or httpx.Client(timeout=self.settings.llm_timeout_seconds)

    @property
    def model(self) -> str:
        return self._model

    def close(self) -> None:
        if self._owns:
            self._http.close()

    def _response_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        if self.settings.llm_structured_mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "filing_extraction",
                    "strict": True,
                    "schema": schema,
                },
            }
        return {"type": "json_object"}

    def _system_text(self, system: str | None, schema: dict[str, Any]) -> str:
        """In json_object mode the schema is not enforced, so it is described.

        Without this the model returns valid JSON of a shape it invented, which
        parses cleanly and then fails validation - the most annoying possible
        failure, because nothing errors until pydantic rejects it.
        """
        base = system or ""
        if self.settings.llm_structured_mode == "json_schema":
            return base
        return (
            base
            + "\n\nRespond with a single JSON object matching exactly this schema. "
            + "Include every required key, using null where you abstain.\n"
            + json.dumps(schema, separators=(",", ":"))
        )

    def _post(self, prompt: str, schema: dict[str, Any], system: str | None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "response_format": self._response_format(schema),
            "messages": [
                {"role": "system", "content": self._system_text(system, schema)},
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = self._http.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"transport error: {exc}") from exc

        if response.status_code == 429:
            raise self._classify_openai_429(response)
        if response.status_code in RETRYABLE_STATUS:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
        if not response.is_success:
            raise NonRetryableError(
                f"HTTP {response.status_code} (not retryable): {response.text[:300]}"
            )

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"no choices returned: {json.dumps(body)[:300]}")

        text = (choices[0].get("message") or {}).get("content") or ""
        if not text.strip():
            reason = choices[0].get("finish_reason", "unknown")
            raise LLMError(f"empty completion (finish_reason={reason})")

        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
            cost_usd=self._cost(input_tokens, output_tokens),
        )

    @staticmethod
    def _classify_openai_429(response: httpx.Response) -> LLMError:
        """Daily quota versus momentary rate limit, as with Gemini.

        The two need opposite responses: a per-minute limit should be waited
        out, a per-day limit must abort, because every retry spends the budget
        that has already run out.
        """
        body = response.text[:400].lower()
        retry_after = response.headers.get("retry-after")
        daily_markers = ("per day", "daily limit", "quota exceeded", "rpd")
        if any(marker in body for marker in daily_markers):
            return QuotaExhaustedError(
                f"daily quota exhausted: {response.text[:200]}",
                retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
            )
        return LLMError(f"HTTP 429 rate limited (retry-after={retry_after}): {response.text[:160]}")


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    """Construct the client the configuration asks for."""
    settings = settings or get_settings()
    provider = settings.llm_provider

    if provider == "gemini":
        return GeminiClient(settings)

    # Every remaining provider speaks the OpenAI wire format.
    endpoints: dict[str, tuple[str, SecretStr | None]] = {
        "groq": (settings.groq_api_base, settings.groq_api_key),
        "openrouter": (settings.openrouter_api_base, settings.openrouter_api_key),
        "cerebras": (settings.cerebras_api_base, settings.cerebras_api_key),
        "openai": (settings.openai_api_base, settings.openai_api_key),
    }
    if provider not in endpoints:
        raise LLMError(f"no client implemented for provider {provider!r}")

    base_url, key = endpoints[provider]
    if key is None:
        raise LLMError(f"{provider.upper()}_API_KEY is not set but LLM_PROVIDER={provider}")
    return OpenAICompatibleClient(
        settings,
        base_url=base_url,
        api_key=key.get_secret_value(),
        model=settings.llm_model,
        provider=provider,
    )

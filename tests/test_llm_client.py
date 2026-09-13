"""LLM client tests.

Transport is mocked with respx, so these run offline and deterministically.
What is tested is the wire contract and the failure classification - the two
things that differ between providers and that silently corrupt a run when wrong.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.config.settings import Settings
from src.extract.llm_client import (
    GeminiClient,
    LLMError,
    OpenAICompatibleClient,
    QuotaExhaustedError,
    build_llm_client,
)
from src.extract.schemas import EXTRACTION_JSON_SCHEMA, GEMINI_RESPONSE_SCHEMA

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PAYLOAD = '{"fiscal_year": 2023, "total_revenue": 1, "net_income": null}'


def groq_settings(**overrides) -> Settings:
    base = {
        "_env_file": None,
        "edgar_user_agent": "Test Harness test@example.org",
        "llm_provider": "groq",
        "groq_api_key": "test-key",
        "llm_model": "test-model",
        "llm_max_retries": 1,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def ok_body(text: str = PAYLOAD) -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 45},
    }


def client(settings: Settings) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        settings,
        base_url=settings.groq_api_base,
        api_key="test-key",
        model="test-model",
        provider="groq",
        sleep=lambda _s: None,
    )


# --- schema translation ---------------------------------------------------
def test_gemini_schema_is_derived_from_the_canonical_one() -> None:
    """Two hand-maintained schemas would drift the moment a field is added, and
    the symptom would be one provider silently omitting it."""
    assert GEMINI_RESPONSE_SCHEMA["type"] == "OBJECT"
    assert set(GEMINI_RESPONSE_SCHEMA["required"]) == set(EXTRACTION_JSON_SCHEMA["required"])


def test_nullable_union_becomes_a_nullable_flag() -> None:
    """Gemini rejects union types; it wants `nullable` instead."""
    revenue = GEMINI_RESPONSE_SCHEMA["properties"]["total_revenue"]

    assert revenue["type"] == "NUMBER"
    assert revenue["nullable"] is True


def test_additional_properties_is_dropped_for_gemini() -> None:
    """Not part of the accepted dialect; sending it is a 400."""
    assert "additionalProperties" not in GEMINI_RESPONSE_SCHEMA


# --- the OpenAI wire format ----------------------------------------------
@respx.mock
def test_sends_bearer_auth_and_chat_messages() -> None:
    route = respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=ok_body()))

    client(groq_settings()).generate("prompt", schema=EXTRACTION_JSON_SCHEMA, system="sys")

    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer test-key"
    body = request.read().decode()
    assert '"role":"system"' in body.replace(" ", "")
    assert '"role":"user"' in body.replace(" ", "")


@respx.mock
def test_json_object_mode_describes_the_schema_in_the_prompt() -> None:
    """Without enforcement the model returns valid JSON of a shape it invented,
    which parses cleanly and only fails later at pydantic validation."""
    route = respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=ok_body()))

    client(groq_settings(llm_structured_mode="json_object")).generate(
        "prompt", schema=EXTRACTION_JSON_SCHEMA, system="sys"
    )

    body = route.calls[0].request.read().decode()
    assert '"type": "json_object"' in body or '"type":"json_object"' in body
    assert "fiscal_year" in body, "schema was not described to the model"


@respx.mock
def test_json_schema_mode_sends_the_schema_for_enforcement() -> None:
    route = respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=ok_body()))

    client(groq_settings(llm_structured_mode="json_schema")).generate(
        "prompt", schema=EXTRACTION_JSON_SCHEMA, system="sys"
    )

    body = route.calls[0].request.read().decode()
    assert "json_schema" in body
    assert "filing_extraction" in body


@respx.mock
def test_usage_and_cost_come_from_the_provider() -> None:
    respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=ok_body()))

    response = client(groq_settings()).generate("p", schema=EXTRACTION_JSON_SCHEMA)

    assert response.input_tokens == 120
    assert response.output_tokens == 45
    assert response.cost_usd > 0
    assert response.parsed()["fiscal_year"] == 2023


# --- failure classification ----------------------------------------------
@respx.mock
def test_daily_quota_is_not_retried() -> None:
    """Every retry spends the budget that has already run out. Gemini's generic
    retry path burned four units of a twenty-per-day allowance on one request."""
    route = respx.post(GROQ_URL).mock(
        return_value=httpx.Response(429, text='{"error":{"message":"Rate limit reached per day"}}')
    )

    with pytest.raises(QuotaExhaustedError):
        client(groq_settings()).generate("p", schema=EXTRACTION_JSON_SCHEMA)

    assert route.call_count == 1, "a daily quota error must not be retried"


@respx.mock
def test_per_minute_rate_limit_is_retried() -> None:
    route = respx.post(GROQ_URL).mock(
        side_effect=[
            httpx.Response(429, text='{"error":{"message":"rate limit, try again in 2s"}}'),
            httpx.Response(200, json=ok_body()),
        ]
    )

    client(groq_settings()).generate("p", schema=EXTRACTION_JSON_SCHEMA)

    assert route.call_count == 2


@respx.mock
def test_an_empty_completion_is_an_error_not_an_abstention() -> None:
    """A truncated or filtered response must not be read as the model declining."""
    respx.post(GROQ_URL).mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        )
    )

    with pytest.raises(LLMError, match="empty completion"):
        client(groq_settings()).generate("p", schema=EXTRACTION_JSON_SCHEMA)


@respx.mock
def test_a_bad_request_is_not_retried() -> None:
    route = respx.post(GROQ_URL).mock(return_value=httpx.Response(400, text="bad model"))

    with pytest.raises(LLMError, match="not retryable"):
        client(groq_settings()).generate("p", schema=EXTRACTION_JSON_SCHEMA)

    assert route.call_count == 1


# --- provider selection ---------------------------------------------------
def test_builder_returns_an_openai_compatible_client_for_groq() -> None:
    built = build_llm_client(groq_settings())

    assert isinstance(built, OpenAICompatibleClient)
    assert built.model == "test-model"


def test_builder_returns_the_gemini_client_for_gemini() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        edgar_user_agent="Test Harness test@example.org",
        llm_provider="gemini",
        gemini_api_key="k",
    )

    assert isinstance(build_llm_client(settings), GeminiClient)


def test_a_missing_key_is_a_clear_error_not_a_401_later() -> None:
    settings = groq_settings(groq_api_key=None)

    with pytest.raises(LLMError, match="GROQ_API_KEY is not set"):
        build_llm_client(settings)

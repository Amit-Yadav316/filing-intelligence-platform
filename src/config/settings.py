"""Central configuration.

Every tunable in the platform lives here. No module may hardcode a URL, a model
name, a threshold or a credential; if a value could reasonably differ between a
laptop, CI and a deployment, it is a field on ``Settings``.

Values resolve in precedence order: process environment, then ``.env``, then the
default declared below. ``.env`` is gitignored and never committed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the Filing Intelligence Platform."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- EDGAR ------------------------------------------------------------
    edgar_user_agent: str = Field(
        ...,
        description=(
            "Sent on every SEC request. The SEC returns HTTP 403 without a "
            "descriptive User-Agent carrying a reachable contact address. "
            "Format: 'Name email@domain'."
        ),
    )
    edgar_archives_base: str = "https://www.sec.gov/Archives"
    edgar_daily_index_base: str = "https://www.sec.gov/Archives/edgar/daily-index"
    edgar_data_api_base: str = "https://data.sec.gov/api/xbrl"
    edgar_submissions_base: str = "https://data.sec.gov/submissions"
    edgar_company_tickers_url: str = "https://www.sec.gov/files/company_tickers.json"

    # The SEC publishes a 10 req/s ceiling. We sit below it on purpose: the
    # headroom absorbs clock skew and concurrent Airflow workers.
    edgar_rate_limit_per_sec: float = Field(default=8.0, gt=0, le=10.0)
    edgar_timeout_seconds: float = Field(default=30.0, gt=0)
    edgar_max_retries: int = Field(default=4, ge=0)
    edgar_backoff_base_seconds: float = Field(default=1.0, gt=0)
    edgar_backoff_max_seconds: float = Field(default=30.0, gt=0)
    # A server-sent Retry-After is obeyed as given, bounded only by this
    # much larger ceiling. Clamping it to the jitter ceiling would mean
    # retrying sooner than EDGAR asked, which is how an IP gets blocked.
    edgar_retry_after_max_seconds: float = Field(default=300.0, gt=0)

    # Circuit breaker: trip after N consecutive failures, stay open for M seconds.
    edgar_breaker_fail_threshold: int = Field(default=5, ge=1)
    edgar_breaker_reset_seconds: float = Field(default=60.0, gt=0)

    # --- Corpus scope -----------------------------------------------------
    universe_path: Path = REPO_ROOT / "config" / "universe.json"
    xbrl_tag_map_path: Path = REPO_ROOT / "config" / "xbrl_tag_map.yaml"
    target_forms: tuple[str, ...] = ("10-K", "10-Q")
    fiscal_years: tuple[int, ...] = (2022, 2023, 2024)

    # --- Chunking ---------------------------------------------------------
    chunk_token_window: int = Field(default=512, gt=0)
    chunk_token_overlap: int = Field(default=64, ge=0)
    chunk_min_tokens: int = Field(default=32, gt=0)

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=64, gt=0)

    # --- Retrieval --------------------------------------------------------
    rrf_k: int = Field(default=60, gt=0, description="Reciprocal Rank Fusion constant.")
    retrieval_top_k: int = Field(default=10, gt=0)
    retrieval_candidate_k: int = Field(default=50, gt=0, description="Per-arm depth before fusion.")

    # --- Extraction -------------------------------------------------------
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5"
    llm_max_tokens: int = Field(default=4096, gt=0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_schema_version: str = "v1"

    # --- Evaluation -------------------------------------------------------
    # A value within this relative band of the XBRL fact counts as a match.
    # 0.005 == 0.5%, the figure quoted in the README's SLO table.
    extraction_tolerance: float = Field(default=0.005, gt=0, lt=1)

    # --- SLOs -------------------------------------------------------------
    slo_revenue_accuracy_floor: float = Field(default=0.90, gt=0, le=1)
    slo_index_freshness_seconds: int = Field(default=86_400, gt=0)
    slo_search_p95_seconds: float = Field(default=0.8, gt=0)

    # --- Object store -----------------------------------------------------
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: SecretStr = SecretStr("minioadmin")
    minio_secret_key: SecretStr = SecretStr("minioadmin")
    minio_bucket: str = "filings"

    # --- Datastores -------------------------------------------------------
    postgres_dsn: str = "postgresql://filing:filing@localhost:5432/filings"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "filings"
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl_seconds: int = Field(default=60 * 60 * 24 * 30, gt=0)

    # --- Observability ----------------------------------------------------
    prometheus_pushgateway: str = "http://localhost:9091"
    log_level: str = "INFO"
    metrics_namespace: str = "filing_intel"

    # --- Local paths ------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    sample_dir: Path = REPO_ROOT / "data" / "sample"
    docs_dir: Path = REPO_ROOT / "docs"

    @field_validator("edgar_user_agent")
    @classmethod
    def _user_agent_must_carry_contact(cls, v: str) -> str:
        """Fail fast at construction rather than on a 403 twenty minutes in."""
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(
                "EDGAR_USER_AGENT must contain a reachable contact email, "
                "e.g. 'Jane Doe jane@example.com'. The SEC rejects requests without one."
            )
        if "example.com" in v:
            raise ValueError(
                "EDGAR_USER_AGENT still holds the .env.example placeholder. "
                "Set a real contact address before calling the SEC."
            )
        return v

    @field_validator("chunk_token_overlap")
    @classmethod
    def _overlap_below_window(cls, v: int, info: object) -> int:
        # Guards the classic off-by-one that makes the chunker loop forever.
        window = getattr(info, "data", {}).get("chunk_token_window")
        if window is not None and v >= window:
            raise ValueError("chunk_token_overlap must be smaller than chunk_token_window")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Cached so ``.env`` is read once."""
    return Settings()  # type: ignore[call-arg]

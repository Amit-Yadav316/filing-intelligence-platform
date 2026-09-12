"""Shared test fixtures.

Tests never read the developer's real ``.env`` and never touch the live SEC
API unless explicitly marked ``@pytest.mark.network``. Settings are built
in-process so a missing or differently-configured ``.env`` cannot change a
test outcome.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.config.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings: fast retries, tight breaker, no real .env."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        edgar_user_agent="Test Harness test@example.org",
        edgar_rate_limit_per_sec=10.0,
        edgar_max_retries=2,
        edgar_backoff_base_seconds=0.001,
        edgar_backoff_max_seconds=0.002,
        edgar_breaker_fail_threshold=3,
        edgar_breaker_reset_seconds=5.0,
    )


@pytest.fixture
def no_sleep() -> Callable[[float], None]:
    """Drop-in for ``time.sleep`` so retry tests run instantly."""

    def _sleep(_seconds: float) -> None:
        return None

    return _sleep


class FakeClock:
    """A clock that only advances when someone sleeps.

    Lets a rate-limiter test assert on elapsed time deterministically, with no
    wall-clock flakiness on a loaded CI runner.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def master_idx() -> str:
    return (FIXTURES / "master.20240102.idx").read_text(encoding="utf-8")

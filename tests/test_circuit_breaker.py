from __future__ import annotations

import pytest

from src.ingest.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from tests.conftest import FakeClock


@pytest.fixture
def breaker(clock: FakeClock) -> CircuitBreaker:
    return CircuitBreaker(fail_threshold=3, reset_seconds=10.0, name="test", clock=clock.time)


def test_starts_closed_and_passes_calls(breaker: CircuitBreaker) -> None:
    assert breaker.state is CircuitState.CLOSED
    breaker.before_call()  # does not raise


def test_trips_after_threshold_consecutive_failures(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        breaker.record_failure()

    assert breaker.state is CircuitState.OPEN
    with pytest.raises(CircuitOpenError) as err:
        breaker.before_call()
    assert err.value.retry_after == pytest.approx(10.0)


def test_success_resets_the_failure_run(breaker: CircuitBreaker) -> None:
    """Consecutive, not cumulative. Two failures, a success, two more failures
    must leave the breaker closed."""
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state is CircuitState.CLOSED


def test_half_opens_after_cooldown(breaker: CircuitBreaker, clock: FakeClock) -> None:
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is CircuitState.OPEN

    clock.advance(9.9)
    assert breaker.state is CircuitState.OPEN

    clock.advance(0.2)
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.before_call()  # the single trial call is admitted


def test_success_in_half_open_closes_the_circuit(breaker: CircuitBreaker, clock: FakeClock) -> None:
    for _ in range(3):
        breaker.record_failure()
    clock.advance(10.1)
    assert breaker.state is CircuitState.HALF_OPEN

    breaker.record_success()

    assert breaker.state is CircuitState.CLOSED
    # Counter genuinely reset: it takes a full threshold to trip again.
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED


def test_failure_in_half_open_reopens_for_a_full_cooldown(
    breaker: CircuitBreaker, clock: FakeClock
) -> None:
    for _ in range(3):
        breaker.record_failure()
    clock.advance(10.1)
    assert breaker.state is CircuitState.HALF_OPEN

    breaker.record_failure()  # the trial call failed

    assert breaker.state is CircuitState.OPEN
    clock.advance(9.9)
    assert breaker.state is CircuitState.OPEN, "cooldown restarted, not resumed"
    clock.advance(0.2)
    assert breaker.state is CircuitState.HALF_OPEN


@pytest.mark.parametrize(
    ("threshold", "reset", "match"),
    [(0, 10.0, "fail_threshold"), (3, 0.0, "reset_seconds")],
)
def test_rejects_invalid_configuration(threshold: int, reset: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CircuitBreaker(threshold, reset)

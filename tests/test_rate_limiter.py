"""The rate limiter carries a correctness claim - that it provably caps the
outbound request rate - so it is tested against that claim, not merely smoke-run.
"""

from __future__ import annotations

import threading
import time

import pytest

from src.ingest.rate_limiter import RateLimiter
from tests.conftest import FakeClock


def test_burst_up_to_capacity_is_free(clock: FakeClock) -> None:
    limiter = RateLimiter(10.0, capacity=5, clock=clock.time, sleep=clock.sleep)

    waits = [limiter.acquire() for _ in range(5)]

    assert waits == [0.0] * 5
    assert clock.now == pytest.approx(0.0)


def test_rate_is_capped_over_the_window(clock: FakeClock) -> None:
    """The claim: N acquisitions at rate R, after a burst of C, cannot complete
    faster than (N - C) / R seconds."""
    rate, capacity, n = 10.0, 1.0, 21
    limiter = RateLimiter(rate, capacity=capacity, clock=clock.time, sleep=clock.sleep)

    for _ in range(n):
        limiter.acquire()

    lower_bound = (n - capacity) / rate
    assert clock.now == pytest.approx(lower_bound, rel=1e-9)
    # And the average rate never exceeded the configured ceiling.
    assert n / clock.now <= rate * (1 + capacity / (n - capacity))


def test_tokens_regenerate_while_idle(clock: FakeClock) -> None:
    limiter = RateLimiter(10.0, capacity=5, clock=clock.time, sleep=clock.sleep)
    for _ in range(5):
        limiter.acquire()

    clock.advance(1.0)  # 1s idle regenerates 10 tokens, capped at capacity 5

    assert [limiter.acquire() for _ in range(5)] == [0.0] * 5


def test_requesting_more_than_capacity_raises_rather_than_hanging() -> None:
    limiter = RateLimiter(10.0, capacity=5)
    with pytest.raises(ValueError, match="block forever"):
        limiter.acquire(6)


@pytest.mark.parametrize("bad_rate", [0, -1.0])
def test_rejects_nonpositive_rate(bad_rate: float) -> None:
    with pytest.raises(ValueError, match="rate_per_sec must be positive"):
        RateLimiter(bad_rate)


@pytest.mark.slow
def test_cap_holds_across_threads_in_real_time() -> None:
    """Concurrent Airflow workers share one budget, so the cap must be global,
    not per-thread. Uses the real clock deliberately."""
    rate, capacity, per_thread, threads = 50.0, 1.0, 15, 4
    limiter = RateLimiter(rate, capacity=capacity)
    total = per_thread * threads

    def worker() -> None:
        for _ in range(per_thread):
            limiter.acquire()

    started = time.monotonic()
    pool = [threading.Thread(target=worker) for _ in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    elapsed = time.monotonic() - started

    lower_bound = (total - capacity) / rate
    assert elapsed >= lower_bound * 0.9, (
        f"{total} acquisitions finished in {elapsed:.3f}s, "
        f"faster than the {lower_bound:.3f}s floor implied by {rate} req/s"
    )

"""Token-bucket rate limiter for outbound SEC requests.

Contract
--------
* Over any window long enough to amortise the initial burst, the number of
  granted acquisitions does not exceed ``rate_per_sec * window``.
* ``capacity`` tokens may be spent immediately as a burst; it defaults to one
  second's worth.
* :meth:`acquire` blocks until a token is available and returns the seconds it
  waited, so callers can record the wait as a metric.
* Thread-safe. Multiple Airflow workers sharing one instance share one budget.

The SEC publishes a 10 requests/second ceiling for automated access. We hold a
single limiter below that figure rather than trusting each caller to behave,
because the consequence of exceeding it is an IP-level block.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        rate_per_sec: float,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self._rate = rate_per_sec
        self._capacity = capacity if capacity is not None else rate_per_sec
        if self._capacity <= 0:
            raise ValueError("capacity must be positive")
        self._clock = clock
        self._sleep = sleep
        self._tokens = self._capacity
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available. Returns seconds waited."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens > self._capacity:
            raise ValueError(
                f"requested {tokens} tokens but bucket capacity is {self._capacity}; "
                "this would block forever"
            )

        with self._lock:
            now = self._clock()
            self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
            self._last = now
            # Reserve under the lock and let the balance go negative. The debt
            # is the caller's wait. Sleeping outside the lock lets concurrent
            # callers queue at the correct spacing instead of serialising on it.
            self._tokens -= tokens
            wait = 0.0 if self._tokens >= 0 else (-self._tokens) / self._rate

        if wait > 0:
            self._sleep(wait)
        return wait

    @property
    def rate_per_sec(self) -> float:
        return self._rate

    def __repr__(self) -> str:
        return f"RateLimiter(rate_per_sec={self._rate}, capacity={self._capacity})"

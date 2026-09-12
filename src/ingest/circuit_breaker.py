"""A three-state circuit breaker.

Contract
--------
* ``closed``    - calls pass through. ``fail_threshold`` *consecutive* failures
                  trip the breaker to ``open``.
* ``open``      - calls are rejected immediately with :class:`CircuitOpenError`,
                  for ``reset_seconds``. No request leaves the process.
* ``half_open`` - after the cooldown, exactly one trial call is admitted. It
                  succeeds and the breaker closes with a reset counter; it fails
                  and the breaker re-opens for another full cooldown.

Why this exists: when EDGAR is rate-limiting or down, continuing to retry turns
a degraded dependency into a self-inflicted outage and risks an IP block. The
breaker converts that into a fast, loud, bounded failure - which is what
"incident response" means in code rather than in a document.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import StrEnum


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Numeric encoding for the Prometheus gauge; a gauge cannot carry a string.
STATE_CODE = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}


class CircuitOpenError(RuntimeError):
    """Raised instead of attempting a call while the breaker is open."""

    def __init__(self, name: str, retry_after: float) -> None:
        super().__init__(f"circuit '{name}' is open; retry in {retry_after:.1f}s")
        self.name = name
        self.retry_after = retry_after


class CircuitBreaker:
    def __init__(
        self,
        fail_threshold: int,
        reset_seconds: float,
        *,
        name: str = "default",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if fail_threshold < 1:
            raise ValueError("fail_threshold must be >= 1")
        if reset_seconds <= 0:
            raise ValueError("reset_seconds must be positive")
        self._fail_threshold = fail_threshold
        self._reset_seconds = reset_seconds
        self._name = name
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Current state, after accounting for an elapsed cooldown."""
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if (
            self._state is CircuitState.OPEN
            and self._clock() - self._opened_at >= self._reset_seconds
        ):
            self._state = CircuitState.HALF_OPEN

    def before_call(self) -> None:
        """Raise :class:`CircuitOpenError` if the call must not be attempted."""
        with self._lock:
            self._maybe_half_open()
            if self._state is CircuitState.OPEN:
                elapsed = self._clock() - self._opened_at
                raise CircuitOpenError(self._name, max(0.0, self._reset_seconds - elapsed))

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            # A failed trial in half_open re-opens immediately: the dependency
            # has not recovered, so spend another full cooldown, not one more
            # increment toward the threshold.
            if self._state is CircuitState.HALF_OPEN:
                self._trip()
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._fail_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._consecutive_failures = self._fail_threshold

    def __repr__(self) -> str:
        return f"CircuitBreaker(name={self._name!r}, state={self.state.value})"

"""Structured JSON logging with a correlation id.

One filing's journey - API request, retrieval, LLM call, response - shares a
single ``correlation_id``, so a production incident is one ``grep`` rather than
a reconstruction across four services.

The id lives in a :class:`contextvars.ContextVar`, so it propagates through
async request handlers without being threaded through every function signature.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:16]


def set_correlation_id(value: str | None = None) -> str:
    cid = value or new_correlation_id()
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str | None:
    return _correlation_id.get()


@contextmanager
def correlation_scope(value: str | None = None) -> Iterator[str]:
    """Bind a correlation id for the duration of a block, then restore."""
    token = _correlation_id.set(value or new_correlation_id())
    try:
        yield _correlation_id.get()  # type: ignore[misc]
    finally:
        _correlation_id.reset(token)


def _inject_correlation_id(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    cid = _correlation_id.get()
    if cid is not None:
        event_dict["correlation_id"] = cid
    return event_dict


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Idempotent. Safe to call from an Airflow task and from the API process."""
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper(), logging.INFO))

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _inject_correlation_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)

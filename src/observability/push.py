"""Publish batch-job metrics to the Prometheus Pushgateway.

Airflow tasks and CLI runs are batch jobs: they exit long before a scrape
interval comes round, so pulling from them is impossible. The Pushgateway is
the standard answer - the job pushes on completion and Prometheus scrapes the
gateway instead.

Grouping matters more than it looks. Metrics are pushed under a grouping key of
job plus instance, and ``push_to_gateway`` REPLACES everything in that group.
Pushing per-field accuracy under one group is therefore correct - a field that
stops being scored disappears rather than going stale - but pushing two
different jobs under one key would have each silently delete the other's series.
"""

from __future__ import annotations

from typing import Any

from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger
from src.observability.metrics import REGISTRY

log = get_logger(__name__)


def push_metrics(
    job: str,
    *,
    settings: Settings | None = None,
    grouping: dict[str, str] | None = None,
    registry: Any = None,
) -> bool:
    """Push the collected metrics. Returns whether it succeeded.

    Never raises. A pipeline run that produced correct results must not be
    marked failed because a monitoring sidecar was unreachable - the metrics
    are an observation of the work, not the work itself.
    """
    settings = settings or get_settings()
    gateway = settings.prometheus_pushgateway
    if not gateway:
        return False

    try:
        from prometheus_client import push_to_gateway

        push_to_gateway(
            gateway.replace("http://", "").replace("https://", ""),
            job=job,
            registry=registry if registry is not None else REGISTRY,
            grouping_key=grouping or {},
            timeout=10,
        )
    except Exception as exc:
        log.warning("pushgateway_unavailable", gateway=gateway, job=job, error=str(exc))
        return False

    log.info("metrics_pushed", gateway=gateway, job=job)
    return True

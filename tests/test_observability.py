"""What a batch job is allowed to publish.

These tests exist because of a real incident in this repository, not because the
code looked suspicious. The extraction runner pushed the shared module registry
to the Pushgateway, which published every metric the module owns - including
``index_freshness_seconds`` at its untouched default of 0.

Nothing detected it while the API was up, because the API's own series carried
the true value (663 days) and ``IndexStale`` fired correctly on it. When the API
process died the pushed 0 was the only series left, and the alert silently
cleared. The monitoring reported a 663-day-stale index as perfectly fresh.

An untouched Gauge reads 0. Once it reaches Prometheus, 0 is not "no data" - it
is a confident measurement of zero. So the rule these tests enforce is narrow
and absolute: a batch job publishes what it measured, and nothing else.
"""

from __future__ import annotations

import pytest

from src.observability import metrics
from src.observability.push import push_metrics


def _series_names(registry) -> set[str]:  # type: ignore[no-untyped-def]
    """Every series name a registry would expose, suffixes stripped."""
    names = set()
    for metric in registry.collect():
        for sample in metric.samples:
            name = sample.name
            for suffix in ("_bucket", "_count", "_sum", "_created", "_total"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            names.add(name)
    return names


class TestExtractionEvalRegistry:
    def test_publishes_only_what_the_job_measures(self) -> None:
        registry, accuracy, resolution = metrics.extraction_eval_registry()
        accuracy.labels(field="total_revenue").set(0.826)
        resolution.labels(field="total_revenue").set(1.0)

        assert _series_names(registry) == {
            f"{metrics._NS}_extraction_accuracy_ratio",
            f"{metrics._NS}_ground_truth_resolution_ratio",
        }

    def test_does_not_publish_index_freshness(self) -> None:
        """The specific series whose default 0 cleared a live alert."""
        registry, _, _ = metrics.extraction_eval_registry()
        assert f"{metrics._NS}_index_freshness_seconds" not in _series_names(registry)

    def test_shared_registry_would_have_leaked_it(self) -> None:
        """Proves the bug was real, so the guard above is not cargo cult.

        If this ever fails, the shared registry stopped owning index freshness
        and the reasoning in these tests needs rewriting - not the assertion.
        """
        assert f"{metrics._NS}_index_freshness_seconds" in _series_names(metrics.REGISTRY)

    def test_each_call_is_isolated(self) -> None:
        """A fresh registry per run, so one run cannot leak into the next."""
        _first, first_accuracy, _ = metrics.extraction_eval_registry()
        first_accuracy.labels(field="net_income").set(0.87)
        second, _, _ = metrics.extraction_eval_registry()

        samples = [s for m in second.collect() for s in m.samples]
        assert samples == []


class TestPushMetrics:
    def test_registry_is_required(self) -> None:
        """No default. A caller must state what it is publishing."""
        with pytest.raises(TypeError):
            push_metrics("extraction_eval")  # type: ignore[call-arg]

    def test_returns_false_when_no_gateway_configured(self) -> None:
        registry, _, _ = metrics.extraction_eval_registry()
        settings = metrics.get_settings().model_copy(update={"prometheus_pushgateway": ""})
        assert push_metrics("extraction_eval", registry, settings=settings) is False

    def test_never_raises_when_the_gateway_is_unreachable(self) -> None:
        """A correct run must not be failed by an unreachable sidecar.

        The metrics are an observation of the work, not the work itself.
        """
        registry, _, _ = metrics.extraction_eval_registry()
        settings = metrics.get_settings().model_copy(
            update={"prometheus_pushgateway": "http://127.0.0.1:1"}
        )
        assert push_metrics("extraction_eval", registry, settings=settings) is False

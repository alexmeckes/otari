from types import SimpleNamespace
from typing import Any

import pytest

from gateway.services.routing_policy_service import RoutingCandidate
from gateway.services.routing_provider_health import (
    ProviderHealth,
    _provider_health_from_counts,
    apply_provider_health_gate,
    apply_provider_health_order,
    attach_provider_health,
)


def _health(status: str, provider: str = "openai") -> ProviderHealth:
    return ProviderHealth(
        provider=provider,
        status=status,
        sample_count=4,
        success_count=1,
        error_count=3,
        failure_rate=0.75,
        reason="failure_rate_exceeds_unhealthy_threshold",
    )


def _candidate(model: str, *, position: int) -> RoutingCandidate:
    provider, provider_model = model.split(":", 1)
    return RoutingCandidate(
        model=model,
        provider=provider,
        provider_model=provider_model,
        position=position,
        tier=None,
        estimated_cost=None,
        input_price_per_million=None,
        output_price_per_million=None,
        quality_score=None,
        average_latency_ms=None,
        latency_sample_count=0,
        routing_score=None,
        score_components=None,
        provider_health=None,
        metadata={},
    )


class _Trace:
    def __init__(
        self,
        attempts: list[Any] | None = None,
        *,
        selected_provider: str | None = None,
        status: str = "success",
    ) -> None:
        self._attempts = attempts or []
        self.selected_provider = selected_provider
        self.status = status

    def attempt_list(self) -> list[Any]:
        return self._attempts


class _Scalars:
    def __init__(self, traces: list[_Trace]) -> None:
        self._traces = traces

    def all(self) -> list[_Trace]:
        return self._traces


class _Result:
    def __init__(self, traces: list[_Trace]) -> None:
        self._traces = traces

    def scalars(self) -> _Scalars:
        return _Scalars(self._traces)


class _Db:
    def __init__(self, traces: list[_Trace]) -> None:
        self._traces = traces
        self.statement: Any | None = None

    async def execute(self, statement: Any) -> _Result:
        self.statement = statement
        return _Result(self._traces)


def test_provider_health_from_counts_uses_configured_min_samples_and_thresholds() -> None:
    health = _provider_health_from_counts(
        "openai",
        success_count=1,
        error_count=1,
        health_config={
            "min_samples": 2,
            "degraded_failure_rate": 0.25,
            "unhealthy_failure_rate": 0.75,
        },
    )

    assert health.status == "degraded"
    assert health.failure_rate == 0.5
    assert health.reason == "failure_rate_exceeds_degraded_threshold"


def test_provider_health_from_counts_preserves_unknown_and_healthy_results() -> None:
    no_samples = _provider_health_from_counts(
        "openai",
        success_count=0,
        error_count=0,
        health_config={"min_samples": 3},
    )
    unknown = _provider_health_from_counts(
        "openai",
        success_count=1,
        error_count=0,
        health_config={"min_samples": 3},
    )
    healthy = _provider_health_from_counts(
        "openai",
        success_count=4,
        error_count=0,
        health_config={"min_samples": 3},
    )

    assert no_samples.status == "unknown"
    assert no_samples.sample_count == 0
    assert no_samples.failure_rate is None
    assert no_samples.reason == "insufficient_samples"
    assert unknown.status == "unknown"
    assert unknown.sample_count == 1
    assert unknown.failure_rate == 0.0
    assert unknown.reason == "insufficient_samples"
    assert healthy.status == "healthy"
    assert healthy.sample_count == 4
    assert healthy.failure_rate == 0.0
    assert healthy.reason == "failure_rate_below_threshold"


def test_provider_health_from_counts_preserves_unhealthy_results() -> None:
    health = _provider_health_from_counts(
        "openai",
        success_count=1,
        error_count=3,
        health_config={
            "min_samples": 3,
            "unhealthy_failure_rate": 0.75,
        },
    )

    assert health.status == "unhealthy"
    assert health.sample_count == 4
    assert health.failure_rate == 0.75
    assert health.reason == "failure_rate_exceeds_unhealthy_threshold"


def test_provider_health_from_counts_caps_configured_thresholds() -> None:
    health = _provider_health_from_counts(
        "openai",
        success_count=0,
        error_count=4,
        health_config={
            "min_samples": 1,
            "degraded_failure_rate": 2.0,
            "unhealthy_failure_rate": 2.0,
        },
    )

    assert health.status == "unhealthy"
    assert health.failure_rate == 1.0
    assert health.reason == "failure_rate_exceeds_unhealthy_threshold"


def test_provider_health_from_counts_preserves_zero_thresholds() -> None:
    health = _provider_health_from_counts(
        "openai",
        success_count=4,
        error_count=0,
        health_config={
            "min_samples": 1,
            "degraded_failure_rate": 0.0,
            "unhealthy_failure_rate": 0.0,
        },
    )

    assert health.status == "unhealthy"
    assert health.failure_rate == 0.0
    assert health.reason == "failure_rate_exceeds_unhealthy_threshold"


@pytest.mark.asyncio
async def test_attach_provider_health_counts_known_attempt_and_trace_outcomes() -> None:
    db = _Db(
        [
            _Trace(
                [
                    {"provider": "openai", "status": "success"},
                    {"provider": "openai", "status": "error"},
                    {"provider": "unknown", "status": "success"},
                    {"provider": "anthropic", "status": "timeout"},
                    "not-an-attempt",
                ]
            ),
            _Trace(selected_provider="anthropic", status="error"),
            _Trace(selected_provider="unknown", status="success"),
            _Trace(selected_provider="openai", status="timeout"),
        ]
    )

    enriched = await attach_provider_health(
        db,
        [
            _candidate("openai:gpt-4o-mini", position=1),
            _candidate("anthropic:claude-3-5-haiku-latest", position=2),
        ],
        config={
            "health": {
                "enabled": True,
                "min_samples": 1,
                "degraded_failure_rate": 0.75,
                "unhealthy_failure_rate": 0.9,
            }
        },
    )

    health_by_provider = {candidate.provider: candidate.provider_health for candidate in enriched}
    openai_health = health_by_provider["openai"]
    anthropic_health = health_by_provider["anthropic"]

    assert openai_health is not None
    assert openai_health.success_count == 1
    assert openai_health.error_count == 1
    assert openai_health.status == "healthy"
    assert openai_health.failure_rate == 0.5
    assert anthropic_health is not None
    assert anthropic_health.success_count == 0
    assert anthropic_health.error_count == 1
    assert anthropic_health.status == "unhealthy"
    assert anthropic_health.failure_rate == 1.0


def test_apply_provider_health_gate_partitions_unhealthy_candidates() -> None:
    healthy_candidate = SimpleNamespace(
        model="openai:gpt-4o",
        provider="openai",
        estimated_cost=0.001,
        provider_health=_health("healthy"),
    )
    unknown_candidate = SimpleNamespace(provider_health=None)
    unhealthy_candidate = SimpleNamespace(
        model="anthropic:claude-3-5-haiku-latest",
        provider="anthropic",
        estimated_cost=0.003,
        provider_health=_health("unhealthy", provider="anthropic"),
    )

    allowed, rejected = apply_provider_health_gate(
        [healthy_candidate, unknown_candidate, unhealthy_candidate],
        config={"health": {"enabled": True, "mode": "skip_unhealthy"}},
    )

    assert allowed == [healthy_candidate, unknown_candidate]
    assert rejected == [
        {
            "model": "anthropic:claude-3-5-haiku-latest",
            "provider": "anthropic",
            "reason": "provider_unhealthy",
            "estimated_cost": 0.003,
            "provider_health": {
                "provider": "anthropic",
                "status": "unhealthy",
                "sample_count": 4,
                "success_count": 1,
                "error_count": 3,
                "failure_rate": 0.75,
                "reason": "failure_rate_exceeds_unhealthy_threshold",
            },
        }
    ]


def test_apply_provider_health_gate_requires_enabled_skip_mode() -> None:
    unhealthy_candidate = SimpleNamespace(
        model="openai:gpt-4o-mini",
        provider="openai",
        estimated_cost=0.002,
        provider_health=_health("unhealthy"),
    )

    assert apply_provider_health_gate(
        [unhealthy_candidate],
        config={"health": {"enabled": True, "mode": "downrank"}},
    ) == ([unhealthy_candidate], [])
    assert apply_provider_health_gate(
        [unhealthy_candidate],
        config={"health": {"enabled": True, "mode": "observe"}},
    ) == ([unhealthy_candidate], [])
    assert apply_provider_health_gate(
        [unhealthy_candidate],
        config={"health": {"mode": "skip_unhealthy"}},
    ) == ([unhealthy_candidate], [])
    assert apply_provider_health_gate(
        [unhealthy_candidate],
        config={"health": "skip_unhealthy"},
    ) == ([unhealthy_candidate], [])


def test_apply_provider_health_order_requires_enabled_downrank_mode() -> None:
    unhealthy_candidate = SimpleNamespace(position=1, provider_health=_health("unhealthy"))
    healthy_candidate = SimpleNamespace(position=2, provider_health=_health("healthy"))
    unknown_first_candidate = SimpleNamespace(position=0, provider_health=None)
    unknown_later_candidate = SimpleNamespace(position=3, provider_health=None)
    degraded_candidate = SimpleNamespace(position=4, provider_health=_health("degraded"))
    candidates = [
        unhealthy_candidate,
        degraded_candidate,
        unknown_later_candidate,
        healthy_candidate,
        unknown_first_candidate,
    ]

    assert apply_provider_health_order(
        candidates,
        config={"health": {"enabled": True, "mode": "downrank"}},
    ) == [
        healthy_candidate,
        unknown_first_candidate,
        unknown_later_candidate,
        degraded_candidate,
        unhealthy_candidate,
    ]
    assert apply_provider_health_order(
        candidates,
        config={"health": {"enabled": True, "mode": "watch"}},
    ) == [
        healthy_candidate,
        unknown_first_candidate,
        unknown_later_candidate,
        degraded_candidate,
        unhealthy_candidate,
    ]
    assert apply_provider_health_order(
        candidates,
        config={"health": {"enabled": True, "mode": "skip_unhealthy"}},
    ) == candidates
    assert apply_provider_health_order(
        candidates,
        config={"health": {"enabled": True, "mode": "observe"}},
    ) == candidates
    assert apply_provider_health_order(
        candidates,
        config={"health": {"mode": "downrank"}},
    ) == candidates
    assert apply_provider_health_order(
        candidates,
        config={"health": "downrank"},
    ) == candidates

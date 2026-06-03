from types import SimpleNamespace

from gateway.services.routing_provider_health import (
    ProviderHealth,
    _provider_health_enabled,
    _provider_health_from_counts,
    _provider_health_mode,
    _record_provider_outcome,
    apply_provider_health_gate,
    apply_provider_health_order,
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


def test_provider_health_helpers_read_shared_health_config_values() -> None:
    health_config = {
        "enabled": True,
        "mode": "skip_unhealthy",
    }

    assert _provider_health_enabled(health_config) is True
    assert _provider_health_mode(health_config) == "skip_unhealthy"


def test_provider_health_mode_defaults_for_unknown_mode() -> None:
    assert _provider_health_mode({"mode": "watch"}) == "downrank"


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


def test_record_provider_outcome_counts_known_success_and_error_only() -> None:
    counts_by_provider = {
        "openai": {"success": 0, "error": 0},
        "anthropic": {"success": 0, "error": 0},
    }

    _record_provider_outcome(counts_by_provider, "openai", "success")
    _record_provider_outcome(counts_by_provider, "openai", "error")
    _record_provider_outcome(counts_by_provider, "unknown", "success")
    _record_provider_outcome(counts_by_provider, "anthropic", "timeout")
    _record_provider_outcome(counts_by_provider, None, "error")
    _record_provider_outcome(counts_by_provider, "anthropic", None)

    assert counts_by_provider == {
        "openai": {"success": 1, "error": 1},
        "anthropic": {"success": 0, "error": 0},
    }


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
        config={"health": {"enabled": True, "mode": "skip_unhealthy"}},
    ) == candidates
    assert apply_provider_health_order(
        candidates,
        config={"health": {"mode": "downrank"}},
    ) == candidates
    assert apply_provider_health_order(
        candidates,
        config={"health": "downrank"},
    ) == candidates

from types import SimpleNamespace

from gateway.services.routing_provider_health import (
    ProviderHealth,
    _provider_health_config,
    _provider_health_enabled,
    _provider_health_enabled_for_mode,
    _provider_health_from_counts,
    _provider_health_gate_rejection,
    _provider_health_mode,
    _provider_health_rate,
    _provider_health_status_reason,
    _record_provider_outcome,
    apply_provider_health_gate,
)


def _health(status: str) -> ProviderHealth:
    return ProviderHealth(
        provider="openai",
        status=status,
        sample_count=4,
        success_count=1,
        error_count=3,
        failure_rate=0.75,
        reason="failure_rate_exceeds_unhealthy_threshold",
    )


def test_provider_health_helpers_read_shared_health_config_values() -> None:
    config = {
        "health": {
            "enabled": True,
            "mode": "skip_unhealthy",
            "degraded_failure_rate": 2.0,
        }
    }
    health_config = _provider_health_config(config)

    assert _provider_health_enabled(health_config) is True
    assert _provider_health_mode(health_config) == "skip_unhealthy"
    assert _provider_health_rate(health_config, "degraded_failure_rate", 0.25) == 1.0


def test_provider_health_enabled_for_mode_requires_enabled_matching_mode() -> None:
    health_config = _provider_health_config({"health": {"enabled": True, "mode": "skip_unhealthy"}})

    assert _provider_health_enabled_for_mode(health_config, "skip_unhealthy") is True
    assert _provider_health_enabled_for_mode(health_config, "downrank") is False
    assert (
        _provider_health_enabled_for_mode(
            _provider_health_config({"health": {"mode": "skip_unhealthy"}}),
            "skip_unhealthy",
        )
        is False
    )


def test_provider_health_mode_defaults_for_unknown_mode() -> None:
    assert _provider_health_mode(_provider_health_config({"health": {"mode": "watch"}})) == "downrank"


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


def test_provider_health_status_reason_preserves_threshold_ordering() -> None:
    assert (
        _provider_health_status_reason(
            sample_count=0,
            failure_rate=None,
            min_samples=3,
            degraded_rate=0.25,
            unhealthy_rate=0.50,
        )
        == ("unknown", "insufficient_samples")
    )
    assert (
        _provider_health_status_reason(
            sample_count=4,
            failure_rate=0.75,
            min_samples=3,
            degraded_rate=0.25,
            unhealthy_rate=0.50,
        )
        == ("unhealthy", "failure_rate_exceeds_unhealthy_threshold")
    )
    assert (
        _provider_health_status_reason(
            sample_count=4,
            failure_rate=0.25,
            min_samples=3,
            degraded_rate=0.25,
            unhealthy_rate=0.50,
        )
        == ("degraded", "failure_rate_exceeds_degraded_threshold")
    )
    assert (
        _provider_health_status_reason(
            sample_count=4,
            failure_rate=0.0,
            min_samples=3,
            degraded_rate=0.25,
            unhealthy_rate=0.50,
        )
        == ("healthy", "failure_rate_below_threshold")
    )


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


def test_provider_health_gate_rejection_returns_payload_only_for_unhealthy_candidates() -> None:
    healthy_candidate = SimpleNamespace(provider_health=_health("healthy"))
    unknown_candidate = SimpleNamespace(provider_health=None)
    unhealthy_candidate = SimpleNamespace(
        model="openai:gpt-4o-mini",
        provider="openai",
        estimated_cost=0.002,
        provider_health=_health("unhealthy"),
    )

    assert _provider_health_gate_rejection(healthy_candidate) is None
    assert _provider_health_gate_rejection(unknown_candidate) is None
    assert _provider_health_gate_rejection(unhealthy_candidate) == {
        "model": "openai:gpt-4o-mini",
        "provider": "openai",
        "reason": "provider_unhealthy",
        "estimated_cost": 0.002,
        "provider_health": {
            "provider": "openai",
            "status": "unhealthy",
            "sample_count": 4,
            "success_count": 1,
            "error_count": 3,
            "failure_rate": 0.75,
            "reason": "failure_rate_exceeds_unhealthy_threshold",
        },
    }


def test_apply_provider_health_gate_partitions_unhealthy_candidates() -> None:
    healthy_candidate = SimpleNamespace(
        model="openai:gpt-4o",
        provider="openai",
        estimated_cost=0.001,
        provider_health=_health("healthy"),
    )
    unhealthy_candidate = SimpleNamespace(
        model="anthropic:claude-3-5-haiku-latest",
        provider="anthropic",
        estimated_cost=0.003,
        provider_health=_health("unhealthy"),
    )

    allowed, rejected = apply_provider_health_gate(
        [healthy_candidate, unhealthy_candidate],
        config={"health": {"enabled": True, "mode": "skip_unhealthy"}},
    )

    assert allowed == [healthy_candidate]
    assert [(item["model"], item["reason"]) for item in rejected] == [
        ("anthropic:claude-3-5-haiku-latest", "provider_unhealthy")
    ]

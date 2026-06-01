from gateway.services.routing_provider_health import (
    _provider_health_enabled,
    _provider_health_enabled_for_mode,
    _provider_health_from_counts,
    _provider_health_mode,
    _provider_health_rate,
)


def test_provider_health_helpers_read_shared_health_config_values() -> None:
    config = {
        "health": {
            "enabled": True,
            "mode": "skip_unhealthy",
            "degraded_failure_rate": 2.0,
        }
    }

    assert _provider_health_enabled(config) is True
    assert _provider_health_mode(config) == "skip_unhealthy"
    assert _provider_health_rate(config, "degraded_failure_rate", 0.25) == 1.0


def test_provider_health_enabled_for_mode_requires_enabled_matching_mode() -> None:
    config = {"health": {"enabled": True, "mode": "skip_unhealthy"}}

    assert _provider_health_enabled_for_mode(config, "skip_unhealthy") is True
    assert _provider_health_enabled_for_mode(config, "downrank") is False
    assert _provider_health_enabled_for_mode({"health": {"mode": "skip_unhealthy"}}, "skip_unhealthy") is False


def test_provider_health_mode_defaults_for_unknown_mode() -> None:
    assert _provider_health_mode({"health": {"mode": "watch"}}) == "downrank"


def test_provider_health_from_counts_uses_configured_min_samples_and_thresholds() -> None:
    health = _provider_health_from_counts(
        "openai",
        success_count=1,
        error_count=1,
        config={
            "health": {
                "min_samples": 2,
                "degraded_failure_rate": 0.25,
                "unhealthy_failure_rate": 0.75,
            }
        },
    )

    assert health.status == "degraded"
    assert health.failure_rate == 0.5
    assert health.reason == "failure_rate_exceeds_degraded_threshold"

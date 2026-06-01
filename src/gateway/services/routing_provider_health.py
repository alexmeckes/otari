from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import RouteTrace
from gateway.services.routing_config_values import bool_config, dict_or_empty, int_config, non_negative_float_or_none
from gateway.services.routing_trace_attempts import attempt_outcome, attempt_provider

_HEALTH_MODES = {"observe", "downrank", "skip_unhealthy"}
_HEALTH_RANK = {"healthy": 0, "unknown": 1, "degraded": 2, "unhealthy": 3}


@dataclass(frozen=True)
class ProviderHealth:
    """Passive provider health summary derived from recent route traces."""

    provider: str
    status: str
    sample_count: int
    success_count: int
    error_count: int
    failure_rate: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe health representation."""
        return {
            "provider": self.provider,
            "status": self.status,
            "sample_count": self.sample_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "failure_rate": self.failure_rate,
            "reason": self.reason,
        }


def _provider_health_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict_or_empty(config.get("health"))


def _provider_health_enabled(health_config: Mapping[str, Any]) -> bool:
    return bool_config(health_config.get("enabled"), False)


def _provider_health_mode(health_config: Mapping[str, Any]) -> str:
    mode = health_config.get("mode")
    return mode if isinstance(mode, str) and mode in _HEALTH_MODES else "downrank"


def _provider_health_rate(health_config: Mapping[str, Any], key: str, default: float) -> float:
    rate = non_negative_float_or_none(health_config.get(key))
    if rate is None:
        return default
    return min(rate, 1.0)


def _record_provider_outcome(
    counts_by_provider: dict[str, dict[str, int]],
    provider: str | None,
    outcome: str | None,
) -> None:
    if provider not in counts_by_provider or outcome not in {"success", "error"}:
        return
    counts_by_provider[provider][outcome] += 1


def _provider_health_status_reason(
    *,
    sample_count: int,
    failure_rate: float | None,
    min_samples: int,
    degraded_rate: float,
    unhealthy_rate: float,
) -> tuple[str, str]:
    if sample_count < min_samples or failure_rate is None:
        return "unknown", "insufficient_samples"
    if failure_rate >= unhealthy_rate:
        return "unhealthy", "failure_rate_exceeds_unhealthy_threshold"
    if failure_rate >= degraded_rate:
        return "degraded", "failure_rate_exceeds_degraded_threshold"
    return "healthy", "failure_rate_below_threshold"


def _provider_health_from_counts(
    provider: str,
    *,
    success_count: int,
    error_count: int,
    health_config: Mapping[str, Any],
) -> ProviderHealth:
    sample_count = success_count + error_count
    min_samples = int_config(health_config.get("min_samples"), 3)
    degraded_rate = _provider_health_rate(health_config, "degraded_failure_rate", 0.25)
    unhealthy_rate = _provider_health_rate(health_config, "unhealthy_failure_rate", 0.50)
    failure_rate = None if sample_count == 0 else error_count / sample_count
    status, reason = _provider_health_status_reason(
        sample_count=sample_count,
        failure_rate=failure_rate,
        min_samples=min_samples,
        degraded_rate=degraded_rate,
        unhealthy_rate=unhealthy_rate,
    )

    return ProviderHealth(
        provider=provider,
        status=status,
        sample_count=sample_count,
        success_count=success_count,
        error_count=error_count,
        failure_rate=failure_rate,
        reason=reason,
    )


async def attach_provider_health(
    db: AsyncSession,
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[Any]:
    health_config = _provider_health_config(config)
    if not _provider_health_enabled(health_config):
        return list(candidates)

    candidate_providers = {candidate.provider for candidate in candidates}
    if not candidate_providers:
        return list(candidates)

    sample_limit = int_config(health_config.get("sample_limit"), 200)
    counts_by_provider = {
        provider: {"success": 0, "error": 0}
        for provider in candidate_providers
    }
    result = await db.execute(select(RouteTrace).order_by(RouteTrace.timestamp.desc()).limit(sample_limit))
    for trace in result.scalars().all():
        attempts = trace.attempt_list()
        if attempts:
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                provider = attempt_provider(attempt)
                outcome = attempt_outcome(attempt)
                _record_provider_outcome(counts_by_provider, provider, outcome)
            continue

        _record_provider_outcome(counts_by_provider, trace.selected_provider, trace.status)

    health_by_provider = {
        provider: _provider_health_from_counts(
            provider,
            success_count=counts["success"],
            error_count=counts["error"],
            health_config=health_config,
        )
        for provider, counts in counts_by_provider.items()
    }
    return [
        replace(candidate, provider_health=health_by_provider.get(candidate.provider))
        for candidate in candidates
    ]


def apply_provider_health_gate(
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    health_config = _provider_health_config(config)
    if not _provider_health_enabled(health_config) or _provider_health_mode(health_config) != "skip_unhealthy":
        return list(candidates), []

    allowed: list[Any] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        rejection = _provider_health_gate_rejection(candidate)
        if rejection is None:
            allowed.append(candidate)
            continue
        rejected.append(rejection)
    return allowed, rejected


def _provider_health_gate_rejection(candidate: Any) -> dict[str, Any] | None:
    if candidate.provider_health is None or candidate.provider_health.status != "unhealthy":
        return None
    return {
        "model": candidate.model,
        "provider": candidate.provider,
        "reason": "provider_unhealthy",
        "estimated_cost": candidate.estimated_cost,
        "provider_health": candidate.provider_health.to_dict(),
    }


def _by_health(candidate: Any) -> tuple[int, int]:
    status_value = candidate.provider_health.status if candidate.provider_health else "unknown"
    return (_HEALTH_RANK.get(status_value, _HEALTH_RANK["unknown"]), candidate.position)


def apply_provider_health_order(
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[Any]:
    health_config = _provider_health_config(config)
    if not _provider_health_enabled(health_config) or _provider_health_mode(health_config) != "downrank":
        return list(candidates)
    return sorted(candidates, key=_by_health)

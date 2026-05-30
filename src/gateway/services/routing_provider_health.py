from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import RouteTrace
from gateway.services import routing_request_analysis as _routing_request_analysis

_HEALTH_MODES = {"observe", "downrank", "skip_unhealthy"}
_HEALTH_RANK = {"healthy": 0, "unknown": 1, "degraded": 2, "unhealthy": 3}
_bool_config = _routing_request_analysis.bool_config
_int_config = _routing_request_analysis.int_config


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


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _non_negative_float_or_none(value: Any) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _provider_health_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    health = config.get("health")
    return health if isinstance(health, dict) else {}


def _provider_health_enabled(config: Mapping[str, Any]) -> bool:
    return _bool_config(_provider_health_config(config).get("enabled"), False)


def _provider_health_mode(config: Mapping[str, Any]) -> str:
    mode = _provider_health_config(config).get("mode")
    return mode if isinstance(mode, str) and mode in _HEALTH_MODES else "downrank"


def _provider_health_rate(config: Mapping[str, Any], key: str, default: float) -> float:
    rate = _non_negative_float_or_none(_provider_health_config(config).get(key))
    if rate is None:
        return default
    return min(rate, 1.0)


def _provider_health_from_counts(
    provider: str,
    *,
    success_count: int,
    error_count: int,
    config: Mapping[str, Any],
) -> ProviderHealth:
    sample_count = success_count + error_count
    min_samples = _int_config(_provider_health_config(config).get("min_samples"), 3)
    degraded_rate = _provider_health_rate(config, "degraded_failure_rate", 0.25)
    unhealthy_rate = _provider_health_rate(config, "unhealthy_failure_rate", 0.50)
    failure_rate = None if sample_count == 0 else error_count / sample_count

    if sample_count < min_samples or failure_rate is None:
        return ProviderHealth(
            provider=provider,
            status="unknown",
            sample_count=sample_count,
            success_count=success_count,
            error_count=error_count,
            failure_rate=failure_rate,
            reason="insufficient_samples",
        )
    if failure_rate >= unhealthy_rate:
        return ProviderHealth(
            provider=provider,
            status="unhealthy",
            sample_count=sample_count,
            success_count=success_count,
            error_count=error_count,
            failure_rate=failure_rate,
            reason="failure_rate_exceeds_unhealthy_threshold",
        )
    if failure_rate >= degraded_rate:
        return ProviderHealth(
            provider=provider,
            status="degraded",
            sample_count=sample_count,
            success_count=success_count,
            error_count=error_count,
            failure_rate=failure_rate,
            reason="failure_rate_exceeds_degraded_threshold",
        )
    return ProviderHealth(
        provider=provider,
        status="healthy",
        sample_count=sample_count,
        success_count=success_count,
        error_count=error_count,
        failure_rate=failure_rate,
        reason="failure_rate_below_threshold",
    )


def _attempt_provider(attempt: Mapping[str, Any]) -> str | None:
    provider = attempt.get("provider")
    return provider if isinstance(provider, str) and provider else None


def _attempt_outcome(attempt: Mapping[str, Any]) -> str | None:
    status = attempt.get("status")
    if status in {"success", "error"}:
        return str(status)
    return None


async def attach_provider_health(
    db: AsyncSession,
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[Any]:
    if not _provider_health_enabled(config):
        return list(candidates)

    candidate_providers = {candidate.provider for candidate in candidates}
    if not candidate_providers:
        return list(candidates)

    sample_limit = _int_config(_provider_health_config(config).get("sample_limit"), 200)
    counts_by_provider = {
        provider: {"success": 0, "error": 0}
        for provider in candidate_providers
    }
    result = await db.execute(select(RouteTrace).order_by(RouteTrace.timestamp.desc()).limit(sample_limit))
    for trace in result.scalars().all():
        attempts = trace.attempts if isinstance(trace.attempts, list) else []
        if attempts:
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                provider = _attempt_provider(attempt)
                outcome = _attempt_outcome(attempt)
                if provider in counts_by_provider and outcome is not None:
                    counts_by_provider[provider][outcome] += 1
            continue

        if trace.selected_provider in counts_by_provider and trace.status in {"success", "error"}:
            counts_by_provider[trace.selected_provider][trace.status] += 1

    health_by_provider = {
        provider: _provider_health_from_counts(
            provider,
            success_count=counts["success"],
            error_count=counts["error"],
            config=config,
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
    if not _provider_health_enabled(config) or _provider_health_mode(config) != "skip_unhealthy":
        return list(candidates), []

    allowed: list[Any] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.provider_health is None or candidate.provider_health.status != "unhealthy":
            allowed.append(candidate)
            continue
        rejected.append(
            {
                "model": candidate.model,
                "provider": candidate.provider,
                "reason": "provider_unhealthy",
                "estimated_cost": candidate.estimated_cost,
                "provider_health": candidate.provider_health.to_dict(),
            }
        )
    return allowed, rejected


def _by_health(candidate: Any) -> tuple[int, int]:
    status_value = candidate.provider_health.status if candidate.provider_health else "unknown"
    return (_HEALTH_RANK.get(status_value, _HEALTH_RANK["unknown"]), candidate.position)


def apply_provider_health_order(
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[Any]:
    if not _provider_health_enabled(config) or _provider_health_mode(config) != "downrank":
        return list(candidates)
    return sorted(candidates, key=_by_health)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import RouteTrace
from gateway.services.routing_config_values import bool_config, dict_or_empty, int_config, non_negative_float_or_none
from gateway.services.routing_trace_attempts import attempt_outcome, attempt_provider


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


def _provider_health_from_counts(
    provider: str,
    *,
    success_count: int,
    error_count: int,
    health_config: Mapping[str, Any],
) -> ProviderHealth:
    sample_count = success_count + error_count
    min_samples = int_config(health_config.get("min_samples"), 3)
    degraded_rate = _failure_rate_threshold(health_config.get("degraded_failure_rate"), 0.25)
    unhealthy_rate = _failure_rate_threshold(health_config.get("unhealthy_failure_rate"), 0.50)
    failure_rate = None if sample_count == 0 else error_count / sample_count
    if sample_count < min_samples or failure_rate is None:
        status, reason = "unknown", "insufficient_samples"
    elif failure_rate >= unhealthy_rate:
        status, reason = "unhealthy", "failure_rate_exceeds_unhealthy_threshold"
    elif failure_rate >= degraded_rate:
        status, reason = "degraded", "failure_rate_exceeds_degraded_threshold"
    else:
        status, reason = "healthy", "failure_rate_below_threshold"

    return ProviderHealth(
        provider=provider,
        status=status,
        sample_count=sample_count,
        success_count=success_count,
        error_count=error_count,
        failure_rate=failure_rate,
        reason=reason,
    )


def _failure_rate_threshold(value: Any, default: float) -> float:
    parsed = non_negative_float_or_none(value)
    return default if parsed is None else min(parsed, 1.0)


def _health_mode(health_config: Mapping[str, Any]) -> str:
    mode = health_config.get("mode")
    return mode if isinstance(mode, str) and mode in {"observe", "downrank", "skip_unhealthy"} else "downrank"


def _record_provider_outcome(
    counts_by_provider: dict[str, dict[str, int]],
    *,
    provider: str | None,
    outcome: str | None,
) -> None:
    if provider is not None and outcome in {"success", "error"} and provider in counts_by_provider:
        counts_by_provider[provider][outcome] += 1


async def attach_provider_health(
    db: AsyncSession,
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[Any]:
    health_config = dict_or_empty(config.get("health"))
    if not bool_config(health_config.get("enabled"), False):
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
                _record_provider_outcome(
                    counts_by_provider,
                    provider=attempt_provider(attempt),
                    outcome=attempt_outcome(attempt),
                )
            continue

        _record_provider_outcome(
            counts_by_provider,
            provider=trace.selected_provider,
            outcome=trace.status,
        )

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
    health_config = dict_or_empty(config.get("health"))
    if not bool_config(health_config.get("enabled"), False) or _health_mode(health_config) != "skip_unhealthy":
        return list(candidates), []

    allowed: list[Any] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        provider_health = candidate.provider_health
        if provider_health is None or provider_health.status != "unhealthy":
            allowed.append(candidate)
            continue
        rejected.append(
            {
                "model": candidate.model,
                "provider": candidate.provider,
                "reason": "provider_unhealthy",
                "estimated_cost": candidate.estimated_cost,
                "provider_health": provider_health.to_dict(),
            }
        )
    return allowed, rejected


def apply_provider_health_order(
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[Any]:
    health_config = dict_or_empty(config.get("health"))
    if not bool_config(health_config.get("enabled"), False) or _health_mode(health_config) != "downrank":
        return list(candidates)
    health_rank = {
        "healthy": 0,
        "unknown": 1,
        "degraded": 2,
        "unhealthy": 3,
    }
    return sorted(
        candidates,
        key=lambda candidate: (
            health_rank.get(
                candidate.provider_health.status if candidate.provider_health else "unknown",
                health_rank["unknown"],
            ),
            candidate.position,
        ),
    )

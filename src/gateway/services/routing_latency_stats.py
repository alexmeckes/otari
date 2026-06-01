from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import RouteTrace
from gateway.services.routing_request_analysis import int_config
from gateway.services.routing_trace_attempts import attempt_duration_ms, attempt_model_key


def _latency_config_value(config: Mapping[str, Any], key: str) -> Any:
    return config.get(key)


def _latency_sample_limit(config: Mapping[str, Any]) -> int:
    return int_config(_latency_config_value(config, "latency_sample_limit"), 200)


def _latency_min_samples(config: Mapping[str, Any]) -> int:
    return int_config(_latency_config_value(config, "latency_min_samples"), 1)


async def attach_latency_stats(
    db: AsyncSession,
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[Any]:
    """Attach recent successful route latency stats to candidate dataclasses."""
    candidate_models = {candidate.model for candidate in candidates}
    if not candidate_models:
        return list(candidates)

    sample_limit = _latency_sample_limit(config)
    min_samples = _latency_min_samples(config)
    result = await db.execute(
        select(RouteTrace)
        .where(RouteTrace.status == "success")
        .order_by(RouteTrace.timestamp.desc())
        .limit(sample_limit)
    )
    durations_by_model: dict[str, list[float]] = {model: [] for model in candidate_models}
    for trace in result.scalars().all():
        for attempt in trace.attempt_list():
            if not isinstance(attempt, dict) or attempt.get("status") != "success":
                continue
            model_key = attempt_model_key(attempt)
            duration_ms = attempt_duration_ms(attempt)
            if model_key in durations_by_model and duration_ms is not None:
                durations_by_model[model_key].append(duration_ms)

    enriched: list[Any] = []
    for candidate in candidates:
        durations = durations_by_model.get(candidate.model, [])
        if len(durations) < min_samples:
            enriched.append(candidate)
            continue
        enriched.append(
            replace(
                candidate,
                average_latency_ms=sum(durations) / len(durations),
                latency_sample_count=len(durations),
            )
        )
    return enriched

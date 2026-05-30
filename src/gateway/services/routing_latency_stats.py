from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import RouteTrace
from gateway.services.routing_request_analysis import int_config


def _attempt_model_key(attempt: Mapping[str, Any]) -> str | None:
    model_key = attempt.get("model_key")
    if isinstance(model_key, str) and model_key:
        return model_key

    provider = attempt.get("provider")
    model = attempt.get("model")
    if isinstance(provider, str) and provider and isinstance(model, str) and model:
        return f"{provider}:{model}"
    return None


def _attempt_duration_ms(attempt: Mapping[str, Any]) -> float | None:
    duration = attempt.get("duration_ms")
    if isinstance(duration, bool):
        return None
    if isinstance(duration, int | float) and duration >= 0:
        return float(duration)
    return None


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

    sample_limit = int_config(config.get("latency_sample_limit"), 200)
    min_samples = int_config(config.get("latency_min_samples"), 1)
    result = await db.execute(
        select(RouteTrace)
        .where(RouteTrace.status == "success")
        .order_by(RouteTrace.timestamp.desc())
        .limit(sample_limit)
    )
    durations_by_model: dict[str, list[float]] = {model: [] for model in candidate_models}
    for trace in result.scalars().all():
        attempts = trace.attempts if isinstance(trace.attempts, list) else []
        for attempt in attempts:
            if not isinstance(attempt, dict) or attempt.get("status") != "success":
                continue
            model_key = _attempt_model_key(attempt)
            duration_ms = _attempt_duration_ms(attempt)
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

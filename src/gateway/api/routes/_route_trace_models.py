"""Response models and summary helpers for route trace routes."""

from typing import Any

from pydantic import BaseModel

from gateway.api.routes._response_datetime import datetime_isoformat
from gateway.api.routes._summary_buckets import add_status_counts, new_status_bucket, summary_bucket
from gateway.models.entities import RouteTrace
from gateway.services.routing_trace_attempts import attempt_duration_ms


class RouteTraceResponse(BaseModel):
    """Response model for route trace information."""

    trace_id: str
    timestamp: str
    api_key_id: str | None
    user_id: str | None
    project_id: str | None
    policy_id: str | None
    requested_model: str
    endpoint: str
    selected_model: str | None
    selected_provider: str | None
    strategy: str | None
    status: str
    error_message: str | None
    selected_reason: str | None
    estimated_prompt_tokens: int | None
    estimated_output_tokens: int | None
    estimated_cost: float | None
    fallback_enabled: bool
    policy_source: str | None
    tags: dict[str, Any]
    guardrails: dict[str, Any]
    context: dict[str, Any]
    candidates: list[dict[str, Any]]
    attempts: list[dict[str, Any]]

    @classmethod
    def from_model(cls, trace: RouteTrace) -> "RouteTraceResponse":
        """Create a response from an ORM model."""
        return cls(
            trace_id=trace.trace_id,
            timestamp=datetime_isoformat(trace.timestamp),
            api_key_id=trace.api_key_id,
            user_id=trace.user_id,
            project_id=trace.project_id,
            policy_id=trace.policy_id,
            requested_model=trace.requested_model,
            endpoint=trace.endpoint,
            selected_model=trace.selected_model,
            selected_provider=trace.selected_provider,
            strategy=trace.strategy,
            status=trace.status,
            error_message=trace.error_message,
            selected_reason=trace.selected_reason,
            estimated_prompt_tokens=trace.estimated_prompt_tokens,
            estimated_output_tokens=trace.estimated_output_tokens,
            estimated_cost=trace.estimated_cost,
            fallback_enabled=bool(trace.fallback_enabled),
            policy_source=trace.policy_source,
            tags=trace.tag_dict(),
            guardrails=trace.guardrail_dict(),
            context=trace.context_dict(),
            candidates=trace.candidate_list(),
            attempts=trace.attempt_list(),
        )


class RouteTraceSummaryBucket(BaseModel):
    """Aggregated route trace metrics for one grouping key."""

    key: str
    count: int
    success_count: int
    error_count: int
    estimated_cost: float
    average_latency_ms: float | None


class RouteTraceSummaryResponse(BaseModel):
    """Aggregated route trace metrics."""

    total_count: int
    success_count: int
    error_count: int
    estimated_cost: float
    average_latency_ms: float | None
    by_model: list[RouteTraceSummaryBucket]
    by_policy: list[RouteTraceSummaryBucket]
    by_policy_source: list[RouteTraceSummaryBucket]
    by_endpoint: list[RouteTraceSummaryBucket]
    by_provider: list[RouteTraceSummaryBucket]
    by_strategy: list[RouteTraceSummaryBucket]


def _new_bucket(key: str) -> dict[str, Any]:
    return new_status_bucket(key, estimated_cost=0.0, latency_total_ms=0.0, latency_count=0)


def _add_trace_to_bucket(bucket: dict[str, Any], trace: RouteTrace, latency_ms: float | None) -> None:
    add_status_counts(bucket, trace.status)
    if trace.estimated_cost:
        bucket["estimated_cost"] += trace.estimated_cost
    if latency_ms is not None:
        bucket["latency_total_ms"] += latency_ms
        bucket["latency_count"] += 1


def _bucket_response(bucket: dict[str, Any]) -> RouteTraceSummaryBucket:
    latency_count = int(bucket["latency_count"])
    average_latency_ms = None
    if latency_count:
        average_latency_ms = float(bucket["latency_total_ms"]) / latency_count
    return RouteTraceSummaryBucket(
        key=str(bucket["key"]),
        count=int(bucket["count"]),
        success_count=int(bucket["success_count"]),
        error_count=int(bucket["error_count"]),
        estimated_cost=float(bucket["estimated_cost"]),
        average_latency_ms=average_latency_ms,
    )


def summarize_route_trace_logs(traces: list[RouteTrace]) -> RouteTraceSummaryResponse:
    model_buckets: dict[str, dict[str, Any]] = {}
    policy_buckets: dict[str, dict[str, Any]] = {}
    policy_source_buckets: dict[str, dict[str, Any]] = {}
    endpoint_buckets: dict[str, dict[str, Any]] = {}
    provider_buckets: dict[str, dict[str, Any]] = {}
    strategy_buckets: dict[str, dict[str, Any]] = {}
    total_bucket = _new_bucket("__total__")

    for trace in traces:
        latency_ms = None
        for attempt in trace.attempt_list():
            if not isinstance(attempt, dict) or attempt.get("status") != "success":
                continue
            duration = attempt_duration_ms(attempt)
            if duration is not None:
                latency_ms = duration
                break
        _add_trace_to_bucket(total_bucket, trace, latency_ms)

        for buckets, key in zip(
            (
                model_buckets,
                policy_buckets,
                policy_source_buckets,
                endpoint_buckets,
                provider_buckets,
                strategy_buckets,
            ),
            (
                trace.selected_model or "unknown",
                trace.policy_id or "unknown",
                trace.policy_source or "unknown",
                trace.endpoint or "unknown",
                trace.selected_provider or "unknown",
                trace.strategy or "unknown",
            ),
            strict=True,
        ):
            bucket = summary_bucket(buckets, key, _new_bucket)
            _add_trace_to_bucket(bucket, trace, latency_ms)

    total = _bucket_response(total_bucket)
    return RouteTraceSummaryResponse(
        total_count=total.count,
        success_count=total.success_count,
        error_count=total.error_count,
        estimated_cost=total.estimated_cost,
        average_latency_ms=total.average_latency_ms,
        by_model=[_bucket_response(bucket) for bucket in model_buckets.values()],
        by_policy=[_bucket_response(bucket) for bucket in policy_buckets.values()],
        by_policy_source=[_bucket_response(bucket) for bucket in policy_source_buckets.values()],
        by_endpoint=[_bucket_response(bucket) for bucket in endpoint_buckets.values()],
        by_provider=[_bucket_response(bucket) for bucket in provider_buckets.values()],
        by_strategy=[_bucket_response(bucket) for bucket in strategy_buckets.values()],
    )

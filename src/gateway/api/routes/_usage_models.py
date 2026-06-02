"""Response models and summary helpers for usage routes."""

from typing import Any, Self

from pydantic import BaseModel

from gateway.api.deps import _as_utc
from gateway.api.routes._response_datetime import datetime_isoformat
from gateway.api.routes._summary_buckets import add_status_counts, new_status_bucket, summary_bucket
from gateway.models.entities import UsageLog


class UsageEntry(BaseModel):
    """A single usage log entry."""

    id: str
    user_id: str | None
    api_key_id: str | None
    project_id: str | None
    timestamp: str
    model: str
    provider: str | None
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost: float | None
    status: str
    error_message: str | None
    tags: dict[str, Any]

    @classmethod
    def from_model(cls, log: UsageLog) -> Self:
        return cls(
            id=log.id,
            user_id=log.user_id,
            api_key_id=log.api_key_id,
            project_id=log.project_id,
            timestamp=datetime_isoformat(_as_utc(log.timestamp) or log.timestamp),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cost=log.cost,
            status=log.status,
            error_message=log.error_message,
            tags=log.tag_dict(),
        )


class UsageSummaryBucket(BaseModel):
    """Aggregated usage metrics for one grouping key."""

    key: str
    count: int
    success_count: int
    error_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class UsageSummaryResponse(BaseModel):
    """Aggregated usage metrics."""

    total_count: int
    success_count: int
    error_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    by_project: list[UsageSummaryBucket]
    by_user: list[UsageSummaryBucket]
    by_model: list[UsageSummaryBucket]
    by_provider: list[UsageSummaryBucket]
    by_endpoint: list[UsageSummaryBucket]
    by_status: list[UsageSummaryBucket]
    by_tag: list[UsageSummaryBucket]


def matches_tag_filter(log: UsageLog, tag_key: str | None, tag_value: str | None) -> bool:
    if tag_key is None:
        return True
    tags = log.tag_dict()
    if tag_key not in tags:
        return False
    if tag_value is None:
        return True
    return str(tags[tag_key]) == tag_value


def _new_bucket(key: str) -> dict[str, Any]:
    return new_status_bucket(key, prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0)


def _add_log_to_bucket(bucket: dict[str, Any], log: UsageLog) -> None:
    add_status_counts(bucket, log.status)
    bucket["prompt_tokens"] += log.prompt_tokens or 0
    bucket["completion_tokens"] += log.completion_tokens or 0
    bucket["total_tokens"] += log.total_tokens or 0
    bucket["cost"] += log.cost or 0.0


def _bucket_response(bucket: dict[str, Any]) -> UsageSummaryBucket:
    return UsageSummaryBucket(
        key=str(bucket["key"]),
        count=int(bucket["count"]),
        success_count=int(bucket["success_count"]),
        error_count=int(bucket["error_count"]),
        prompt_tokens=int(bucket["prompt_tokens"]),
        completion_tokens=int(bucket["completion_tokens"]),
        total_tokens=int(bucket["total_tokens"]),
        cost=float(bucket["cost"]),
    )


def _bucket_responses(buckets: dict[str, dict[str, Any]]) -> list[UsageSummaryBucket]:
    responses = [_bucket_response(bucket) for bucket in buckets.values()]
    return sorted(responses, key=lambda bucket: (-bucket.cost, -bucket.count, bucket.key))


def summarize_usage_logs(logs: list[UsageLog]) -> UsageSummaryResponse:
    total_bucket = _new_bucket("__total__")
    project_buckets: dict[str, dict[str, Any]] = {}
    user_buckets: dict[str, dict[str, Any]] = {}
    model_buckets: dict[str, dict[str, Any]] = {}
    provider_buckets: dict[str, dict[str, Any]] = {}
    endpoint_buckets: dict[str, dict[str, Any]] = {}
    status_buckets: dict[str, dict[str, Any]] = {}
    tag_buckets: dict[str, dict[str, Any]] = {}

    for log in logs:
        _add_log_to_bucket(total_bucket, log)
        for buckets, key in zip(
            (
                project_buckets,
                user_buckets,
                model_buckets,
                provider_buckets,
                endpoint_buckets,
                status_buckets,
            ),
            (
                log.project_id or "unknown",
                log.user_id or "unknown",
                log.model or "unknown",
                log.provider or "unknown",
                log.endpoint or "unknown",
                log.status or "unknown",
            ),
            strict=True,
        ):
            bucket = summary_bucket(buckets, key, _new_bucket)
            _add_log_to_bucket(bucket, log)

        for key, value in log.tag_dict().items():
            tag_bucket_key = f"{key}={value}"
            bucket = summary_bucket(tag_buckets, tag_bucket_key, _new_bucket)
            _add_log_to_bucket(bucket, log)

    total = _bucket_response(total_bucket)
    return UsageSummaryResponse(
        total_count=total.count,
        success_count=total.success_count,
        error_count=total.error_count,
        prompt_tokens=total.prompt_tokens,
        completion_tokens=total.completion_tokens,
        total_tokens=total.total_tokens,
        cost=total.cost,
        by_project=_bucket_responses(project_buckets),
        by_user=_bucket_responses(user_buckets),
        by_model=_bucket_responses(model_buckets),
        by_provider=_bucket_responses(provider_buckets),
        by_endpoint=_bucket_responses(endpoint_buckets),
        by_status=_bucket_responses(status_buckets),
        by_tag=_bucket_responses(tag_buckets),
    )

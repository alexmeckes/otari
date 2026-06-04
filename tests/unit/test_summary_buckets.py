from typing import Any

from gateway.api.routes._summary_buckets import add_status_counts, new_status_bucket, summary_bucket


def test_status_bucket_starts_with_shared_counts_and_custom_metrics() -> None:
    assert new_status_bucket("model-a", total_tokens=0, cost=0.0) == {
        "key": "model-a",
        "count": 0,
        "success_count": 0,
        "error_count": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }


def test_add_status_counts_tracks_success_and_error_only() -> None:
    bucket = new_status_bucket("total")

    add_status_counts(bucket, "success")
    add_status_counts(bucket, "error")
    add_status_counts(bucket, "pending")

    assert bucket["count"] == 3
    assert bucket["success_count"] == 1
    assert bucket["error_count"] == 1


def test_summary_bucket_reuses_existing_bucket_without_creating_another() -> None:
    existing = new_status_bucket("model-a", cost=1.0)
    buckets = {"model-a": existing}
    created: list[str] = []

    def create_bucket(key: str) -> dict[str, Any]:
        created.append(key)
        return new_status_bucket(key, cost=0.0)

    assert summary_bucket(buckets, "model-a", create_bucket) is existing
    assert created == []


def test_summary_bucket_creates_missing_bucket() -> None:
    buckets: dict[str, dict[str, Any]] = {}

    bucket = summary_bucket(buckets, "model-b", lambda key: new_status_bucket(key, cost=0.0))

    assert bucket == {
        "key": "model-b",
        "count": 0,
        "success_count": 0,
        "error_count": 0,
        "cost": 0.0,
    }
    assert buckets["model-b"] is bucket

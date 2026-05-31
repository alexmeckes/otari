from gateway.api.routes._summary_buckets import add_status_counts, new_status_bucket


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

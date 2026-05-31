"""Shared helpers for API summary bucket aggregation."""

from typing import Any


def new_status_bucket(key: str, **metrics: Any) -> dict[str, Any]:
    return {
        "key": key,
        "count": 0,
        "success_count": 0,
        "error_count": 0,
        **metrics,
    }


def add_status_counts(bucket: dict[str, Any], status: str | None) -> None:
    bucket["count"] += 1
    if status == "success":
        bucket["success_count"] += 1
    elif status == "error":
        bucket["error_count"] += 1

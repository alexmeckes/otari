"""Shared helpers for budget period windows."""

from datetime import UTC, datetime, timedelta


def calculate_next_reset(start: datetime, duration_sec: int) -> datetime:
    """Calculate the next budget reset time from a period start."""
    return start + timedelta(seconds=duration_sec)


def budget_period_window(
    duration_sec: int | None,
    start: datetime | None = None,
) -> tuple[datetime, datetime | None]:
    started_at = start or datetime.now(UTC)
    next_reset_at = calculate_next_reset(started_at, duration_sec) if duration_sec else None
    return started_at, next_reset_at


def as_utc(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)

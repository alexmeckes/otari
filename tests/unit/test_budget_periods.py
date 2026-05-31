from datetime import UTC, datetime, timedelta, timezone

from gateway.services.budget_periods import as_utc, budget_period_window, calculate_next_reset


def test_calculate_next_reset_adds_seconds() -> None:
    start = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)

    assert calculate_next_reset(start, 60) == start + timedelta(seconds=60)


def test_budget_period_window_uses_supplied_start() -> None:
    start = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)

    assert budget_period_window(3600, start) == (start, start + timedelta(seconds=3600))
    assert budget_period_window(None, start) == (start, None)


def test_budget_period_window_defaults_to_aware_now() -> None:
    started_at, next_reset_at = budget_period_window(60)

    assert started_at.tzinfo is UTC
    assert next_reset_at == started_at + timedelta(seconds=60)


def test_as_utc_preserves_none_and_aware_values() -> None:
    aware = datetime(2026, 5, 31, 8, 0, tzinfo=timezone(timedelta(hours=-4)))

    assert as_utc(None) is None
    assert as_utc(aware) is aware


def test_as_utc_marks_naive_values_as_utc() -> None:
    naive = datetime(2026, 5, 31, 12, 0)

    assert as_utc(naive) == datetime(2026, 5, 31, 12, 0, tzinfo=UTC)

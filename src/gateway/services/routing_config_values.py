"""Shared config value parsers for routing services."""

from typing import Any


def dict_or_empty(value: Any, *, copy_value: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if copy_value:
        return dict(value)
    return value


def float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def non_negative_float_or_none(value: Any) -> float | None:
    parsed = float_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def score_or_none(value: Any) -> float | None:
    parsed = non_negative_float_or_none(value)
    if parsed is None:
        return None
    if parsed <= 1.0:
        return parsed
    if parsed <= 100.0:
        return parsed / 100.0
    return None

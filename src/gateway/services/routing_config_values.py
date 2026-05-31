"""Shared config value parsers for routing services."""

from typing import Any


def dict_or_empty(value: Any, *, copy_value: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if copy_value:
        return dict(value)
    return value


def string_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def int_config(value: Any, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return default


def non_negative_int_config(value: Any, default: int) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    return default


def bool_config(value: Any, default: bool, *, coerce_strings: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if coerce_strings and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


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

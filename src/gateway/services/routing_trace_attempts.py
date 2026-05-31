"""Shared accessors for stored route-trace attempt payloads."""

from collections.abc import Mapping
from typing import Any


def attempt_model_key(attempt: Mapping[str, Any]) -> str | None:
    model_key = attempt.get("model_key")
    if isinstance(model_key, str) and model_key:
        return model_key

    provider = attempt.get("provider")
    model = attempt.get("model")
    if isinstance(provider, str) and provider and isinstance(model, str) and model:
        return f"{provider}:{model}"
    return None


def attempt_duration_ms(attempt: Mapping[str, Any]) -> float | None:
    duration = attempt.get("duration_ms")
    if isinstance(duration, bool):
        return None
    if isinstance(duration, int | float) and duration >= 0:
        return float(duration)
    return None


def attempt_provider(attempt: Mapping[str, Any]) -> str | None:
    provider = attempt.get("provider")
    return provider if isinstance(provider, str) and provider else None


def attempt_outcome(attempt: Mapping[str, Any]) -> str | None:
    status = attempt.get("status")
    if status in {"success", "error"}:
        return str(status)
    return None

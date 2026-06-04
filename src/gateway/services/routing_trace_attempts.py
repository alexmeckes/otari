"""Shared accessors for stored route-trace attempt payloads."""

from collections.abc import Mapping
from typing import Any

from gateway.services.pricing_service import pricing_model_ref


def attempt_model_key(attempt: Mapping[str, Any]) -> str | None:
    if isinstance(model_key := attempt.get("model_key"), str) and model_key:
        return model_key

    if (
        isinstance(provider := attempt.get("provider"), str)
        and provider
        and isinstance(model := attempt.get("model"), str)
        and model
    ):
        return pricing_model_ref(provider, model)
    return None


def attempt_duration_ms(attempt: Mapping[str, Any]) -> float | None:
    duration = attempt.get("duration_ms")
    if isinstance(duration, bool) or not isinstance(duration, int | float) or duration < 0:
        return None
    return float(duration)


def attempt_provider(attempt: Mapping[str, Any]) -> str | None:
    return provider if isinstance(provider := attempt.get("provider"), str) and provider else None


def attempt_outcome(attempt: Mapping[str, Any]) -> str | None:
    return str(status) if (status := attempt.get("status")) in {"success", "error"} else None

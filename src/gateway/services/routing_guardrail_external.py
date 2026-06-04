"""External guardrail classifier helpers."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from gateway.services.routing_config_values import (
    bool_config,
    coerced_string_or_none,
    non_negative_float_or_none,
    string_or_none,
)
from gateway.services.routing_guardrail_helpers import guardrail_violation

ExternalClassifierPostResult = tuple[int | None, dict[str, Any] | None, str | None]
ExternalClassifierPost = Callable[..., Awaitable[ExternalClassifierPostResult]]


@dataclass(frozen=True)
class _ClassifierSettings:
    name: str
    url: str | None
    timeout_seconds: float
    threshold: float | None
    headers: dict[str, str] | None
    fail_closed: bool


async def post_external_guardrail_classifier(
    *,
    url: str,
    request_text: str,
    timeout_seconds: float,
    headers: dict[str, str] | None,
) -> ExternalClassifierPostResult:
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, json={"text": request_text}, headers=headers)
    except httpx.HTTPError as exc:
        return None, None, str(exc)

    try:
        parsed = response.json()
    except ValueError:
        payload = None
    else:
        payload = parsed if isinstance(parsed, dict) else None
    if not response.is_success:
        error = response.text
        if payload is not None:
            detail = string_or_none(payload.get("detail")) or string_or_none(payload.get("error"))
            if detail is not None:
                error = detail
        return response.status_code, payload, f"HTTP {response.status_code}: {error}"
    if payload is None:
        return response.status_code, None, "classifier returned non-object JSON"
    return response.status_code, payload, None


def _classifier_rule(value: Any, *, fallback: str) -> str:
    rule = string_or_none(value)
    if rule is not None:
        return rule
    if isinstance(value, dict):
        for key in ("rule", "type", "label", "category", "name"):
            item = string_or_none(value.get(key))
            if item is not None:
                return item
    return fallback


def _classifier_payload_flagged(
    settings: _ClassifierSettings,
    payload: Mapping[str, Any],
    score: float | None,
) -> bool:
    return (
        payload.get("blocked") is True
        or payload.get("flagged") is True
        or (settings.threshold is not None and score is not None and score >= settings.threshold)
    )


def _classifier_violations(
    settings: _ClassifierSettings,
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, str]], float | None]:
    raw_violations = payload.get("violations")
    violations = [
        guardrail_violation("external_classifier", _classifier_rule(item, fallback=settings.name))
        for item in raw_violations
    ] if isinstance(raw_violations, list) else []
    score = non_negative_float_or_none(payload.get("score"))
    if violations or not _classifier_payload_flagged(settings, payload, score):
        return violations, score
    return [
        guardrail_violation("external_classifier", _classifier_rule(payload.get("label"), fallback=settings.name))
    ], score


def _classifier_settings(classifier: Mapping[str, Any], *, index: int) -> _ClassifierSettings:
    raw_headers = classifier.get("headers")
    headers = None
    if isinstance(raw_headers, dict):
        headers = {
            str(key): str(header_value)
            for key, header_value in raw_headers.items()
            if coerced_string_or_none(key) is not None
        } or None

    return _ClassifierSettings(
        name=string_or_none(classifier.get("name")) or f"classifier_{index}",
        url=string_or_none(classifier.get("url")),
        timeout_seconds=non_negative_float_or_none(classifier.get("timeout_seconds")) or 2.0,
        threshold=non_negative_float_or_none(classifier.get("threshold")),
        headers=headers,
        fail_closed=bool_config(classifier.get("fail_closed"), False),
    )


async def _evaluate_classifier_from_settings(
    settings: _ClassifierSettings,
    *,
    request_text: str,
    post_classifier: ExternalClassifierPost,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if settings.url is None:
        return [], {"name": settings.name, "status": "skipped", "reason": "missing_url"}

    status_code, payload, error = await post_classifier(
        url=settings.url,
        request_text=request_text,
        timeout_seconds=settings.timeout_seconds,
        headers=settings.headers,
    )
    if error is not None:
        return (
            [guardrail_violation("external_classifier_error", settings.name)] if settings.fail_closed else [],
            {
                "name": settings.name,
                "status": "error",
                "status_code": status_code,
                "error": error[:200],
                "fail_closed": settings.fail_closed,
            },
        )

    assert payload is not None
    violations, score = _classifier_violations(settings, payload)
    return violations, {
        "name": settings.name,
        "status": "flagged" if violations else "passed",
        "status_code": status_code,
        "score": score,
        "threshold": settings.threshold,
        "label": label if isinstance(label := payload.get("label"), str) else None,
        "violations": violations,
    }


def _external_classifier_configs(guardrails: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    external_classifiers = guardrails.get("external_classifiers")
    if isinstance(external_classifiers, list):
        return [classifier for classifier in external_classifiers if isinstance(classifier, dict)]
    if isinstance(external_classifiers, dict):
        return [external_classifiers]
    return []


async def evaluate_external_classifiers(
    *,
    guardrails: Mapping[str, Any],
    request_text: str,
    post_classifier: ExternalClassifierPost,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    evaluations = [
        await _evaluate_classifier_from_settings(
            _classifier_settings(classifier, index=index),
            request_text=request_text,
            post_classifier=post_classifier,
        )
        for index, classifier in enumerate(_external_classifier_configs(guardrails), start=1)
    ]
    return (
        [violation for classifier_violations, _ in evaluations for violation in classifier_violations],
        [classifier_result for _, classifier_result in evaluations],
    )

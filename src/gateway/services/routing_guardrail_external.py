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
ExternalClassifierEvaluation = tuple[list[dict[str, str]], dict[str, Any]]


@dataclass(frozen=True)
class _ClassifierSettings:
    name: str
    url: str | None
    timeout_seconds: float
    threshold: float | None
    headers: dict[str, str] | None
    fail_closed: bool


def _classifier_http_error_text(response: httpx.Response, payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return response.text
    return string_or_none(payload.get("detail")) or string_or_none(payload.get("error")) or response.text


def _classifier_response_payload(response: httpx.Response) -> dict[str, Any] | None:
    try:
        parsed = response.json()
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


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

    payload = _classifier_response_payload(response)
    if not response.is_success:
        error = _classifier_http_error_text(response, payload)
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


def _classifier_headers(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    return {
        str(key): str(header_value)
        for key, header_value in value.items()
        if coerced_string_or_none(key) is not None
    } or None


def _classifier_explicit_violations(
    settings: _ClassifierSettings,
    payload: Mapping[str, Any],
) -> list[dict[str, str]]:
    raw_violations = payload.get("violations")
    if not isinstance(raw_violations, list):
        return []
    return [
        guardrail_violation("external_classifier", _classifier_rule(item, fallback=settings.name))
        for item in raw_violations
    ]


def _classifier_violations(
    settings: _ClassifierSettings,
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, str]], float | None]:
    violations = _classifier_explicit_violations(settings, payload)
    score = non_negative_float_or_none(payload.get("score"))
    if violations or not _classifier_payload_flagged(settings, payload, score):
        return violations, score
    return [
        guardrail_violation("external_classifier", _classifier_rule(payload.get("label"), fallback=settings.name))
    ], score


def _classifier_settings(classifier: Mapping[str, Any], *, index: int) -> _ClassifierSettings:
    return _ClassifierSettings(
        name=string_or_none(classifier.get("name")) or f"classifier_{index}",
        url=string_or_none(classifier.get("url")),
        timeout_seconds=non_negative_float_or_none(classifier.get("timeout_seconds")) or 2.0,
        threshold=non_negative_float_or_none(classifier.get("threshold")),
        headers=_classifier_headers(classifier.get("headers")),
        fail_closed=bool_config(classifier.get("fail_closed"), False),
    )


def _classifier_error_evaluation(
    settings: _ClassifierSettings,
    *,
    status_code: int | None,
    error: str,
) -> ExternalClassifierEvaluation:
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


def _classifier_success_evaluation(
    settings: _ClassifierSettings,
    *,
    status_code: int | None,
    payload: Mapping[str, Any],
) -> ExternalClassifierEvaluation:
    violations, score = _classifier_violations(settings, payload)
    label = payload.get("label")
    return violations, {
        "name": settings.name,
        "status": "flagged" if violations else "passed",
        "status_code": status_code,
        "score": score,
        "threshold": settings.threshold,
        "label": label if isinstance(label, str) else None,
        "violations": violations,
    }


async def _evaluate_classifier_from_settings(
    settings: _ClassifierSettings,
    *,
    request_text: str,
    post_classifier: ExternalClassifierPost,
) -> ExternalClassifierEvaluation:
    if settings.url is None:
        return [], {"name": settings.name, "status": "skipped", "reason": "missing_url"}

    status_code, payload, error = await post_classifier(
        url=settings.url,
        request_text=request_text,
        timeout_seconds=settings.timeout_seconds,
        headers=settings.headers,
    )
    if error is not None:
        return _classifier_error_evaluation(settings, status_code=status_code, error=error)

    assert payload is not None
    return _classifier_success_evaluation(settings, status_code=status_code, payload=payload)


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

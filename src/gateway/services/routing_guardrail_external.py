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
_ClassifierEvaluationResult = tuple[list[dict[str, str]], dict[str, Any]]

_CLASSIFIER_RULE_KEYS = ("rule", "type", "label", "category", "name")
_CLASSIFIER_ERROR_TEXT_LIMIT = 200
_CLASSIFIER_DEFAULT_TIMEOUT_SECONDS = 2.0
_CLASSIFIER_NON_OBJECT_JSON_ERROR = "classifier returned non-object JSON"
_CLASSIFIER_MISSING_URL_REASON = "missing_url"


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

    payload = _classifier_response_payload(response)
    if not response.is_success:
        return response.status_code, payload, _classifier_http_error_text(response, payload)
    if payload is None:
        return response.status_code, None, _CLASSIFIER_NON_OBJECT_JSON_ERROR
    return response.status_code, payload, None


def _classifier_response_payload(response: httpx.Response) -> dict[str, Any] | None:
    try:
        parsed = response.json()
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _classifier_http_error_text(response: httpx.Response, payload: Mapping[str, Any] | None) -> str:
    error = response.text
    if payload is not None:
        detail = string_or_none(payload.get("detail")) or string_or_none(payload.get("error"))
        if detail is not None:
            error = detail
    return f"HTTP {response.status_code}: {error}"


def _classifier_rule(value: Any, *, fallback: str) -> str:
    rule = string_or_none(value)
    if rule is not None:
        return rule
    if isinstance(value, dict):
        for key in _CLASSIFIER_RULE_KEYS:
            item = string_or_none(value.get(key))
            if item is not None:
                return item
    return fallback


def _classifier_headers(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    return {
        str(key): str(header_value)
        for key, header_value in value.items()
        if coerced_string_or_none(key) is not None
    } or None


def _classifier_configs(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [classifier for classifier in value if isinstance(classifier, dict)]
    return [value] if isinstance(value, dict) else []


def _classifier_flagged(
    payload: Mapping[str, Any],
    *,
    score: float | None,
    threshold: float | None,
) -> bool:
    return (
        payload.get("blocked") is True
        or payload.get("flagged") is True
        or (threshold is not None and score is not None and score >= threshold)
    )


def _explicit_classifier_violations(settings: _ClassifierSettings, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_violations = payload.get("violations")
    return [
        guardrail_violation("external_classifier", _classifier_rule(item, fallback=settings.name))
        for item in raw_violations
    ] if isinstance(raw_violations, list) else []


def _classifier_violations(
    settings: _ClassifierSettings,
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, str]], float | None]:
    violations = _explicit_classifier_violations(settings, payload)
    score = non_negative_float_or_none(payload.get("score"))
    if violations or not _classifier_flagged(payload, score=score, threshold=settings.threshold):
        return violations, score
    return [
        guardrail_violation("external_classifier", _classifier_rule(payload.get("label"), fallback=settings.name))
    ], score


def _classifier_success_result(
    settings: _ClassifierSettings,
    *,
    status_code: int | None,
    payload: Mapping[str, Any],
    score: float | None,
    violations: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "name": settings.name,
        "status": "flagged" if violations else "passed",
        "status_code": status_code,
        "score": score,
        "threshold": settings.threshold,
        "label": label if isinstance(label := payload.get("label"), str) else None,
        "violations": violations,
    }


def _classifier_success_evaluation(
    settings: _ClassifierSettings,
    *,
    status_code: int | None,
    payload: Mapping[str, Any],
) -> _ClassifierEvaluationResult:
    violations, score = _classifier_violations(settings, payload)
    return violations, _classifier_success_result(
        settings,
        status_code=status_code,
        payload=payload,
        score=score,
        violations=violations,
    )


def _classifier_error_result(
    settings: _ClassifierSettings,
    *,
    status_code: int | None,
    error: str,
) -> dict[str, Any]:
    return {
        "name": settings.name,
        "status": "error",
        "status_code": status_code,
        "error": error[:_CLASSIFIER_ERROR_TEXT_LIMIT],
        "fail_closed": settings.fail_closed,
    }


def _classifier_error_evaluation(
    settings: _ClassifierSettings,
    *,
    status_code: int | None,
    error: str,
) -> _ClassifierEvaluationResult:
    return (
        [guardrail_violation("external_classifier_error", settings.name)] if settings.fail_closed else [],
        _classifier_error_result(
            settings,
            status_code=status_code,
            error=error,
        ),
    )


def _classifier_settings(classifier: Mapping[str, Any], *, index: int) -> _ClassifierSettings:
    return _ClassifierSettings(
        name=string_or_none(classifier.get("name")) or f"classifier_{index}",
        url=string_or_none(classifier.get("url")),
        timeout_seconds=non_negative_float_or_none(classifier.get("timeout_seconds"))
        or _CLASSIFIER_DEFAULT_TIMEOUT_SECONDS,
        threshold=non_negative_float_or_none(classifier.get("threshold")),
        headers=_classifier_headers(classifier.get("headers")),
        fail_closed=bool_config(classifier.get("fail_closed"), False),
    )


async def _post_classifier_from_settings(
    settings: _ClassifierSettings,
    *,
    request_text: str,
    post_classifier: ExternalClassifierPost,
) -> ExternalClassifierPostResult:
    assert settings.url is not None
    return await post_classifier(
        url=settings.url,
        request_text=request_text,
        timeout_seconds=settings.timeout_seconds,
        headers=settings.headers,
    )


async def _evaluate_classifier_from_settings(
    settings: _ClassifierSettings,
    *,
    request_text: str,
    post_classifier: ExternalClassifierPost,
) -> _ClassifierEvaluationResult:
    if settings.url is None:
        return [], {"name": settings.name, "status": "skipped", "reason": _CLASSIFIER_MISSING_URL_REASON}

    status_code, payload, error = await _post_classifier_from_settings(
        settings,
        request_text=request_text,
        post_classifier=post_classifier,
    )
    if error is not None:
        return _classifier_error_evaluation(
            settings,
            status_code=status_code,
            error=error,
        )

    assert payload is not None
    return _classifier_success_evaluation(
        settings,
        status_code=status_code,
        payload=payload,
    )


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
        for index, classifier in enumerate(_classifier_configs(guardrails.get("external_classifiers")), start=1)
    ]
    return (
        [violation for classifier_violations, _ in evaluations for violation in classifier_violations],
        [classifier_result for _, classifier_result in evaluations],
    )

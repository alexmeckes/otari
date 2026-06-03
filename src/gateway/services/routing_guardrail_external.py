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

ExternalClassifierPost = Callable[
    ...,
    Awaitable[tuple[int | None, dict[str, Any] | None, str | None]],
]

_CLASSIFIER_RULE_KEYS = ("rule", "type", "label", "category", "name")


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
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, json={"text": request_text}, headers=headers)
    except httpx.HTTPError as exc:
        return None, None, str(exc)

    payload = _classifier_response_payload(response)
    if not response.is_success:
        return response.status_code, payload, _classifier_http_error_text(response, payload)
    if payload is None:
        return response.status_code, None, "classifier returned non-object JSON"
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
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [classifier for classifier in value if isinstance(classifier, dict)]
    return []


def _classifier_flagged(
    payload: Mapping[str, Any],
    *,
    score: float | None,
    threshold: float | None,
) -> bool:
    if payload.get("blocked") is True or payload.get("flagged") is True:
        return True
    return threshold is not None and score is not None and score >= threshold


def _explicit_classifier_violations(payload: Mapping[str, Any], *, name: str) -> list[dict[str, str]]:
    raw_violations = payload.get("violations")
    if not isinstance(raw_violations, list):
        return []
    return [
        guardrail_violation("external_classifier", _classifier_rule(item, fallback=name))
        for item in raw_violations
    ]


def _fallback_classifier_violation(payload: Mapping[str, Any], *, name: str) -> dict[str, str]:
    return guardrail_violation("external_classifier", _classifier_rule(payload.get("label"), fallback=name))


def _classifier_violations(
    payload: Mapping[str, Any],
    *,
    name: str,
    threshold: float | None,
) -> tuple[list[dict[str, str]], float | None]:
    violations = _explicit_classifier_violations(payload, name=name)
    score = non_negative_float_or_none(payload.get("score"))
    if _classifier_flagged(payload, score=score, threshold=threshold) and not violations:
        violations.append(_fallback_classifier_violation(payload, name=name))
    return violations, score


def _classifier_result_label(payload: Mapping[str, Any]) -> str | None:
    label = payload.get("label")
    if isinstance(label, str):
        return label
    return None


def _classifier_success_result(
    *,
    name: str,
    status_code: int | None,
    payload: Mapping[str, Any],
    score: float | None,
    threshold: float | None,
    violations: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "flagged" if violations else "passed",
        "status_code": status_code,
        "score": score,
        "threshold": threshold,
        "label": _classifier_result_label(payload),
        "violations": violations,
    }


def _classifier_error_text(error: str) -> str:
    return error[:200]


def _classifier_error_result(
    *,
    name: str,
    status_code: int | None,
    error: str,
    fail_closed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "error",
        "status_code": status_code,
        "error": _classifier_error_text(error),
        "fail_closed": fail_closed,
    }


def _classifier_skipped_result(name: str) -> dict[str, str]:
    return {"name": name, "status": "skipped", "reason": "missing_url"}


def _classifier_settings(classifier: Mapping[str, Any], *, index: int) -> _ClassifierSettings:
    return _ClassifierSettings(
        name=string_or_none(classifier.get("name")) or f"classifier_{index}",
        url=string_or_none(classifier.get("url")),
        timeout_seconds=non_negative_float_or_none(classifier.get("timeout_seconds")) or 2.0,
        threshold=non_negative_float_or_none(classifier.get("threshold")),
        headers=_classifier_headers(classifier.get("headers")),
        fail_closed=bool_config(classifier.get("fail_closed"), False),
    )


async def evaluate_external_classifiers(
    *,
    guardrails: Mapping[str, Any],
    request_text: str,
    post_classifier: ExternalClassifierPost,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    violations: list[dict[str, str]] = []
    classifier_results: list[dict[str, Any]] = []
    for index, classifier in enumerate(_classifier_configs(guardrails.get("external_classifiers")), start=1):
        settings = _classifier_settings(classifier, index=index)
        if settings.url is None:
            classifier_results.append(_classifier_skipped_result(settings.name))
            continue
        status_code, payload, error = await post_classifier(
            url=settings.url,
            request_text=request_text,
            timeout_seconds=settings.timeout_seconds,
            headers=settings.headers,
        )
        if error is not None:
            classifier_results.append(
                _classifier_error_result(
                    name=settings.name,
                    status_code=status_code,
                    error=error,
                    fail_closed=settings.fail_closed,
                )
            )
            if settings.fail_closed:
                violations.append(guardrail_violation("external_classifier_error", settings.name))
            continue
        assert payload is not None
        classifier_violations, score = _classifier_violations(
            payload,
            name=settings.name,
            threshold=settings.threshold,
        )
        violations.extend(classifier_violations)
        classifier_results.append(
            _classifier_success_result(
                name=settings.name,
                status_code=status_code,
                payload=payload,
                score=score,
                threshold=settings.threshold,
                violations=classifier_violations,
            )
        )
    return violations, classifier_results

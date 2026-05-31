"""External guardrail classifier helpers."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx

from gateway.services.routing_request_analysis import bool_config

ExternalClassifierPost = Callable[
    ...,
    Awaitable[tuple[int | None, dict[str, Any] | None, str | None]],
]


def _non_negative_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and value >= 0:
        return float(value)
    return None


def _guardrail_violation(kind: str, rule: str) -> dict[str, str]:
    return {"type": kind, "rule": rule}


def _external_classifier_configs(guardrails: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    classifiers = guardrails.get("external_classifiers")
    if isinstance(classifiers, dict):
        return [classifiers]
    if not isinstance(classifiers, list):
        return []
    return [classifier for classifier in classifiers if isinstance(classifier, dict)]


def _external_classifier_name(classifier: Mapping[str, Any], index: int) -> str:
    name = classifier.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"classifier_{index}"


def _external_classifier_headers(classifier: Mapping[str, Any]) -> dict[str, str] | None:
    headers = classifier.get("headers")
    if not isinstance(headers, dict):
        return None
    normalized = {str(key): str(value) for key, value in headers.items() if str(key).strip()}
    return normalized or None


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

    payload: dict[str, Any] | None = None
    try:
        parsed = response.json()
        if isinstance(parsed, dict):
            payload = parsed
    except ValueError:
        payload = None

    if response.status_code < 200 or response.status_code >= 300:
        error = response.text
        if payload is not None:
            detail = payload.get("detail") or payload.get("error")
            if isinstance(detail, str) and detail.strip():
                error = detail.strip()
        return response.status_code, payload, f"HTTP {response.status_code}: {error}"
    if payload is None:
        return response.status_code, None, "classifier returned non-object JSON"
    return response.status_code, payload, None


def _classifier_rule(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("rule", "type", "label", "category", "name"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
    return fallback


def _classifier_violations(name: str, payload: Mapping[str, Any], threshold: float | None) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    raw_violations = payload.get("violations")
    if isinstance(raw_violations, list):
        for item in raw_violations:
            violations.append(_guardrail_violation("external_classifier", _classifier_rule(item, fallback=name)))

    flagged = payload.get("blocked") is True or payload.get("flagged") is True
    score = _non_negative_float_or_none(payload.get("score"))
    if threshold is not None and score is not None and score >= threshold:
        flagged = True
    if flagged and not violations:
        label = _classifier_rule(payload.get("label"), fallback=name)
        violations.append(_guardrail_violation("external_classifier", label))
    return violations


async def evaluate_external_classifiers(
    *,
    guardrails: Mapping[str, Any],
    request_text: str,
    post_classifier: ExternalClassifierPost,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    violations: list[dict[str, str]] = []
    classifier_results: list[dict[str, Any]] = []
    for index, classifier in enumerate(_external_classifier_configs(guardrails), start=1):
        name = _external_classifier_name(classifier, index)
        url = classifier.get("url")
        if not isinstance(url, str) or not url.strip():
            classifier_results.append({"name": name, "status": "skipped", "reason": "missing_url"})
            continue
        timeout_seconds = _non_negative_float_or_none(classifier.get("timeout_seconds")) or 2.0
        threshold = _non_negative_float_or_none(classifier.get("threshold"))
        status_code, payload, error = await post_classifier(
            url=url.strip(),
            request_text=request_text,
            timeout_seconds=timeout_seconds,
            headers=_external_classifier_headers(classifier),
        )
        fail_closed = bool_config(classifier.get("fail_closed"), False)
        if error is not None:
            classifier_results.append(
                {
                    "name": name,
                    "status": "error",
                    "status_code": status_code,
                    "error": error[:200],
                    "fail_closed": fail_closed,
                }
            )
            if fail_closed:
                violations.append(_guardrail_violation("external_classifier_error", name))
            continue
        assert payload is not None
        classifier_violations = _classifier_violations(name, payload, threshold)
        violations.extend(classifier_violations)
        score = _non_negative_float_or_none(payload.get("score"))
        label = payload.get("label") if isinstance(payload.get("label"), str) else None
        classifier_results.append(
            {
                "name": name,
                "status": "flagged" if classifier_violations else "passed",
                "status_code": status_code,
                "score": score,
                "threshold": threshold,
                "label": label,
                "violations": classifier_violations,
            }
        )
    return violations, classifier_results

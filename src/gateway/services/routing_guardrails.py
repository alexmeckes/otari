"""Guardrail evaluation and redaction helpers for routing policies."""

import copy
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import httpx

from gateway.log_config import logger
from gateway.services.routing_request_analysis import bool_config, jsonable_text

ExternalClassifierPost = Callable[
    ...,
    Awaitable[tuple[int | None, dict[str, Any] | None, str | None]],
]

_GUARDRAIL_ACTIONS = {"block", "observe"}
_PROMPT_INJECTION_PHRASES = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard previous instructions",
    "reveal the system prompt",
    "show me the system prompt",
    "print the system prompt",
    "developer message",
    "jailbreak",
)
_PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}
_CREDENTIAL_BLOCKED_PATTERNS = (
    {"name": "openai_api_key", "pattern": r"\bsk-[A-Za-z0-9_-]{20,}\b"},
    {"name": "aws_access_key_id", "pattern": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"},
    {"name": "github_token", "pattern": r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"},
    {"name": "slack_token", "pattern": r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"},
    {"name": "private_key_block", "pattern": r"-----BEGIN [A-Z ]*PRIVATE KEY-----"},
)
_GUARDRAIL_PRESETS: dict[str, dict[str, Any]] = {
    "prompt_injection": {"prompt_injection": {"enabled": True}},
    "pii": {"pii": {"enabled": True}},
    "credential_leak": {"blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS)},
    "secrets": {"blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS)},
    "dlp": {
        "pii": {"enabled": True},
        "blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS),
    },
    "baseline": {
        "prompt_injection": {"enabled": True},
        "blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS),
    },
    "strict": {
        "pii": {"enabled": True},
        "prompt_injection": {"enabled": True},
        "blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS),
    },
}
_GUARDRAIL_PRESET_ALIASES = {
    "credentials": "credential_leak",
    "credential": "credential_leak",
    "secret": "secrets",
    "data_loss_prevention": "dlp",
    "data_loss": "dlp",
    "prompt_shield": "prompt_injection",
    "prompt_injection_detection": "prompt_injection",
}


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _non_negative_float_or_none(value: Any) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _guardrails_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    guardrails = config.get("guardrails")
    return guardrails if isinstance(guardrails, dict) else {}


def _normalize_guardrail_preset_name(value: Any) -> str | None:
    name: Any = value
    if isinstance(value, dict):
        name = value.get("name") or value.get("preset")
    if not isinstance(name, str) or not name.strip():
        return None
    normalized = name.strip().lower().replace("-", "_")
    return _GUARDRAIL_PRESET_ALIASES.get(normalized, normalized)


def _guardrail_preset_values(guardrails: Mapping[str, Any]) -> list[Any]:
    presets = guardrails.get("presets")
    if presets is None:
        presets = guardrails.get("managed_presets")
    if isinstance(presets, list):
        return presets
    if isinstance(presets, str) and presets.strip():
        return [presets]
    return []


def _guardrail_list_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _combine_guardrail_list(existing: Any, incoming: Any) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    for item in [*_guardrail_list_items(existing), *_guardrail_list_items(incoming)]:
        try:
            key = json.dumps(item, sort_keys=True)
        except TypeError:
            key = str(item)
        if key in seen:
            continue
        seen.add(key)
        values.append(copy.deepcopy(item))
    return values


def _combine_guardrail_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    combined = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in {"blocked_terms", "blocked_patterns", "external_classifiers"}:
            combined[key] = _combine_guardrail_list(combined.get(key), value)
            continue
        if isinstance(combined.get(key), dict) and isinstance(value, dict):
            nested = dict(combined[key])
            nested.update(copy.deepcopy(value))
            combined[key] = nested
            continue
        combined[key] = copy.deepcopy(value)
    return combined


def _guardrail_preset_expansion(
    guardrails: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]] | None]:
    preset_config: dict[str, Any] = {}
    applied: list[str] = []
    ignored: list[str] = []
    seen: set[str] = set()
    for value in _guardrail_preset_values(guardrails):
        normalized = _normalize_guardrail_preset_name(value)
        if normalized is None:
            continue
        if normalized not in _GUARDRAIL_PRESETS:
            ignored.append(normalized)
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        applied.append(normalized)
        preset_config = _combine_guardrail_config(preset_config, _GUARDRAIL_PRESETS[normalized])

    if not applied and not ignored:
        return preset_config, None
    metadata = {"applied": applied, "ignored": ignored}
    return preset_config, metadata


def _effective_guardrails_config(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, list[str]] | None]:
    guardrails = _guardrails_config(config)
    preset_config, preset_metadata = _guardrail_preset_expansion(guardrails)
    if not preset_config:
        return guardrails, preset_metadata
    return _combine_guardrail_config(preset_config, guardrails), preset_metadata


def _redactions_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    redactions = _guardrails_config(config).get("redactions")
    return redactions if isinstance(redactions, dict) else {}


def _redactions_enabled(config: Mapping[str, Any]) -> bool:
    redactions = _redactions_config(config)
    return bool_config(redactions.get("enabled"), bool(redactions))


def _guardrails_enabled(config: Mapping[str, Any]) -> bool:
    guardrails = _guardrails_config(config)
    return bool_config(guardrails.get("enabled"), bool(guardrails))


def guardrail_action(config: Mapping[str, Any]) -> str:
    action = _guardrails_config(config).get("action")
    if isinstance(action, str) and action.strip().lower() in _GUARDRAIL_ACTIONS:
        return action.strip().lower()
    return "block"


def _request_guardrail_text(request_body: Mapping[str, Any]) -> str:
    parts = [
        jsonable_text(request_body.get("messages")),
        jsonable_text(request_body.get("input")),
        jsonable_text(request_body.get("instructions")),
    ]
    return "\n".join(part for part in parts if part)


def _named_patterns(value: Any) -> list[tuple[str, re.Pattern[str]]]:
    if not isinstance(value, list):
        return []
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for index, item in enumerate(value, start=1):
        name = f"pattern_{index}"
        pattern_value: Any = item
        if isinstance(item, dict):
            name_value = item.get("name")
            if isinstance(name_value, str) and name_value.strip():
                name = name_value.strip()
            pattern_value = item.get("pattern")
        if not isinstance(pattern_value, str) or not pattern_value.strip():
            continue
        try:
            patterns.append((name, re.compile(pattern_value, re.IGNORECASE)))
        except re.error:
            logger.warning("Ignoring invalid routing guardrail regex pattern '%s'", name)
    return patterns


def _guardrail_violation(kind: str, rule: str) -> dict[str, str]:
    return {"type": kind, "rule": rule}


def _redaction_rules(config: Mapping[str, Any]) -> list[tuple[str, str, re.Pattern[str]]]:
    redactions = _redactions_config(config)
    rules: list[tuple[str, str, re.Pattern[str]]] = []

    pii_config = redactions.get("pii")
    pii_enabled = bool_config(pii_config.get("enabled") if isinstance(pii_config, dict) else pii_config, False)
    if pii_enabled:
        type_config = pii_config.get("types") if isinstance(pii_config, dict) else redactions.get("pii_types")
        pii_types = _string_list(type_config) or sorted(_PII_PATTERNS)
        for pii_type in pii_types:
            pattern = _PII_PATTERNS.get(pii_type)
            if pattern is not None:
                rules.append(("pii", pii_type, pattern))

    for name, pattern in _named_patterns(redactions.get("patterns")):
        rules.append(("pattern", name, pattern))
    return rules


def _redact_text(
    value: str,
    *,
    rules: Sequence[tuple[str, str, re.Pattern[str]]],
    replacement: str,
    counts: dict[tuple[str, str], int],
) -> str:
    redacted = value
    for kind, rule, pattern in rules:
        redacted, count = pattern.subn(replacement, redacted)
        if count:
            key = (kind, rule)
            counts[key] = counts.get(key, 0) + count
    return redacted


def _redact_content(
    value: Any,
    *,
    rules: Sequence[tuple[str, str, re.Pattern[str]]],
    replacement: str,
    counts: dict[tuple[str, str], int],
) -> Any:
    if isinstance(value, str):
        return _redact_text(value, rules=rules, replacement=replacement, counts=counts)
    if isinstance(value, list):
        return [_redact_content(item, rules=rules, replacement=replacement, counts=counts) for item in value]
    if isinstance(value, dict):
        return {
            key: _redact_content(item, rules=rules, replacement=replacement, counts=counts)
            for key, item in value.items()
        }
    return value


def _redaction_replacement(config: Mapping[str, Any]) -> str:
    replacement = _redactions_config(config).get("replacement")
    if isinstance(replacement, str):
        return replacement
    return "[REDACTED]"


def apply_guardrail_redactions(
    config: Mapping[str, Any],
    request_body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply policy redactions to provider-bound request content."""
    body = copy.deepcopy(dict(request_body))
    if not _redactions_enabled(config):
        return body, None

    rules = _redaction_rules(config)
    redactions = _redactions_config(config)
    replacement = _redaction_replacement(config)
    if not rules:
        return body, {
            "enabled": True,
            "status": "skipped",
            "reason": "missing_rules",
            "replacement": replacement,
        }

    counts: dict[tuple[str, str], int] = {}
    messages = body.get("messages")
    if isinstance(messages, list):
        redacted_messages: list[Any] = []
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                redacted_message = dict(message)
                redacted_message["content"] = _redact_content(
                    message.get("content"),
                    rules=rules,
                    replacement=replacement,
                    counts=counts,
                )
                redacted_messages.append(redacted_message)
            else:
                redacted_messages.append(message)
        body["messages"] = redacted_messages

    for key in ("input", "instructions"):
        if key in body:
            body[key] = _redact_content(body[key], rules=rules, replacement=replacement, counts=counts)

    count_items = [
        {"type": kind, "rule": rule, "count": count}
        for (kind, rule), count in sorted(counts.items())
    ]
    total_replacements = sum(counts.values())
    return body, {
        "enabled": True,
        "status": "redacted" if total_replacements else "unchanged",
        "replacement": replacement,
        "total_replacements": total_replacements,
        "counts": count_items,
        "pattern_count": len(_named_patterns(redactions.get("patterns"))),
    }


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


async def _evaluate_external_classifiers(
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


async def evaluate_guardrails(
    config: Mapping[str, Any],
    request_body: Mapping[str, Any],
    *,
    post_classifier: ExternalClassifierPost = post_external_guardrail_classifier,
) -> dict[str, Any] | None:
    if not _guardrails_enabled(config):
        return None

    guardrails, preset_metadata = _effective_guardrails_config(config)
    request_text = _request_guardrail_text(request_body)
    normalized_text = request_text.lower()
    violations: list[dict[str, str]] = []
    classifier_results: list[dict[str, Any]] = []

    for term in _string_list(guardrails.get("blocked_terms")):
        if term.lower() in normalized_text:
            violations.append(_guardrail_violation("blocked_term", term))

    for name, pattern in _named_patterns(guardrails.get("blocked_patterns")):
        if pattern.search(request_text):
            violations.append(_guardrail_violation("blocked_pattern", name))

    pii_config = guardrails.get("pii")
    pii_enabled = bool_config(pii_config.get("enabled") if isinstance(pii_config, dict) else pii_config, False)
    if pii_enabled:
        type_config = pii_config.get("types") if isinstance(pii_config, dict) else None
        pii_types = _string_list(type_config) or sorted(_PII_PATTERNS)
        for pii_type in pii_types:
            pii_pattern = _PII_PATTERNS.get(pii_type)
            if pii_pattern is not None and pii_pattern.search(request_text):
                violations.append(_guardrail_violation("pii", pii_type))

    injection_config = guardrails.get("prompt_injection")
    injection_enabled = bool_config(
        injection_config.get("enabled") if isinstance(injection_config, dict) else injection_config,
        False,
    )
    if injection_enabled:
        phrases = _string_list(injection_config.get("phrases") if isinstance(injection_config, dict) else None)
        for phrase in [*phrases, *_PROMPT_INJECTION_PHRASES]:
            if phrase.lower() in normalized_text:
                violations.append(_guardrail_violation("prompt_injection", phrase))

    external_violations, classifier_results = await _evaluate_external_classifiers(
        guardrails=guardrails,
        request_text=request_text,
        post_classifier=post_classifier,
    )
    violations.extend(external_violations)

    action = guardrail_action(config)
    status_value = "passed"
    if violations:
        status_value = "blocked" if action == "block" else "observed"
    result: dict[str, Any] = {
        "enabled": True,
        "status": status_value,
        "action": action,
        "violations": violations,
        "external_classifiers": classifier_results,
        "checked_text_chars": len(request_text),
    }
    if preset_metadata is not None:
        result["presets"] = preset_metadata
    return result

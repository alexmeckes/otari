"""Guardrail redaction helpers for provider-bound request content."""

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

from gateway.services.routing_config_values import bool_config, dict_or_empty
from gateway.services.routing_guardrail_helpers import guardrails_config, named_patterns, pii_patterns_from_config

_RedactionRule = tuple[str, str, re.Pattern[str]]
_RedactionCountKey = tuple[str, str]
_RedactionCounts = dict[_RedactionCountKey, int]


def _redact_string(
    value: str,
    *,
    rules: Sequence[_RedactionRule],
    replacement: str,
    counts: _RedactionCounts,
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
    rules: Sequence[_RedactionRule],
    replacement: str,
    counts: _RedactionCounts,
) -> Any:
    if isinstance(value, str):
        return _redact_string(value, rules=rules, replacement=replacement, counts=counts)
    if isinstance(value, list):
        return _redact_list(value, rules=rules, replacement=replacement, counts=counts)
    if isinstance(value, dict):
        return _redact_mapping(value, rules=rules, replacement=replacement, counts=counts)
    return value


def _redact_list(
    value: list[Any],
    *,
    rules: Sequence[_RedactionRule],
    replacement: str,
    counts: _RedactionCounts,
) -> list[Any]:
    return [_redact_content(item, rules=rules, replacement=replacement, counts=counts) for item in value]


def _redact_mapping(
    value: Mapping[Any, Any],
    *,
    rules: Sequence[_RedactionRule],
    replacement: str,
    counts: _RedactionCounts,
) -> dict[Any, Any]:
    return {
        key: _redact_content(item, rules=rules, replacement=replacement, counts=counts)
        for key, item in value.items()
    }


def _redaction_rules(redactions: Mapping[str, Any]) -> tuple[list[_RedactionRule], int]:
    rules: list[_RedactionRule] = []
    for pii_type, pattern in pii_patterns_from_config(
        redactions.get("pii"),
        fallback_types=redactions.get("pii_types"),
    ):
        rules.append(("pii", pii_type, pattern))
    pattern_rules = named_patterns(redactions.get("patterns"))
    for name, pattern in pattern_rules:
        rules.append(("pattern", name, pattern))
    return rules, len(pattern_rules)


def _redactions_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict_or_empty(guardrails_config(config).get("redactions"))


def _redaction_replacement(redactions: Mapping[str, Any]) -> str:
    replacement = redactions.get("replacement")
    if isinstance(replacement, str):
        return replacement
    return "[REDACTED]"


def _redactions_enabled(redactions: Mapping[str, Any]) -> bool:
    return bool_config(redactions.get("enabled"), bool(redactions))


def _redact_message(
    message: Any,
    *,
    rules: Sequence[_RedactionRule],
    replacement: str,
    counts: _RedactionCounts,
) -> Any:
    if isinstance(message, dict) and "content" in message:
        redacted_message = dict(message)
        redacted_message["content"] = _redact_content(
            message.get("content"),
            rules=rules,
            replacement=replacement,
            counts=counts,
        )
        return redacted_message
    return message


def _redact_messages(
    messages: Any,
    *,
    rules: Sequence[_RedactionRule],
    replacement: str,
    counts: _RedactionCounts,
) -> list[Any] | None:
    if not isinstance(messages, list):
        return None

    return [_redact_message(message, rules=rules, replacement=replacement, counts=counts) for message in messages]


def _redact_request_fields(
    body: dict[str, Any],
    *,
    rules: Sequence[_RedactionRule],
    replacement: str,
    counts: _RedactionCounts,
) -> None:
    for key in ("input", "instructions"):
        if key in body:
            body[key] = _redact_content(body[key], rules=rules, replacement=replacement, counts=counts)


def _missing_redaction_rules_trace(replacement: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "status": "skipped",
        "reason": "missing_rules",
        "replacement": replacement,
    }


def _redaction_trace(
    *,
    replacement: str,
    counts: Mapping[_RedactionCountKey, int],
    pattern_count: int,
) -> dict[str, Any]:
    count_items = [
        {"type": kind, "rule": rule, "count": count}
        for (kind, rule), count in sorted(counts.items())
    ]
    total_replacements = sum(counts.values())
    return {
        "enabled": True,
        "status": "redacted" if total_replacements else "unchanged",
        "replacement": replacement,
        "total_replacements": total_replacements,
        "counts": count_items,
        "pattern_count": pattern_count,
    }


def apply_guardrail_redactions(
    config: Mapping[str, Any],
    request_body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply policy redactions to provider-bound request content."""
    body = copy.deepcopy(dict(request_body))
    redactions = _redactions_config(config)
    if not _redactions_enabled(redactions):
        return body, None

    rules, pattern_count = _redaction_rules(redactions)
    replacement = _redaction_replacement(redactions)
    if not rules:
        return body, _missing_redaction_rules_trace(replacement)

    counts: _RedactionCounts = {}
    redacted_messages = _redact_messages(
        body.get("messages"),
        rules=rules,
        replacement=replacement,
        counts=counts,
    )
    if redacted_messages is not None:
        body["messages"] = redacted_messages

    _redact_request_fields(
        body,
        rules=rules,
        replacement=replacement,
        counts=counts,
    )

    return body, _redaction_trace(
        replacement=replacement,
        counts=counts,
        pattern_count=pattern_count,
    )

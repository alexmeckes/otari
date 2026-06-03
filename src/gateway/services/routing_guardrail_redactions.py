"""Guardrail redaction helpers for provider-bound request content."""

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

from gateway.services.routing_config_values import bool_config, dict_or_empty
from gateway.services.routing_guardrail_helpers import guardrails_config, named_patterns, pii_patterns_from_config

_RedactionRule = tuple[str, str, re.Pattern[str]]


def _redact_content(
    value: Any,
    *,
    rules: Sequence[_RedactionRule],
    replacement: str,
    counts: dict[tuple[str, str], int],
) -> Any:
    if isinstance(value, str):
        redacted = value
        for kind, rule, pattern in rules:
            redacted, count = pattern.subn(replacement, redacted)
            if count:
                key = (kind, rule)
                counts[key] = counts.get(key, 0) + count
        return redacted
    if isinstance(value, list):
        return [_redact_content(item, rules=rules, replacement=replacement, counts=counts) for item in value]
    if isinstance(value, dict):
        return {
            key: _redact_content(item, rules=rules, replacement=replacement, counts=counts)
            for key, item in value.items()
        }
    return value


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


def _redact_messages(
    messages: Any,
    *,
    rules: Sequence[_RedactionRule],
    replacement: str,
    counts: dict[tuple[str, str], int],
) -> list[Any] | None:
    if not isinstance(messages, list):
        return None

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
    return redacted_messages


def apply_guardrail_redactions(
    config: Mapping[str, Any],
    request_body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply policy redactions to provider-bound request content."""
    body = copy.deepcopy(dict(request_body))
    redactions = dict_or_empty(guardrails_config(config).get("redactions"))
    if not bool_config(redactions.get("enabled"), bool(redactions)):
        return body, None

    rules, pattern_count = _redaction_rules(redactions)
    replacement = redactions.get("replacement")
    if not isinstance(replacement, str):
        replacement = "[REDACTED]"
    if not rules:
        return body, {
            "enabled": True,
            "status": "skipped",
            "reason": "missing_rules",
            "replacement": replacement,
        }

    counts: dict[tuple[str, str], int] = {}
    redacted_messages = _redact_messages(
        body.get("messages"),
        rules=rules,
        replacement=replacement,
        counts=counts,
    )
    if redacted_messages is not None:
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
        "pattern_count": pattern_count,
    }

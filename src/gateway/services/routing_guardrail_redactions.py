"""Guardrail redaction helpers for provider-bound request content."""

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

from gateway.services.routing_config_values import bool_config, dict_or_empty
from gateway.services.routing_guardrail_helpers import guardrails_config, named_patterns, pii_patterns_from_config


def _redactions_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict_or_empty(guardrails_config(config).get("redactions"))


def _redactions_enabled(redactions: Mapping[str, Any]) -> bool:
    return bool_config(redactions.get("enabled"), bool(redactions))


def _redaction_rules(redactions: Mapping[str, Any]) -> list[tuple[str, str, re.Pattern[str]]]:
    rules: list[tuple[str, str, re.Pattern[str]]] = []

    for pii_type, pattern in pii_patterns_from_config(
        redactions.get("pii"),
        fallback_types=redactions.get("pii_types"),
    ):
        rules.append(("pii", pii_type, pattern))

    for name, pattern in named_patterns(redactions.get("patterns")):
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


def apply_guardrail_redactions(
    config: Mapping[str, Any],
    request_body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply policy redactions to provider-bound request content."""
    body = copy.deepcopy(dict(request_body))
    redactions = _redactions_config(config)
    if not _redactions_enabled(redactions):
        return body, None

    rules = _redaction_rules(redactions)
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
        "pattern_count": len(named_patterns(redactions.get("patterns"))),
    }

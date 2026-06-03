"""Guardrail redaction helpers for provider-bound request content."""

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gateway.services.routing_config_values import bool_config, dict_or_empty
from gateway.services.routing_guardrail_helpers import guardrails_config, named_patterns, pii_patterns_from_config

_RedactionRule = tuple[str, str, re.Pattern[str]]
_RedactionCountKey = tuple[str, str]
_RedactionCounts = dict[_RedactionCountKey, int]


@dataclass(frozen=True)
class _RedactionContext:
    rules: Sequence[_RedactionRule]
    replacement: str
    counts: _RedactionCounts


def _redact_content(
    value: Any,
    *,
    context: _RedactionContext,
) -> Any:
    if isinstance(value, str):
        redacted = value
        for kind, rule, pattern in context.rules:
            redacted, count = pattern.subn(context.replacement, redacted)
            if count:
                key = (kind, rule)
                context.counts[key] = context.counts.get(key, 0) + count
        return redacted
    if isinstance(value, list):
        return [_redact_content(item, context=context) for item in value]
    if isinstance(value, dict):
        return {key: _redact_content(item, context=context) for key, item in value.items()}
    return value


def apply_guardrail_redactions(
    config: Mapping[str, Any],
    request_body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply policy redactions to provider-bound request content."""
    body = copy.deepcopy(dict(request_body))
    redactions = dict_or_empty(guardrails_config(config).get("redactions"))
    if not bool_config(redactions.get("enabled"), bool(redactions)):
        return body, None

    pattern_rules = [
        ("pattern", name, pattern)
        for name, pattern in named_patterns(redactions.get("patterns"))
    ]
    rules: list[_RedactionRule] = [
        ("pii", name, pattern)
        for name, pattern in pii_patterns_from_config(
            redactions.get("pii"),
            fallback_types=redactions.get("pii_types"),
        )
    ] + pattern_rules
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

    context = _RedactionContext(rules=rules, replacement=replacement, counts={})
    messages = body.get("messages")
    if isinstance(messages, list):
        redacted_messages: list[Any] = []
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                redacted_messages.append(
                    {
                        **message,
                        "content": _redact_content(
                            message.get("content"),
                            context=context,
                        ),
                    }
                )
            else:
                redacted_messages.append(message)
        body["messages"] = redacted_messages

    for key in ("input", "instructions"):
        if key in body:
            body[key] = _redact_content(body[key], context=context)

    total_replacements = sum(context.counts.values())
    return body, {
        "enabled": True,
        "status": "redacted" if total_replacements else "unchanged",
        "replacement": context.replacement,
        "total_replacements": total_replacements,
        "counts": [
            {"type": kind, "rule": rule, "count": count}
            for (kind, rule), count in sorted(context.counts.items())
        ],
        "pattern_count": len(pattern_rules),
    }

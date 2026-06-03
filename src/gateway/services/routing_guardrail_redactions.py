"""Guardrail redaction helpers for provider-bound request content."""

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gateway.services.routing_config_values import bool_config, dict_or_empty
from gateway.services.routing_guardrail_helpers import guardrails_config, named_patterns, pii_patterns_from_config

_RedactionRule = tuple[str, str, re.Pattern[str]]
_PatternRule = tuple[str, re.Pattern[str]]
_RedactionCountKey = tuple[str, str]
_RedactionCounts = dict[_RedactionCountKey, int]
_REDACTABLE_REQUEST_FIELDS = ("input", "instructions")


@dataclass(frozen=True)
class _RedactionContext:
    rules: Sequence[_RedactionRule]
    replacement: str
    counts: _RedactionCounts


def _redact_string(
    value: str,
    *,
    context: _RedactionContext,
) -> str:
    redacted = value
    for kind, rule, pattern in context.rules:
        redacted, count = pattern.subn(context.replacement, redacted)
        if count:
            key = (kind, rule)
            context.counts[key] = context.counts.get(key, 0) + count
    return redacted


def _redact_content(
    value: Any,
    *,
    context: _RedactionContext,
) -> Any:
    if isinstance(value, str):
        return _redact_string(value, context=context)
    if isinstance(value, list):
        return _redact_list(value, context=context)
    if isinstance(value, dict):
        return _redact_mapping(value, context=context)
    return value


def _redact_list(
    value: list[Any],
    *,
    context: _RedactionContext,
) -> list[Any]:
    return [_redact_content(item, context=context) for item in value]


def _redact_mapping(
    value: Mapping[Any, Any],
    *,
    context: _RedactionContext,
) -> dict[Any, Any]:
    return {key: _redact_content(item, context=context) for key, item in value.items()}


def _typed_redaction_rules(kind: str, patterns: list[_PatternRule]) -> list[_RedactionRule]:
    return [(kind, name, pattern) for name, pattern in patterns]


def _redaction_rules(redactions: Mapping[str, Any]) -> tuple[list[_RedactionRule], int]:
    pattern_rules = _typed_redaction_rules("pattern", named_patterns(redactions.get("patterns")))
    return (
        _typed_redaction_rules(
            "pii",
            pii_patterns_from_config(
                redactions.get("pii"),
                fallback_types=redactions.get("pii_types"),
            ),
        )
        + pattern_rules,
        len(pattern_rules),
    )


def _redact_message(
    message: Any,
    *,
    context: _RedactionContext,
) -> Any:
    if isinstance(message, dict) and "content" in message:
        return {
            **message,
            "content": _redact_content(
                message.get("content"),
                context=context,
            ),
        }
    return message


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

    context = _RedactionContext(rules=rules, replacement=replacement, counts={})
    messages = body.get("messages")
    if isinstance(messages, list):
        body["messages"] = [_redact_message(message, context=context) for message in messages]

    for key in _REDACTABLE_REQUEST_FIELDS:
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
        "pattern_count": pattern_count,
    }

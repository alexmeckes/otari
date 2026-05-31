"""Guardrail redaction helpers for provider-bound request content."""

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

from gateway.log_config import logger
from gateway.services.routing_request_analysis import bool_config

_PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _guardrails_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    guardrails = config.get("guardrails")
    return guardrails if isinstance(guardrails, dict) else {}


def _redactions_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    redactions = _guardrails_config(config).get("redactions")
    return redactions if isinstance(redactions, dict) else {}


def _redactions_enabled(config: Mapping[str, Any]) -> bool:
    redactions = _redactions_config(config)
    return bool_config(redactions.get("enabled"), bool(redactions))


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

"""Shared guardrail config parsing helpers."""

import re
from collections.abc import Mapping
from typing import Any

from gateway.log_config import logger
from gateway.services.routing_config_values import bool_config, dict_or_empty, string_list, string_or_none

__all__ = [
    "PII_PATTERNS",
    "guardrail_violation",
    "guardrails_config",
    "named_patterns",
    "pii_patterns_from_config",
    "string_list",
]

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def guardrails_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict_or_empty(config.get("guardrails"))


def pii_patterns_from_config(
    pii_config: Any,
    *,
    fallback_types: Any = None,
) -> list[tuple[str, re.Pattern[str]]]:
    enabled = bool_config(pii_config.get("enabled") if isinstance(pii_config, dict) else pii_config, False)
    if not enabled:
        return []
    type_config = pii_config.get("types") if isinstance(pii_config, dict) else fallback_types
    pii_types = string_list(type_config) or sorted(PII_PATTERNS)
    return [
        (pii_type, pattern)
        for pii_type in pii_types
        if (pattern := PII_PATTERNS.get(pii_type)) is not None
    ]


def named_patterns(value: Any) -> list[tuple[str, re.Pattern[str]]]:
    if not isinstance(value, list):
        return []
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for index, item in enumerate(value, start=1):
        name = f"pattern_{index}"
        pattern_value: Any = item
        if isinstance(item, dict):
            name = string_or_none(item.get("name")) or name
            pattern_value = item.get("pattern")
        if not isinstance(pattern_value, str) or not pattern_value.strip():
            continue
        try:
            patterns.append((name, re.compile(pattern_value, re.IGNORECASE)))
        except re.error:
            logger.warning("Ignoring invalid routing guardrail regex pattern '%s'", name)
    return patterns


def guardrail_violation(kind: str, rule: str) -> dict[str, str]:
    return {"type": kind, "rule": rule}

"""Shared guardrail config parsing helpers."""

import re
from collections.abc import Mapping
from typing import Any

from gateway.log_config import logger
from gateway.services.routing_config_values import dict_or_empty, string_list

__all__ = [
    "PII_PATTERNS",
    "guardrail_violation",
    "guardrails_config",
    "named_patterns",
    "string_list",
]

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def guardrails_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict_or_empty(config.get("guardrails"))


def named_patterns(value: Any) -> list[tuple[str, re.Pattern[str]]]:
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


def guardrail_violation(kind: str, rule: str) -> dict[str, str]:
    return {"type": kind, "rule": rule}

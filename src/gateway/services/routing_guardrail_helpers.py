"""Shared guardrail config parsing helpers."""

import re
from collections.abc import Mapping
from typing import Any

from gateway.log_config import logger
from gateway.services.routing_config_values import dict_or_empty


def string_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


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

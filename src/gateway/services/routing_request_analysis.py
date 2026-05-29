"""Request sizing and complexity helpers for routing policies."""

import json
from collections.abc import Mapping
from typing import Any

_DEFAULT_OUTPUT_TOKENS = 700
_REASONING_HINTS = (
    "prove",
    "proof",
    "derive",
    "theorem",
    "formal",
    "optimization",
    "multi-step",
    "step by step",
)
_COMPLEX_HINTS = (
    "architecture",
    "design doc",
    "debug",
    "refactor",
    "implement",
    "migration",
    "security",
    "compliance",
    "analyze",
)
_MEDIUM_HINTS = (
    "summarize",
    "compare",
    "rewrite",
    "extract",
    "classify",
    "explain",
)


def jsonable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(jsonable_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(jsonable_text(item) for item in value.values())
    if value is None:
        return ""
    return str(value)


def estimate_prompt_tokens(request_body: Mapping[str, Any]) -> int:
    """Estimate prompt tokens without pulling in a tokenizer dependency."""
    text_parts: list[str] = []
    messages = request_body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                text_parts.append(jsonable_text(message.get("content")))

    tools = request_body.get("tools")
    if tools:
        try:
            text_parts.append(json.dumps(tools, sort_keys=True))
        except TypeError:
            text_parts.append(jsonable_text(tools))

    joined = " ".join(part for part in text_parts if part)
    return max(1, len(joined) // 4)


def estimate_output_tokens(request_body: Mapping[str, Any]) -> int:
    """Estimate expected output tokens from request limits."""
    for key in ("max_completion_tokens", "max_tokens"):
        value = request_body.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return _DEFAULT_OUTPUT_TOKENS


def int_config(value: Any, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return default


def non_negative_int_config(value: Any, default: int) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    return default


def bool_config(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def classify_request_tier(
    request_body: Mapping[str, Any],
    *,
    prompt_tokens: int,
    config: Mapping[str, Any],
) -> str:
    """Classify a request into a ClawSwitch-style complexity tier."""
    thresholds = config.get("tier_thresholds")
    threshold_map = thresholds if isinstance(thresholds, dict) else {}
    medium_threshold = int_config(threshold_map.get("medium"), 800)
    complex_threshold = int_config(threshold_map.get("complex"), 3000)
    reasoning_threshold = int_config(threshold_map.get("reasoning"), 9000)

    request_text = jsonable_text(request_body.get("messages")).lower()
    if prompt_tokens >= reasoning_threshold or any(hint in request_text for hint in _REASONING_HINTS):
        return "reasoning"
    if prompt_tokens >= complex_threshold or any(hint in request_text for hint in _COMPLEX_HINTS):
        return "complex"
    if prompt_tokens >= medium_threshold or any(hint in request_text for hint in _MEDIUM_HINTS):
        return "medium"
    return "simple"

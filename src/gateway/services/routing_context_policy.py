"""Deterministic context-window policy helpers for routing policies."""

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from gateway.services.routing_config_values import (
    bool_config,
    int_config,
    nested_dict_or_empty,
    non_negative_int_config,
    string_or_none,
)
from gateway.services.routing_request_analysis import (
    estimate_prompt_tokens,
    jsonable_text,
    stable_json_text,
)

_SYSTEM_MESSAGE_ROLES = {"system", "developer"}


def _message_role(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    return (string_or_none(message.get("role")) or "").lower()


def _message_token_estimate(message: Any) -> int:
    if not isinstance(message, dict):
        return max(1, len(jsonable_text(message)) // 4)
    text = jsonable_text(message.get("content"))
    return max(1, len(text) // 4)


def _non_message_prompt_tokens(request_body: Mapping[str, Any]) -> int:
    tools = request_body.get("tools")
    if not tools:
        return 0
    text = stable_json_text(tools)
    return max(1, len(text) // 4)


def _context_kept_message_indexes(
    messages: Sequence[Any],
    *,
    preserve_system_messages: bool,
    preserve_last_messages: int,
) -> set[int]:
    kept_indexes: set[int] = set()
    if preserve_system_messages:
        for index, message in enumerate(messages):
            if _message_role(message) in _SYSTEM_MESSAGE_ROLES:
                kept_indexes.add(index)

    if preserve_last_messages:
        for index in range(max(0, len(messages) - preserve_last_messages), len(messages)):
            kept_indexes.add(index)
    return kept_indexes


def _summary_line_for_message(message: Any, *, max_chars: int) -> str:
    role = _message_role(message) or "message"
    text = jsonable_text(message.get("content") if isinstance(message, dict) else message)
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = f"{text[: max(0, max_chars - 3)].rstrip()}..."
    return f"- {role}: {text}" if text else f"- {role}: [empty]"


def _trim_summary_to_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return ""
    return f"{text[: max_chars - 3].rstrip()}..."


def _build_context_summary(
    messages: Sequence[Any],
    summarized_indexes: Sequence[int],
    *,
    max_chars: int,
    prefix: str,
    max_message_chars: int,
) -> str:
    if max_chars <= 0:
        return ""
    lines = [string_or_none(prefix) or "Earlier conversation summary:"]
    for index in summarized_indexes:
        lines.append(_summary_line_for_message(messages[index], max_chars=max_message_chars))
    return _trim_summary_to_chars("\n".join(lines), max_chars)


def _summary_message_role(context_config: Mapping[str, Any]) -> str:
    summary_role = (string_or_none(context_config.get("summary_role")) or "system").lower()
    if summary_role in _SYSTEM_MESSAGE_ROLES or summary_role == "user":
        return summary_role
    return "system"


def _messages_with_summary(
    messages: Sequence[Any],
    kept_indexes: set[int],
    *,
    summary_message: dict[str, Any] | None,
) -> list[Any]:
    result: list[Any] = []
    summary_inserted = summary_message is None
    for index, message in enumerate(messages):
        if index not in kept_indexes:
            continue
        if not summary_inserted and _message_role(message) not in _SYSTEM_MESSAGE_ROLES:
            result.append(summary_message)
            summary_inserted = True
        result.append(message)
    if not summary_inserted and summary_message is not None:
        result.append(summary_message)
    return result


def _context_trace(status: str, strategy: str, **extra: Any) -> dict[str, Any]:
    trace: dict[str, Any] = {"enabled": True, "status": status, "strategy": strategy}
    trace.update(extra)
    return trace


def apply_context_policy(
    config: Mapping[str, Any],
    request_body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply deterministic prompt compression from policy config and return trace metadata."""
    body = copy.deepcopy(dict(request_body))
    context_config = nested_dict_or_empty(config, "context", "context_policy")
    if not bool_config(context_config.get("enabled"), bool(context_config)):
        return body, None

    strategy_raw = context_config.get("strategy", "trim_messages")
    strategy_value = string_or_none(strategy_raw)
    strategy = strategy_value.lower() if strategy_value is not None else None
    if strategy not in {"trim_messages", "summarize_messages"}:
        return body, _context_trace("skipped", str(strategy_raw), reason="unsupported_strategy")

    max_prompt_tokens = int_config(context_config.get("max_prompt_tokens"), 0)
    if max_prompt_tokens <= 0:
        return body, _context_trace("skipped", strategy, reason="missing_max_prompt_tokens")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body, _context_trace(
            "skipped",
            strategy,
            reason="missing_messages",
            max_prompt_tokens=max_prompt_tokens,
        )

    original_prompt_tokens = estimate_prompt_tokens(body)
    original_message_count = len(messages)
    preserve_system_messages = bool_config(context_config.get("preserve_system_messages"), True)
    preserve_last_messages = non_negative_int_config(context_config.get("preserve_last_messages"), 4)
    if original_prompt_tokens <= max_prompt_tokens:
        return body, _context_trace(
            "unchanged",
            strategy,
            max_prompt_tokens=max_prompt_tokens,
            original_prompt_tokens=original_prompt_tokens,
            final_prompt_tokens=original_prompt_tokens,
            original_message_count=original_message_count,
            final_message_count=original_message_count,
            trimmed_message_count=0,
            preserve_system_messages=preserve_system_messages,
            preserve_last_messages=preserve_last_messages,
        )

    kept_indexes = _context_kept_message_indexes(
        messages,
        preserve_system_messages=preserve_system_messages,
        preserve_last_messages=preserve_last_messages,
    )

    if strategy == "summarize_messages":
        summarized_indexes = [index for index in range(len(messages)) if index not in kept_indexes]
        if not summarized_indexes:
            body["messages"] = [messages[index] for index in sorted(kept_indexes)]
            final_prompt_tokens = estimate_prompt_tokens(body)
            return body, _context_trace(
                "unchanged",
                strategy,
                max_prompt_tokens=max_prompt_tokens,
                original_prompt_tokens=original_prompt_tokens,
                final_prompt_tokens=final_prompt_tokens,
                original_message_count=original_message_count,
                final_message_count=len(body["messages"]),
                summarized_message_count=0,
                preserve_system_messages=preserve_system_messages,
                preserve_last_messages=preserve_last_messages,
            )

        kept_tokens = _non_message_prompt_tokens(body) + sum(
            _message_token_estimate(messages[index]) for index in kept_indexes
        )
        summary_token_budget = max_prompt_tokens - kept_tokens
        configured_summary_tokens = int_config(
            context_config.get("summary_max_tokens"),
            min(512, max(1, max_prompt_tokens // 4)),
        )
        summary_token_budget = min(configured_summary_tokens, summary_token_budget)
        summary_text = _build_context_summary(
            messages,
            summarized_indexes,
            max_chars=max(0, summary_token_budget * 4),
            prefix=str(context_config.get("summary_prefix") or "Earlier conversation summary:"),
            max_message_chars=int_config(context_config.get("summary_message_max_chars"), 240),
        )
        summary_role = _summary_message_role(context_config)
        summary_message = {"role": summary_role, "content": summary_text} if summary_text else None
        body["messages"] = _messages_with_summary(messages, kept_indexes, summary_message=summary_message)
        final_prompt_tokens = estimate_prompt_tokens(body)
        if summary_message is not None and final_prompt_tokens > max_prompt_tokens:
            overflow_chars = ((final_prompt_tokens - max_prompt_tokens) * 4) + 16
            summary_message["content"] = _trim_summary_to_chars(
                str(summary_message["content"]),
                max(0, len(str(summary_message["content"])) - overflow_chars),
            )
            if not summary_message["content"]:
                summary_message = None
            body["messages"] = _messages_with_summary(messages, kept_indexes, summary_message=summary_message)
            final_prompt_tokens = estimate_prompt_tokens(body)

        return body, _context_trace(
            "summarized" if summary_message is not None else "trimmed",
            strategy,
            max_prompt_tokens=max_prompt_tokens,
            original_prompt_tokens=original_prompt_tokens,
            final_prompt_tokens=final_prompt_tokens,
            original_message_count=original_message_count,
            final_message_count=len(body["messages"]),
            trimmed_message_count=original_message_count - len(body["messages"]),
            summarized_message_count=len(summarized_indexes),
            summary_message_role=summary_role,
            summary_chars=len(str(summary_message["content"])) if summary_message is not None else 0,
            preserve_system_messages=preserve_system_messages,
            preserve_last_messages=preserve_last_messages,
            over_budget_after_summarization=final_prompt_tokens > max_prompt_tokens,
        )

    message_budget = max(max_prompt_tokens - _non_message_prompt_tokens(body), 1)
    current_message_tokens = sum(_message_token_estimate(messages[index]) for index in kept_indexes)
    for index in range(len(messages) - 1, -1, -1):
        if index in kept_indexes:
            continue
        token_estimate = _message_token_estimate(messages[index])
        if current_message_tokens + token_estimate <= message_budget:
            kept_indexes.add(index)
            current_message_tokens += token_estimate

    body["messages"] = [messages[index] for index in sorted(kept_indexes)]
    final_prompt_tokens = estimate_prompt_tokens(body)
    trimmed_message_count = original_message_count - len(body["messages"])
    return body, _context_trace(
        "trimmed" if trimmed_message_count else "unchanged",
        strategy,
        max_prompt_tokens=max_prompt_tokens,
        original_prompt_tokens=original_prompt_tokens,
        final_prompt_tokens=final_prompt_tokens,
        original_message_count=original_message_count,
        final_message_count=len(body["messages"]),
        trimmed_message_count=trimmed_message_count,
        preserve_system_messages=preserve_system_messages,
        preserve_last_messages=preserve_last_messages,
        over_budget_after_trimming=final_prompt_tokens > max_prompt_tokens,
    )

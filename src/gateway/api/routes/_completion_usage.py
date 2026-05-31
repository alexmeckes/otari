"""Shared helpers for AnyLLM completion usage objects."""

from any_llm.types.completion import CompletionUsage


def completion_usage_from_token_counts(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None = None,
) -> CompletionUsage:
    """Build CompletionUsage from input/output token counts."""
    prompt_tokens = input_tokens or 0
    completion_tokens = output_tokens or 0
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
    )

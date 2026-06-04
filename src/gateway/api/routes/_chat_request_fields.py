"""Helpers for provider-bound chat request fields."""

from collections.abc import Mapping
from typing import Any

from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.services.chat_tool_config import strip_gateway_fields


def chat_provider_request_fields(
    request: ChatCompletionRequest,
    *,
    tools_extracted: bool = False,
    remaining_user_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return chat request fields suitable for direct provider dispatch."""
    return strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=tools_extracted,
        remaining_user_tools=remaining_user_tools,
    )


def chat_provider_call_kwargs(
    provider_kwargs: Mapping[str, Any],
    request_fields: Mapping[str, Any],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Merge provider credentials/config with provider-bound chat request fields."""
    call_kwargs = {**provider_kwargs, **request_fields}
    if model is not None:
        call_kwargs["model"] = model
    return call_kwargs

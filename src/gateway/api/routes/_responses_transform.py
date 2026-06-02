"""Transform helpers for the OpenAI-compatible Responses endpoint."""

import json
import time
import uuid
from typing import Any

from any_llm.types.completion import ChatCompletion, CompletionUsage
from fastapi import HTTPException, status
from fastapi import Response as FastAPIResponse
from openai.types.responses import ResponseUsage
from openresponses_types.types import Usage as OpenResponsesUsage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._completion_usage import completion_usage_from_token_counts
from gateway.services.pricing_service import legacy_pricing_model_ref
from gateway.services.routing_policy_service import DEFAULT_ROUTING_MODEL, require_routing_model_selector
from gateway.services.routing_policy_shape import split_model_selector


class ResponsesRequest(BaseModel):
    """OpenAI Responses API-compatible request."""

    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_ROUTING_MODEL
    input: Any
    instructions: str | None = None
    stream: bool = False
    user: str | None = None
    project_id: str | None = Field(default=None, description="Optional gateway project id for default routing")
    tags: dict[str, str] | None = Field(default=None, description="Optional trace tags for default routing")

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, v: Any) -> str:
        """Accept omitted/null/case-insensitive default_routing sentinels."""
        return require_routing_model_selector(v)


def usage_to_completion_usage(
    usage: ResponseUsage | OpenResponsesUsage | None,
) -> CompletionUsage | None:
    if usage is None:
        return None
    return completion_usage_from_token_counts(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def _content_to_text(value: Any) -> str:
    """Convert Responses API input/content items into chat-compatible text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_content_to_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        content = value.get("content")
        if content is not None:
            return _content_to_text(content)
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return str(value)
    return str(value)


def response_input_to_chat_messages(input_payload: Any, instructions: str | None) -> list[dict[str, Any]]:
    """Translate a common Responses API input shape into chat messages."""
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    def append_message(role: Any, content: Any) -> None:
        role_value = str(role or "user")
        if role_value == "developer":
            role_value = "system"
        text = _content_to_text(content)
        if text:
            messages.append({"role": role_value, "content": text})

    if isinstance(input_payload, str):
        append_message("user", input_payload)
    elif isinstance(input_payload, dict):
        if "role" in input_payload:
            append_message(input_payload.get("role"), input_payload.get("content", input_payload.get("text")))
        else:
            append_message("user", input_payload)
    elif isinstance(input_payload, list):
        for item in input_payload:
            if isinstance(item, dict) and "role" in item:
                append_message(item.get("role"), item.get("content", item.get("text")))
            else:
                append_message("user", item)
    else:
        append_message("user", input_payload)

    if not messages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Responses input must contain at least one message or text item",
        )
    return messages


def _chat_completion_text(completion: ChatCompletion) -> str:
    """Extract the first assistant text from a chat completion."""
    if not completion.choices:
        return ""
    message = completion.choices[0].message
    return _content_to_text(getattr(message, "content", None))


def served_metadata(provider: str, model: str) -> dict[str, str]:
    """Build served model/vendor metadata."""
    return {
        "model": legacy_pricing_model_ref(provider, model),
        "vendor": provider,
    }


def metadata_from_model_selector(model_selector: str) -> dict[str, str] | None:
    """Build served metadata from a provider-qualified model selector."""
    provider, model = split_model_selector(model_selector)
    if provider is None:
        return None
    return served_metadata(provider, model)


def set_served_headers(response: FastAPIResponse, metadata: dict[str, str]) -> None:
    """Expose the model/vendor that served a Responses request."""
    response.headers["X-Response-Model"] = metadata["model"]
    response.headers["X-Response-Vendor"] = metadata["vendor"]


def chat_completion_to_response_payload(completion: ChatCompletion) -> dict[str, Any]:
    """Wrap a routed chat completion in an OpenAI Responses-like payload."""
    text = _chat_completion_text(completion)
    usage = completion.usage
    metadata = metadata_from_model_selector(completion.model)
    payload: dict[str, Any] = {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": completion.created or int(time.time()),
        "status": "completed",
        "model": metadata["model"] if metadata else completion.model,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": text,
    }
    if metadata is not None:
        payload["vendor"] = metadata["vendor"]
    if usage is not None:
        payload["usage"] = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    return payload


def response_payload_with_served_metadata(
    payload: dict[str, Any],
    *,
    provider: str,
    requested_model: str,
) -> dict[str, Any]:
    """Attach served metadata to a provider-native Responses payload."""
    model_value = payload.get("model")
    provider_from_payload: str | None = None
    model_from_payload = requested_model
    if isinstance(model_value, str):
        provider_from_payload, model_from_payload = split_model_selector(model_value)

    served_provider = provider_from_payload or provider
    payload["model"] = served_metadata(served_provider, model_from_payload)["model"]
    payload["vendor"] = served_provider
    return payload


def chat_request_from_response_request(request_body: ResponsesRequest) -> ChatCompletionRequest:
    """Build a chat completion request for the standalone routing backend."""
    request_fields = request_body.model_dump(exclude_none=True)
    max_output_tokens = request_fields.get("max_output_tokens")
    max_tokens = max_output_tokens if isinstance(max_output_tokens, int) and max_output_tokens > 0 else None
    chat_request = ChatCompletionRequest(
        model=DEFAULT_ROUTING_MODEL,
        messages=response_input_to_chat_messages(request_body.input, request_body.instructions),
        user=request_body.user,
        project_id=request_body.project_id,
        tags=request_body.tags,
        temperature=request_fields.get("temperature"),
        max_tokens=max_tokens,
        top_p=request_fields.get("top_p"),
        tools=request_fields.get("tools"),
        tool_choice=request_fields.get("tool_choice"),
        response_format=request_fields.get("response_format"),
    )
    chat_request.set_route_trace_endpoint("/v1/responses")
    return chat_request

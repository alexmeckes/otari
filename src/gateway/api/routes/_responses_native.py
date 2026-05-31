from typing import Any

from any_llm.types.completion import CompletionUsage
from any_llm.types.responses import ResponseStreamEvent
from fastapi import Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._provider_context import OpenAIProviderRequestContext
from gateway.api.routes._responses_transform import (
    ResponsesRequest,
    response_payload_with_served_metadata,
    served_metadata,
    set_served_headers,
    usage_to_completion_usage,
)
from gateway.api.routes._usage import log_usage
from gateway.services.log_writer import LogWriter
from gateway.streaming import RESPONSES_STREAM_FORMAT, streaming_generator

RESPONSES_ENDPOINT = "/v1/responses"


def native_response_call_kwargs(
    request_body: ResponsesRequest,
    context: OpenAIProviderRequestContext,
) -> tuple[dict[str, Any], bool]:
    request_fields = request_body.model_dump(exclude_none=True)
    input_payload = request_fields.pop("input")
    stream = bool(request_fields.pop("stream", False))
    request_fields.pop("model", None)
    request_fields.pop("user", None)
    request_fields.pop("project_id", None)
    request_fields.pop("tags", None)
    request_fields["user"] = context.user_id

    call_kwargs: dict[str, Any] = {**context.provider_kwargs}
    call_kwargs.update(request_fields)
    call_kwargs["model"] = context.model
    call_kwargs["provider"] = context.provider
    call_kwargs["input_data"] = input_payload
    return call_kwargs, stream


async def log_native_response_usage(
    *,
    db: AsyncSession,
    log_writer: LogWriter,
    context: OpenAIProviderRequestContext,
    request_body: ResponsesRequest,
    usage_data: CompletionUsage | None = None,
    error: str | None = None,
) -> None:
    await log_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=context.api_key_id,
        model=context.model,
        provider=context.provider,
        endpoint=RESPONSES_ENDPOINT,
        user_id=context.user_id,
        project_id=request_body.project_id,
        tags=request_body.tags,
        usage_override=usage_data,
        error=error,
    )


def native_response_streaming_response(
    *,
    stream_result: Any,
    db: AsyncSession,
    log_writer: LogWriter,
    context: OpenAIProviderRequestContext,
    request_body: ResponsesRequest,
) -> StreamingResponse:
    async def _on_complete(usage_data: CompletionUsage) -> None:
        await log_native_response_usage(
            db=db,
            log_writer=log_writer,
            context=context,
            request_body=request_body,
            usage_data=usage_data,
        )

    async def _on_error(error: str) -> None:
        await log_native_response_usage(
            db=db,
            log_writer=log_writer,
            context=context,
            request_body=request_body,
            error=error,
        )

    return StreamingResponse(
        streaming_generator(
            stream=stream_result,
            format_chunk=_format_response_stream_event,
            extract_usage=_extract_response_stream_usage,
            fmt=RESPONSES_STREAM_FORMAT,
            on_complete=_on_complete,
            on_error=_on_error,
            label=f"{context.provider}:{context.model}",
        ),
        media_type="text/event-stream",
        headers=_response_stream_headers(context),
    )


def native_response_payload(
    *,
    result: Any,
    response: FastAPIResponse,
    context: OpenAIProviderRequestContext,
) -> dict[str, Any]:
    context.apply_rate_limit_headers(response)
    metadata = served_metadata(context.provider.value, context.model)
    set_served_headers(response, metadata)
    payload = result.model_dump(exclude_none=True)
    return response_payload_with_served_metadata(
        payload,
        provider=context.provider.value,
        requested_model=context.model,
    )


def _format_response_stream_event(event: ResponseStreamEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"


def _extract_response_stream_usage(event: ResponseStreamEvent) -> CompletionUsage | None:
    response_obj = getattr(event, "response", None)
    if response_obj and getattr(response_obj, "usage", None):
        return usage_to_completion_usage(response_obj.usage)
    return None


def _response_stream_headers(context: OpenAIProviderRequestContext) -> dict[str, str]:
    headers = context.rate_limit_headers()
    metadata = served_metadata(context.provider.value, context.model)
    headers["X-Response-Model"] = metadata["model"]
    headers["X-Response-Vendor"] = metadata["vendor"]
    return headers

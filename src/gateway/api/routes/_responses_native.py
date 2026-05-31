from typing import Any

from any_llm.types.completion import CompletionUsage
from any_llm.types.responses import ResponseStreamEvent
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._responses_transform import ResponsesRequest, served_metadata, usage_to_completion_usage
from gateway.api.routes._usage import log_usage, optional_rate_limit_headers
from gateway.rate_limit import RateLimitInfo
from gateway.services.log_writer import LogWriter
from gateway.streaming import RESPONSES_STREAM_FORMAT, streaming_generator

RESPONSES_ENDPOINT = "/v1/responses"


def native_response_call_kwargs(
    request_body: ResponsesRequest,
    provider_kwargs: dict[str, Any],
    provider: Any,
    model: str,
    user_id: str,
) -> tuple[dict[str, Any], bool]:
    request_fields = request_body.model_dump(exclude_none=True)
    input_payload = request_fields.pop("input")
    stream = bool(request_fields.pop("stream", False))
    request_fields.pop("model", None)
    request_fields.pop("user", None)
    request_fields.pop("project_id", None)
    request_fields.pop("tags", None)
    request_fields["user"] = user_id

    call_kwargs: dict[str, Any] = {**provider_kwargs}
    call_kwargs.update(request_fields)
    call_kwargs["model"] = model
    call_kwargs["provider"] = provider
    call_kwargs["input_data"] = input_payload
    return call_kwargs, stream


async def log_native_response_usage(
    *,
    db: AsyncSession,
    log_writer: LogWriter,
    api_key_id: str | None,
    provider: Any,
    model: str,
    user_id: str,
    request_body: ResponsesRequest,
    usage_data: CompletionUsage | None = None,
    error: str | None = None,
) -> None:
    await log_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=model,
        provider=provider,
        endpoint=RESPONSES_ENDPOINT,
        user_id=user_id,
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
    api_key_id: str | None,
    provider: Any,
    model: str,
    user_id: str,
    request_body: ResponsesRequest,
    rate_limit_info: RateLimitInfo | None,
) -> StreamingResponse:
    async def _on_complete(usage_data: CompletionUsage) -> None:
        await log_native_response_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            provider=provider,
            model=model,
            user_id=user_id,
            request_body=request_body,
            usage_data=usage_data,
        )

    async def _on_error(error: str) -> None:
        await log_native_response_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            provider=provider,
            model=model,
            user_id=user_id,
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
            label=f"{provider}:{model}",
        ),
        media_type="text/event-stream",
        headers=_response_stream_headers(rate_limit_info, provider.value, model),
    )


def _format_response_stream_event(event: ResponseStreamEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"


def _extract_response_stream_usage(event: ResponseStreamEvent) -> CompletionUsage | None:
    response_obj = getattr(event, "response", None)
    if response_obj and getattr(response_obj, "usage", None):
        return usage_to_completion_usage(response_obj.usage)
    return None


def _response_stream_headers(
    rate_limit_info: RateLimitInfo | None,
    provider_value: str,
    model: str,
) -> dict[str, str]:
    headers = optional_rate_limit_headers(rate_limit_info)
    metadata = served_metadata(provider_value, model)
    headers["X-Response-Model"] = metadata["model"]
    headers["X-Response-Vendor"] = metadata["vendor"]
    return headers

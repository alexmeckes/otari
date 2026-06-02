import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from any_llm import LLMProvider
from any_llm.types.completion import ChatCompletionChunk, CompletionUsage
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._completion_usage import completion_usage_from_token_counts
from gateway.api.routes._usage import log_usage, optional_rate_limit_headers, provider_model_label
from gateway.core.config import GatewayConfig
from gateway.rate_limit import RateLimitInfo
from gateway.services.log_writer import LogWriter
from gateway.services.platform_gateway import report_platform_usage
from gateway.streaming import OPENAI_STREAM_FORMAT, streaming_generator

_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"


def _schedule_platform_streaming_usage(
    *,
    config: GatewayConfig,
    correlation_id: str,
    outcome: str,
    usage: CompletionUsage | None,
) -> asyncio.Task[None]:
    return asyncio.create_task(
        report_platform_usage(
            config=config,
            correlation_id=correlation_id,
            outcome=outcome,
            usage=usage,
        )
    )


async def _log_standalone_streaming_usage(
    *,
    db: AsyncSession | None,
    log_writer: LogWriter | None,
    api_key_id: str | None,
    model: str,
    provider: LLMProvider,
    user_id: str | None,
    project_id: str | None,
    tags: Mapping[str, Any] | None,
    usage_data: CompletionUsage | None = None,
    error: str | None = None,
) -> None:
    if db is None or log_writer is None:
        return
    await log_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=model,
        provider=provider,
        endpoint=_CHAT_COMPLETIONS_ENDPOINT,
        user_id=user_id,
        project_id=project_id,
        tags=tags,
        usage_override=usage_data,
        error=error,
    )


def build_chat_streaming_response(
    *,
    stream: AsyncIterator[ChatCompletionChunk],
    provider: LLMProvider,
    model: str,
    platform_mode: bool,
    correlation_id: str | None,
    request_id: str | None,
    config: GatewayConfig,
    db: AsyncSession | None,
    log_writer: LogWriter | None,
    api_key_id: str | None,
    user_id: str | None,
    project_id: str | None,
    tags: Mapping[str, Any] | None,
    rate_limit_info: RateLimitInfo | None,
) -> StreamingResponse:
    """Wrap an already-opened upstream stream in an SSE response."""

    def _extract_usage(chunk: ChatCompletionChunk) -> CompletionUsage | None:
        if not chunk.usage:
            return None
        return completion_usage_from_token_counts(
            input_tokens=chunk.usage.prompt_tokens or 0,
            output_tokens=chunk.usage.completion_tokens or 0,
            total_tokens=chunk.usage.total_tokens or 0,
        )

    async def _on_complete(usage_data: CompletionUsage) -> None:
        if platform_mode and correlation_id:
            _schedule_platform_streaming_usage(
                config=config,
                correlation_id=correlation_id,
                outcome="success",
                usage=usage_data,
            )
            return
        await _log_standalone_streaming_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            user_id=user_id,
            project_id=project_id,
            tags=tags,
            usage_data=usage_data,
        )

    async def _on_error(error: str) -> None:
        if platform_mode and correlation_id:
            _schedule_platform_streaming_usage(
                config=config,
                correlation_id=correlation_id,
                outcome="error",
                usage=None,
            )
            return
        await _log_standalone_streaming_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            user_id=user_id,
            project_id=project_id,
            tags=tags,
            error=error,
        )

    # StreamingResponse builds its own response object, so headers we want on
    # the wire have to be passed here rather than assigned to FastAPI's
    # dependency-injected Response object.
    headers = optional_rate_limit_headers(rate_limit_info)
    if platform_mode and correlation_id:
        headers["X-Correlation-ID"] = correlation_id
    if platform_mode and request_id:
        headers["X-Otari-Request-ID"] = request_id
    return StreamingResponse(
        streaming_generator(
            stream=stream,
            format_chunk=lambda chunk: f"data: {chunk.model_dump_json()}\n\n",
            extract_usage=_extract_usage,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=_on_complete,
            on_error=_on_error,
            label=provider_model_label(provider, model),
        ),
        media_type="text/event-stream",
        headers=headers,
    )

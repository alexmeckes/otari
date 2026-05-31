from dataclasses import dataclass
from typing import Annotated, Any

from any_llm import AnyLLM, amessages
from any_llm.types.completion import CompletionUsage
from any_llm.types.messages import (
    MessageDeltaEvent,
    MessageResponse,
    MessageStartEvent,
    MessageStreamEvent,
)
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._budget_checks import validate_user_request_budget
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes._message_models import MessagesRequest
from gateway.api.routes._usage import apply_rate_limit_headers, log_usage, optional_rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import get_provider_kwargs
from gateway.streaming import ANTHROPIC_STREAM_FORMAT, streaming_generator

router = APIRouter(prefix="/v1", tags=["messages"])


def _anthropic_error(error_type: str, message: str, status_code: int) -> HTTPException:
    """Create an HTTPException with Anthropic-style error body."""
    return HTTPException(
        status_code=status_code,
        detail={"type": "error", "error": {"type": error_type, "message": message}},
    )


_ERR_INVALID_REQUEST = "invalid_request_error"
_ERR_API = "api_error"
_MASTER_KEY_USER_REQUIRED = "When using master key, 'metadata.user_id' is required in request body"
_API_KEY_VALIDATION_FAILED = "API key validation failed"
_API_KEY_NO_USER = "API key has no associated user"
_PROVIDER_ERROR = "The request could not be completed by the provider"
_MESSAGES_ENDPOINT = "/v1/messages"


@dataclass(frozen=True)
class MessageRequestContext:
    api_key_id: str | None
    user_id: str


def _message_completion_usage(input_tokens: int | None, output_tokens: int | None) -> CompletionUsage:
    prompt_tokens = input_tokens or 0
    completion_tokens = output_tokens or 0
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _message_response_usage(result: MessageResponse) -> CompletionUsage | None:
    if not result.usage:
        return None
    return _message_completion_usage(result.usage.input_tokens, result.usage.output_tokens)


def _resolve_message_request_context(
    request: MessagesRequest,
    auth_result: tuple[APIKey | None, bool],
) -> MessageRequestContext:
    api_key, is_master_key = auth_result
    user_from_metadata = request.metadata.get("user_id") if request.metadata else None
    user_id = resolve_user_id(
        user_id_from_request=str(user_from_metadata) if user_from_metadata else None,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=_anthropic_error(
            _ERR_INVALID_REQUEST,
            _MASTER_KEY_USER_REQUIRED,
            status.HTTP_400_BAD_REQUEST,
        ),
        no_api_key_error=_anthropic_error(
            _ERR_API,
            _API_KEY_VALIDATION_FAILED,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ),
        no_user_error=_anthropic_error(
            _ERR_API,
            _API_KEY_NO_USER,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ),
    )
    return MessageRequestContext(
        api_key_id=api_key.id if api_key else None,
        user_id=user_id,
    )


async def _log_message_usage(
    *,
    db: AsyncSession,
    log_writer: LogWriter,
    message_context: MessageRequestContext,
    model: str,
    provider: Any,
    usage_data: CompletionUsage | None = None,
    error: str | None = None,
) -> None:
    await log_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=message_context.api_key_id,
        model=model,
        provider=provider,
        endpoint=_MESSAGES_ENDPOINT,
        user_id=message_context.user_id,
        usage_override=usage_data,
        error=error,
    )


def _format_message_stream_chunk(event: MessageStreamEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"


def _message_stream_event_usage(event: MessageStreamEvent) -> CompletionUsage | None:
    if isinstance(event, MessageDeltaEvent):
        return _message_completion_usage(event.usage.input_tokens, event.usage.output_tokens)
    if isinstance(event, MessageStartEvent):
        input_tokens = event.message.usage.input_tokens or 0
        if input_tokens:
            return _message_completion_usage(input_tokens, 0)
    return None


def _message_streaming_response(
    *,
    stream_result: Any,
    db: AsyncSession,
    log_writer: LogWriter,
    message_context: MessageRequestContext,
    model: str,
    provider: Any,
    rate_limit_info: RateLimitInfo | None,
) -> StreamingResponse:
    async def _on_complete(usage_data: CompletionUsage) -> None:
        await _log_message_usage(
            db=db,
            log_writer=log_writer,
            message_context=message_context,
            model=model,
            provider=provider,
            usage_data=usage_data,
        )

    async def _on_error(error: str) -> None:
        await _log_message_usage(
            db=db,
            log_writer=log_writer,
            message_context=message_context,
            model=model,
            provider=provider,
            error=error,
        )

    return StreamingResponse(
        streaming_generator(
            stream=stream_result,
            format_chunk=_format_message_stream_chunk,
            extract_usage=_message_stream_event_usage,
            fmt=ANTHROPIC_STREAM_FORMAT,
            on_complete=_on_complete,
            on_error=_on_error,
            label=f"{provider}:{model}",
        ),
        media_type="text/event-stream",
        headers=optional_rate_limit_headers(rate_limit_info),
    )


@router.post("/messages", response_model=None)
async def create_message(
    raw_request: Request,
    response: Response,
    request: MessagesRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any] | StreamingResponse:
    """Anthropic Messages API-compatible endpoint."""
    message_context = _resolve_message_request_context(request, auth_result)

    rate_limit_info = check_rate_limit(raw_request, message_context.user_id)

    await validate_user_request_budget(db, message_context.user_id, request.model, strategy=config.budget_strategy)

    provider, model = AnyLLM.split_model_provider(request.model)

    provider_kwargs = get_provider_kwargs(config, provider)

    # Request fields take precedence over provider config defaults
    request_fields = request.model_dump(exclude_unset=True)
    call_kwargs: dict[str, Any] = {**provider_kwargs, **request_fields}

    try:
        if request.stream:
            call_kwargs["stream"] = True
            msg_stream = await amessages(**call_kwargs)
            return _message_streaming_response(
                stream_result=msg_stream,
                db=db,
                log_writer=log_writer,
                message_context=message_context,
                model=model,
                provider=provider,
                rate_limit_info=rate_limit_info,
            )

        result: MessageResponse = await amessages(**call_kwargs)  # type: ignore[assignment]

        usage_data = _message_response_usage(result)
        if usage_data:
            await _log_message_usage(
                db=db,
                log_writer=log_writer,
                message_context=message_context,
                model=model,
                provider=provider,
                usage_data=usage_data,
            )

    except HTTPException:
        raise
    except Exception as e:
        await _log_message_usage(
            db=db,
            log_writer=log_writer,
            message_context=message_context,
            model=model,
            provider=provider,
            error=str(e),
        )
        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise _anthropic_error(
            _ERR_API,
            _PROVIDER_ERROR,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e

    apply_rate_limit_headers(response, rate_limit_info)

    return result.model_dump(exclude_none=True)

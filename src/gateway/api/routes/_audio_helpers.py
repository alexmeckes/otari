from dataclasses import dataclass
from typing import Any, NoReturn

from fastapi import HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._provider_context import resolve_openai_provider_request_context
from gateway.api.routes._usage import make_usage_log
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.rate_limit import RateLimitInfo
from gateway.services.log_writer import LogWriter

_PROVIDER_ERROR_DETAIL = "The request could not be completed by the provider"


@dataclass(frozen=True)
class AudioRequestContext:
    api_key_id: str | None
    user_id: str
    rate_limit_info: RateLimitInfo | None


async def log_audio_usage(
    *,
    log_writer: LogWriter,
    context: AudioRequestContext,
    model: str,
    provider: Any,
    endpoint: str,
    error: str | None = None,
) -> None:
    usage_log = make_usage_log(
        api_key_id=context.api_key_id,
        user_id=context.user_id,
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
        prompt_tokens=0 if error is None else None,
        completion_tokens=0 if error is None else None,
        total_tokens=0 if error is None else None,
    )
    await log_writer.put(usage_log)


async def audio_provider_call_context(
    *,
    raw_request: Request,
    auth_result: tuple[APIKey | None, bool],
    user: str | None,
    db: AsyncSession,
    config: GatewayConfig,
    model: str,
) -> tuple[AudioRequestContext, Any, str, dict[str, Any]]:
    provider_context = await resolve_openai_provider_request_context(
        raw_request=raw_request,
        auth_result=auth_result,
        user=user,
        db=db,
        config=config,
        model=model,
    )
    context = AudioRequestContext(
        api_key_id=provider_context.api_key_id,
        user_id=provider_context.user_id,
        rate_limit_info=provider_context.rate_limit_info,
    )
    return context, provider_context.provider, provider_context.model, provider_context.provider_kwargs


async def raise_audio_provider_error(
    *,
    log_writer: LogWriter,
    context: AudioRequestContext,
    model: str,
    provider: Any,
    endpoint: str,
    error: Exception,
) -> NoReturn:
    await log_audio_usage(
        log_writer=log_writer,
        context=context,
        model=model,
        provider=provider,
        endpoint=endpoint,
        error=str(error),
    )
    logger.error("Provider call failed for %s:%s: %s", provider, model, error)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=_PROVIDER_ERROR_DETAIL,
    ) from error

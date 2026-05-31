from typing import NoReturn

from fastapi import HTTPException, status

from gateway.api.routes._provider_context import OpenAIProviderRequestContext
from gateway.log_config import logger
from gateway.services.log_writer import LogWriter

_PROVIDER_ERROR_DETAIL = "The request could not be completed by the provider"


async def log_audio_usage(
    *,
    log_writer: LogWriter,
    context: OpenAIProviderRequestContext,
    endpoint: str,
    error: str | None = None,
) -> None:
    usage_log = context.usage_log(
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
        prompt_tokens=0 if error is None else None,
        completion_tokens=0 if error is None else None,
        total_tokens=0 if error is None else None,
    )
    await log_writer.put(usage_log)


async def raise_audio_provider_error(
    *,
    log_writer: LogWriter,
    context: OpenAIProviderRequestContext,
    endpoint: str,
    error: Exception,
) -> NoReturn:
    await log_audio_usage(
        log_writer=log_writer,
        context=context,
        endpoint=endpoint,
        error=str(error),
    )
    logger.error("Provider call failed for %s:%s: %s", context.provider, context.model, error)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=_PROVIDER_ERROR_DETAIL,
    ) from error

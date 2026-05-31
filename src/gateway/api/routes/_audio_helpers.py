import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NoReturn

from any_llm import AnyLLM
from fastapi import HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._helpers import resolve_user_id
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, UsageLog
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.budget_service import validate_user_budget
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import get_provider_kwargs

_PROVIDER_ERROR_DETAIL = "The request could not be completed by the provider"


@dataclass(frozen=True)
class AudioRequestContext:
    api_key_id: str | None
    user_id: str
    rate_limit_info: RateLimitInfo | None


def resolve_audio_request_context(
    *,
    raw_request: Request,
    auth_result: tuple[APIKey | None, bool],
    user: str | None,
) -> AudioRequestContext:
    api_key, is_master_key = auth_result
    user_id = resolve_user_id(
        user_id_from_request=user,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="When using master key, 'user' field is required in request body",
        ),
        no_api_key_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key validation failed",
        ),
        no_user_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key has no associated user",
        ),
    )
    return AudioRequestContext(
        api_key_id=api_key.id if api_key else None,
        user_id=user_id,
        rate_limit_info=check_rate_limit(raw_request, user_id),
    )


async def log_audio_usage(
    *,
    log_writer: LogWriter,
    context: AudioRequestContext,
    model: str,
    provider: Any,
    endpoint: str,
    error: str | None = None,
) -> None:
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        api_key_id=context.api_key_id,
        user_id=context.user_id,
        timestamp=datetime.now(UTC),
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
    )
    if error is None:
        # Audio endpoints do not expose measurable usage units yet, so cost is
        # left unset until a dedicated pricing metric is available.
        usage_log.prompt_tokens = 0
        usage_log.completion_tokens = 0
        usage_log.total_tokens = 0
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
    context = resolve_audio_request_context(
        raw_request=raw_request,
        auth_result=auth_result,
        user=user,
    )
    _ = await validate_user_budget(db, context.user_id, model, strategy=config.budget_strategy)
    if config.budget_strategy == "for_update":
        await db.rollback()

    provider, model_name = AnyLLM.split_model_provider(model)
    return context, provider, model_name, get_provider_kwargs(config, provider)


def with_optional_kwargs(call_kwargs: dict[str, Any], **optional: Any) -> dict[str, Any]:
    call_kwargs.update({key: value for key, value in optional.items() if value is not None})
    return call_kwargs


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

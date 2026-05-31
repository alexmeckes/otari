from dataclasses import dataclass
from typing import Any, NoReturn

from any_llm import AnyLLM
from fastapi import HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._budget_checks import validate_user_request_budget
from gateway.api.routes._helpers import resolve_openai_user_id
from gateway.api.routes._usage import make_usage_log
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.rate_limit import RateLimitInfo, check_rate_limit
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
    user_id = resolve_openai_user_id(
        user_id_from_request=user,
        api_key=api_key,
        is_master_key=is_master_key,
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
    context = resolve_audio_request_context(
        raw_request=raw_request,
        auth_result=auth_result,
        user=user,
    )
    await validate_user_request_budget(db, context.user_id, model, strategy=config.budget_strategy)

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

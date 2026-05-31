import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any, NoReturn

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionUsage
from fastapi import HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.metrics import record_cost, record_tokens
from gateway.models.entities import UsageLog
from gateway.rate_limit import RateLimitInfo
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import (
    find_model_pricing,
    log_missing_pricing,
    pricing_model_ref,
    token_usage_cost,
)


def rate_limit_headers(info: RateLimitInfo) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(info.limit),
        "X-RateLimit-Remaining": str(info.remaining),
        "X-RateLimit-Reset": str(int(info.reset)),
    }


def optional_rate_limit_headers(info: RateLimitInfo | None) -> dict[str, str]:
    if info is None:
        return {}
    return rate_limit_headers(info)


def apply_rate_limit_headers(response: Response, info: RateLimitInfo | None) -> None:
    for key, value in optional_rate_limit_headers(info).items():
        response.headers[key] = value


def provider_model_label(provider: Any, model: str) -> str:
    return pricing_model_ref(str(provider), model)


def make_usage_log(
    *,
    api_key_id: str | None,
    user_id: str | None,
    model: str,
    provider: Any,
    endpoint: str,
    project_id: str | None = None,
    status: str = "success",
    error_message: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    tags: Mapping[str, Any] | None = None,
) -> UsageLog:
    return UsageLog(
        id=str(uuid.uuid4()),
        api_key_id=api_key_id,
        user_id=user_id,
        project_id=project_id,
        timestamp=datetime.now(UTC),
        model=model,
        provider=provider,
        endpoint=endpoint,
        status=status,
        error_message=error_message,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tags=dict(tags) if tags is not None else {},
    )


async def log_usage(
    db: AsyncSession,
    log_writer: LogWriter,
    api_key_id: str | None,
    model: str,
    provider: str | None,
    endpoint: str,
    user_id: str | None = None,
    project_id: str | None = None,
    tags: Mapping[str, Any] | None = None,
    response: ChatCompletion | AsyncIterator[ChatCompletionChunk] | None = None,
    usage_override: CompletionUsage | None = None,
    error: str | None = None,
) -> None:
    """Log API usage to database and update user spend."""
    usage_log = make_usage_log(
        api_key_id=api_key_id,
        user_id=user_id,
        project_id=project_id,
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
        tags=dict(tags) if tags is not None else {},
    )

    usage_data = usage_override
    if not usage_data and response and isinstance(response, ChatCompletion) and response.usage:
        usage_data = response.usage

    if usage_data:
        usage_log.prompt_tokens = usage_data.prompt_tokens
        usage_log.completion_tokens = usage_data.completion_tokens
        usage_log.total_tokens = usage_data.total_tokens

        record_tokens(
            str(provider or ""),
            model,
            usage_data.prompt_tokens,
            usage_data.completion_tokens,
        )

        pricing = await find_model_pricing(db, provider, model, as_of=usage_log.timestamp)
        if pricing:
            cost = token_usage_cost(
                pricing,
                prompt_tokens=usage_data.prompt_tokens,
                completion_tokens=usage_data.completion_tokens,
            )
            usage_log.cost = cost
            record_cost(str(provider or ""), model, cost)
        else:
            log_missing_pricing(provider, model)

    await log_writer.put(usage_log)


async def log_usage_error(
    log_writer: LogWriter,
    *,
    api_key_id: str | None,
    user_id: str | None,
    model: str,
    provider: Any,
    endpoint: str,
    error: BaseException,
) -> None:
    await log_writer.put(
        make_usage_log(
            api_key_id=api_key_id,
            user_id=user_id,
            model=model,
            provider=provider,
            endpoint=endpoint,
            status="error",
            error_message=str(error),
        )
    )


async def log_and_raise_provider_error(
    log_writer: LogWriter,
    *,
    api_key_id: str | None,
    user_id: str | None,
    model: str,
    provider: Any,
    endpoint: str,
    error: BaseException,
) -> NoReturn:
    await log_usage_error(
        log_writer,
        api_key_id=api_key_id,
        user_id=user_id,
        model=model,
        provider=provider,
        endpoint=endpoint,
        error=error,
    )
    logger.error("Provider call failed for %s: %s", provider_model_label(provider, model), error)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="The request could not be completed by the provider",
    ) from error

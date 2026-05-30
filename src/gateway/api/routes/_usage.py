import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionUsage
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.metrics import record_cost, record_tokens
from gateway.models.entities import UsageLog
from gateway.rate_limit import RateLimitInfo
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing


def rate_limit_headers(info: RateLimitInfo) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(info.limit),
        "X-RateLimit-Remaining": str(info.remaining),
        "X-RateLimit-Reset": str(int(info.reset)),
    }


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
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        api_key_id=api_key_id,
        user_id=user_id,
        project_id=project_id,
        timestamp=datetime.now(UTC),
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
            cost = (usage_data.prompt_tokens / 1_000_000) * pricing.input_price_per_million + (
                usage_data.completion_tokens / 1_000_000
            ) * pricing.output_price_per_million
            usage_log.cost = cost
            record_cost(str(provider or ""), model, cost)
        else:
            model_ref = f"{provider}:{model}" if provider else model
            logger.warning("No pricing configured for '%s'. Usage will be tracked without cost.", model_ref)

    await log_writer.put(usage_log)

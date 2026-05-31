from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from any_llm import AnyLLM
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._budget_checks import validate_user_request_budget
from gateway.api.routes._helpers import resolve_openai_user_id
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.provider_kwargs import get_provider_kwargs


@dataclass(frozen=True)
class OpenAIProviderRequestContext:
    api_key_id: str | None
    user_id: str
    rate_limit_info: RateLimitInfo | None
    provider: Any
    model: str
    provider_kwargs: dict[str, Any]


async def resolve_openai_provider_request_context(
    *,
    raw_request: Request,
    auth_result: tuple[APIKey | None, bool],
    user: str | None,
    db: AsyncSession,
    config: GatewayConfig,
    model: str,
) -> OpenAIProviderRequestContext:
    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None
    user_id = resolve_openai_user_id(
        user_id_from_request=user,
        api_key=api_key,
        is_master_key=is_master_key,
    )
    rate_limit_info = check_rate_limit(raw_request, user_id)

    await validate_user_request_budget(db, user_id, model, strategy=config.budget_strategy)

    provider, model_name = AnyLLM.split_model_provider(model)
    return OpenAIProviderRequestContext(
        api_key_id=api_key_id,
        user_id=user_id,
        rate_limit_info=rate_limit_info,
        provider=provider,
        model=model_name,
        provider_kwargs=get_provider_kwargs(config, provider),
    )

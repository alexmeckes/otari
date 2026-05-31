from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from any_llm import AnyLLM
from fastapi import Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._budget_checks import validate_user_request_budget
from gateway.api.routes._helpers import resolve_openai_user_id
from gateway.api.routes._usage import (
    apply_rate_limit_headers,
    log_and_raise_provider_error,
    log_usage_error,
    make_usage_log,
    rate_limit_headers,
)
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, UsageLog
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing, log_missing_pricing
from gateway.services.provider_kwargs import get_provider_kwargs


@dataclass(frozen=True)
class OpenAIProviderRequestContext:
    api_key_id: str | None
    user_id: str
    rate_limit_info: RateLimitInfo | None
    provider: Any
    model: str
    provider_kwargs: dict[str, Any]

    def call_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            **kwargs,
            **self.provider_kwargs,
        }

    def rate_limit_headers(self) -> dict[str, str]:
        if self.rate_limit_info is None:
            return {}
        return rate_limit_headers(self.rate_limit_info)

    def apply_rate_limit_headers(self, response: Response) -> None:
        apply_rate_limit_headers(response, self.rate_limit_info)

    def usage_log(
        self,
        *,
        endpoint: str,
        project_id: str | None = None,
        status: str = "success",
        error_message: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> UsageLog:
        return make_usage_log(
            api_key_id=self.api_key_id,
            user_id=self.user_id,
            project_id=project_id,
            model=self.model,
            provider=self.provider,
            endpoint=endpoint,
            status=status,
            error_message=error_message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tags=tags,
        )

    async def log_usage_error(self, log_writer: LogWriter, *, endpoint: str, error: BaseException) -> None:
        await log_usage_error(
            log_writer,
            api_key_id=self.api_key_id,
            user_id=self.user_id,
            model=self.model,
            provider=self.provider,
            endpoint=endpoint,
            error=error,
        )

    async def log_and_raise_provider_error(
        self,
        log_writer: LogWriter,
        *,
        endpoint: str,
        error: BaseException,
    ) -> NoReturn:
        await log_and_raise_provider_error(
            log_writer,
            api_key_id=self.api_key_id,
            user_id=self.user_id,
            model=self.model,
            provider=self.provider,
            endpoint=endpoint,
            error=error,
        )

    async def apply_input_token_cost(
        self,
        db: AsyncSession,
        usage_log: UsageLog,
        *,
        token_count: int | None,
        require_positive_tokens: bool = False,
    ) -> None:
        await self.apply_input_metered_cost(
            db,
            usage_log,
            units=token_count,
            require_positive_units=require_positive_tokens,
        )

    async def apply_input_metered_cost(
        self,
        db: AsyncSession,
        usage_log: UsageLog,
        *,
        units: float | None,
        price_divisor: float = 1_000_000,
        require_positive_units: bool = False,
        missing_cost: float | None = None,
        warn_missing_pricing: bool = True,
    ) -> None:
        pricing = await find_model_pricing(db, self.provider, self.model, as_of=usage_log.timestamp)
        if pricing:
            if units is not None and (units or not require_positive_units):
                usage_log.cost = (units / price_divisor) * pricing.input_price_per_million
        else:
            if missing_cost is not None:
                usage_log.cost = missing_cost
            if warn_missing_pricing:
                log_missing_pricing(self.provider, self.model)


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

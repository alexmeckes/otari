"""Rerank endpoint — reorder documents by relevance to a query."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from any_llm import AnyLLM, arerank
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._budget_checks import validate_user_request_budget
from gateway.api.routes._helpers import resolve_openai_user_id
from gateway.api.routes._rerank_models import RerankRequest
from gateway.api.routes._usage import apply_rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, UsageLog
from gateway.rate_limit import check_rate_limit
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing, log_missing_pricing
from gateway.services.provider_kwargs import get_provider_kwargs

router = APIRouter(prefix="/v1", tags=["rerank"])


@router.post("/rerank", response_model=None)
async def create_rerank(
    raw_request: Request,
    response: Response,
    request: RerankRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> Any:
    """Rerank documents by relevance to a query.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """
    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None

    user_id = resolve_openai_user_id(
        user_id_from_request=request.user,
        api_key=api_key,
        is_master_key=is_master_key,
    )

    rate_limit_info = check_rate_limit(raw_request, user_id)

    await validate_user_request_budget(db, user_id, request.model, strategy=config.budget_strategy)

    provider, model = AnyLLM.split_model_provider(request.model)

    provider_kwargs = get_provider_kwargs(config, provider)

    rerank_kwargs: dict[str, Any] = {
        "model": model,
        "query": request.query,
        "documents": request.documents,
        "provider": provider,
        **provider_kwargs,
    }
    if request.top_n is not None:
        rerank_kwargs["top_n"] = request.top_n
    if request.max_tokens_per_doc is not None:
        rerank_kwargs["max_tokens_per_doc"] = request.max_tokens_per_doc

    try:
        result = await arerank(**rerank_kwargs)

        total_tokens = result.usage.total_tokens if result.usage else None

        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=provider,
            endpoint="/v1/rerank",
            status="success",
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
        )

        if result.usage:
            pricing = await find_model_pricing(db, provider, model, as_of=usage_log.timestamp)
            if pricing and total_tokens:
                cost = (total_tokens / 1_000_000) * pricing.input_price_per_million
                usage_log.cost = cost
            elif not pricing:
                log_missing_pricing(provider, model)

        await log_writer.put(usage_log)

    except HTTPException:
        raise
    except Exception as e:
        error_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=provider,
            endpoint="/v1/rerank",
            status="error",
            error_message=str(e),
        )
        await log_writer.put(error_log)

        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e

    apply_rate_limit_headers(response, rate_limit_info)

    return result

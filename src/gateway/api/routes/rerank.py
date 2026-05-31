"""Rerank endpoint — reorder documents by relevance to a query."""

from typing import Annotated, Any

from any_llm import arerank
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import with_optional_kwargs
from gateway.api.routes._provider_context import resolve_openai_provider_request_context
from gateway.api.routes._rerank_models import RerankRequest
from gateway.api.routes._usage import apply_rate_limit_headers, log_and_raise_provider_error, make_usage_log
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing, log_missing_pricing

router = APIRouter(prefix="/v1", tags=["rerank"])

_RERANK_ENDPOINT = "/v1/rerank"


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
    context = await resolve_openai_provider_request_context(
        raw_request=raw_request,
        auth_result=auth_result,
        user=request.user,
        db=db,
        config=config,
        model=request.model,
    )

    rerank_kwargs = with_optional_kwargs(
        context.call_kwargs(query=request.query, documents=request.documents),
        top_n=request.top_n,
        max_tokens_per_doc=request.max_tokens_per_doc,
    )

    try:
        result = await arerank(**rerank_kwargs)

        total_tokens = result.usage.total_tokens if result.usage else None

        usage_log = make_usage_log(
            api_key_id=context.api_key_id,
            user_id=context.user_id,
            model=context.model,
            provider=context.provider,
            endpoint=_RERANK_ENDPOINT,
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
        )

        if result.usage:
            pricing = await find_model_pricing(db, context.provider, context.model, as_of=usage_log.timestamp)
            if pricing and total_tokens:
                cost = (total_tokens / 1_000_000) * pricing.input_price_per_million
                usage_log.cost = cost
            elif not pricing:
                log_missing_pricing(context.provider, context.model)

        await log_writer.put(usage_log)

    except HTTPException:
        raise
    except Exception as e:
        await log_and_raise_provider_error(
            log_writer,
            api_key_id=context.api_key_id,
            user_id=context.user_id,
            model=context.model,
            provider=context.provider,
            endpoint=_RERANK_ENDPOINT,
            error=e,
        )

    apply_rate_limit_headers(response, context.rate_limit_info)

    return result

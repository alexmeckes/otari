"""Rerank endpoint — reorder documents by relevance to a query."""

from typing import Annotated, Any

from any_llm import arerank
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._provider_context import resolve_openai_provider_request_context
from gateway.api.routes._rerank_models import RerankRequest
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter

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
    context = await resolve_openai_provider_request_context(
        raw_request=raw_request,
        auth_result=auth_result,
        user=request.user,
        db=db,
        config=config,
        model=request.model,
    )

    rerank_kwargs = context.call_kwargs(
        query=request.query,
        documents=request.documents,
        optional={
            "top_n": request.top_n,
            "max_tokens_per_doc": request.max_tokens_per_doc,
        },
    )

    try:
        result = await arerank(**rerank_kwargs)

        usage = result.usage
        total_tokens = usage.total_tokens if usage else None
        await context.log_input_metered_usage(
            db,
            log_writer,
            endpoint="/v1/rerank",
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
            cost_units=total_tokens,
            apply_cost=usage is not None,
            require_positive_units=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        await context.log_and_raise_provider_error(
            log_writer,
            endpoint="/v1/rerank",
            error=e,
        )

    context.apply_rate_limit_headers(response)

    return result

"""OpenAI-compatible embeddings endpoint."""

from typing import Annotated

from any_llm import aembedding
from any_llm.types.completion import CreateEmbeddingResponse
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._embedding_models import EmbeddingRequest
from gateway.api.routes._helpers import with_optional_kwargs
from gateway.api.routes._provider_context import resolve_openai_provider_request_context
from gateway.api.routes._usage import apply_rate_limit_headers, log_and_raise_provider_error, make_usage_log
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing, log_missing_pricing

router = APIRouter(prefix="/v1", tags=["embeddings"])

_EMBEDDINGS_ENDPOINT = "/v1/embeddings"


@router.post("/embeddings", response_model=None)
async def create_embedding(
    raw_request: Request,
    response: Response,
    request: EmbeddingRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> CreateEmbeddingResponse:
    """OpenAI-compatible embeddings endpoint.

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

    embedding_kwargs = with_optional_kwargs(
        {
            "model": context.model,
            "inputs": request.input,
            "provider": context.provider,
            **context.provider_kwargs,
        },
        encoding_format=request.encoding_format,
        dimensions=request.dimensions,
    )

    try:
        result = await aembedding(**embedding_kwargs)

        usage_log = make_usage_log(
            api_key_id=context.api_key_id,
            user_id=context.user_id,
            model=context.model,
            provider=context.provider,
            endpoint=_EMBEDDINGS_ENDPOINT,
            prompt_tokens=result.usage.prompt_tokens if result.usage else None,
            completion_tokens=0,
            total_tokens=result.usage.total_tokens if result.usage else None,
        )

        if result.usage:
            pricing = await find_model_pricing(db, context.provider, context.model, as_of=usage_log.timestamp)
            if pricing:
                cost = (result.usage.prompt_tokens / 1_000_000) * pricing.input_price_per_million
                usage_log.cost = cost
            else:
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
            endpoint=_EMBEDDINGS_ENDPOINT,
            error=e,
        )

    apply_rate_limit_headers(response, context.rate_limit_info)

    return result

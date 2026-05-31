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
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter

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
        context.call_kwargs(inputs=request.input),
        encoding_format=request.encoding_format,
        dimensions=request.dimensions,
    )

    try:
        result = await aembedding(**embedding_kwargs)

        usage_log = context.usage_log(
            endpoint=_EMBEDDINGS_ENDPOINT,
            prompt_tokens=result.usage.prompt_tokens if result.usage else None,
            completion_tokens=0,
            total_tokens=result.usage.total_tokens if result.usage else None,
        )

        if result.usage:
            await context.apply_input_metered_cost(
                db,
                usage_log,
                units=result.usage.prompt_tokens,
            )

        await log_writer.put(usage_log)

    except HTTPException:
        raise
    except Exception as e:
        await context.log_and_raise_provider_error(
            log_writer,
            endpoint=_EMBEDDINGS_ENDPOINT,
            error=e,
        )

    context.apply_rate_limit_headers(response)

    return result

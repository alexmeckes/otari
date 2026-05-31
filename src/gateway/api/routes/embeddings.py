"""OpenAI-compatible embeddings endpoint."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from any_llm import AnyLLM, aembedding
from any_llm.types.completion import CreateEmbeddingResponse
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._budget_checks import validate_user_request_budget
from gateway.api.routes._embedding_models import EmbeddingRequest
from gateway.api.routes._helpers import resolve_openai_user_id
from gateway.api.routes._usage import apply_rate_limit_headers, log_usage_error
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, UsageLog
from gateway.rate_limit import check_rate_limit
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing, log_missing_pricing
from gateway.services.provider_kwargs import get_provider_kwargs

router = APIRouter(prefix="/v1", tags=["embeddings"])


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

    embedding_kwargs: dict[str, Any] = {
        "model": model,
        "inputs": request.input,
        "provider": provider,
        **provider_kwargs,
    }
    if request.encoding_format is not None:
        embedding_kwargs["encoding_format"] = request.encoding_format
    if request.dimensions is not None:
        embedding_kwargs["dimensions"] = request.dimensions

    try:
        result = await aembedding(**embedding_kwargs)

        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=provider,
            endpoint="/v1/embeddings",
            status="success",
            prompt_tokens=result.usage.prompt_tokens if result.usage else None,
            completion_tokens=0,
            total_tokens=result.usage.total_tokens if result.usage else None,
        )

        if result.usage:
            pricing = await find_model_pricing(db, provider, model, as_of=usage_log.timestamp)
            if pricing:
                cost = (result.usage.prompt_tokens / 1_000_000) * pricing.input_price_per_million
                usage_log.cost = cost
            else:
                log_missing_pricing(provider, model)

        await log_writer.put(usage_log)

    except HTTPException:
        raise
    except Exception as e:
        await log_usage_error(
            log_writer,
            api_key_id=api_key_id,
            user_id=user_id,
            model=model,
            provider=provider,
            endpoint="/v1/embeddings",
            error=e,
        )

        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e

    apply_rate_limit_headers(response, rate_limit_info)

    return result

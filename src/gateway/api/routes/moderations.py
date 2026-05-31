"""OpenAI-compatible moderations endpoint."""

from typing import Annotated

from any_llm import amoderation
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._moderation_models import ModerationRequest
from gateway.api.routes._provider_context import resolve_openai_provider_request_context
from gateway.api.routes._usage import apply_rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing
from gateway.types.moderation import ModerationResponse

# Locked phrasing — cross-SDK error contract. Do not reword.
UNSUPPORTED_MODERATION_SUBSTRING = "does not support moderation"

router = APIRouter(prefix="/v1", tags=["moderations"])

_MODERATIONS_ENDPOINT = "/v1/moderations"


@router.post("/moderations", response_model=ModerationResponse)
async def create_moderation(
    raw_request: Request,
    response: Response,
    request: ModerationRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    include_raw: Annotated[bool, Query()] = False,
) -> ModerationResponse:
    """OpenAI-compatible moderations endpoint.

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

    moderation_kwargs = context.call_kwargs(input=request.input, include_raw=include_raw)

    try:
        result = await amoderation(**moderation_kwargs)

        usage_log = context.usage_log(
            endpoint=_MODERATIONS_ENDPOINT,
            prompt_tokens=None,
            completion_tokens=0,
            total_tokens=None,
        )

        pricing = await find_model_pricing(db, context.provider, context.model, as_of=usage_log.timestamp)
        if pricing and pricing.input_price_per_million:
            # Flat per-request rate stored as input_price_per_million (moderation has no token usage).
            usage_log.cost = pricing.input_price_per_million / 1_000_000
        else:
            usage_log.cost = 0.0
            # Intentionally do NOT emit "No pricing configured" warning for
            # moderations (free at most providers; keeps logs clean).

        await log_writer.put(usage_log)

    except HTTPException:
        raise
    except NotImplementedError as e:
        await context.log_usage_error(
            log_writer,
            endpoint=_MODERATIONS_ENDPOINT,
            error=e,
        )
        if UNSUPPORTED_MODERATION_SUBSTRING in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        logger.error("Provider implementation gap for %s:%s: %s", context.provider, context.model, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e
    except Exception as e:
        await context.log_and_raise_provider_error(
            log_writer,
            endpoint=_MODERATIONS_ENDPOINT,
            error=e,
        )

    apply_rate_limit_headers(response, context.rate_limit_info)

    return result

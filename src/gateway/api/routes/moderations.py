"""OpenAI-compatible moderations endpoint."""

from typing import Annotated

from any_llm import amoderation
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._moderation_models import ModerationRequest
from gateway.api.routes._provider_context import resolve_openai_provider_request_context
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter
from gateway.types.moderation import ModerationResponse

router = APIRouter(prefix="/v1", tags=["moderations"])


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

        # Moderation has no token usage; missing pricing is intentionally treated as free.
        await context.log_input_metered_usage(
            db,
            log_writer,
            endpoint="/v1/moderations",
            prompt_tokens=None,
            total_tokens=None,
            cost_units=1,
            apply_cost=True,
            missing_cost=0.0,
            warn_missing_pricing=False,
        )

    except HTTPException:
        raise
    except NotImplementedError as e:
        await context.log_usage_error(
            log_writer,
            endpoint="/v1/moderations",
            error=e,
        )
        # Locked phrasing — cross-SDK error contract. Do not reword.
        if "does not support moderation" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        logger.error("Provider implementation gap for %s: %s", context.provider_label, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e
    except Exception as e:
        await context.log_and_raise_provider_error(
            log_writer,
            endpoint="/v1/moderations",
            error=e,
        )

    context.apply_rate_limit_headers(response)

    return result

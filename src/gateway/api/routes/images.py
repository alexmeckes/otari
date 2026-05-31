"""OpenAI-compatible image generation endpoint."""

from typing import Annotated, Any

from any_llm import aimage_generation
from any_llm.types.image import ImagesResponse
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import with_optional_kwargs
from gateway.api.routes._image_models import ImageGenerationRequest
from gateway.api.routes._provider_context import resolve_openai_provider_request_context
from gateway.api.routes._usage import apply_rate_limit_headers, log_and_raise_provider_error
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing, log_missing_pricing

router = APIRouter(prefix="/v1", tags=["images"])

_IMAGE_GENERATIONS_ENDPOINT = "/v1/images/generations"


@router.post("/images/generations", response_model=None)
async def create_image(
    raw_request: Request,
    response: Response,
    request: ImageGenerationRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """OpenAI-compatible image generation endpoint.

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

    image_kwargs = with_optional_kwargs(
        context.call_kwargs(prompt=request.prompt),
        n=request.n,
        size=request.size,
        quality=request.quality,
        style=request.style,
        response_format=request.response_format,
    )

    try:
        result: ImagesResponse = await aimage_generation(**image_kwargs)

        n_images = len(result.data) if result.data else (request.n or 1)

        usage_log = context.usage_log(
            endpoint=_IMAGE_GENERATIONS_ENDPOINT,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        # Image pricing: repurpose input_price_per_million as price-per-image
        pricing = await find_model_pricing(db, context.provider, context.model, as_of=usage_log.timestamp)
        if pricing:
            cost = n_images * pricing.input_price_per_million
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
            endpoint=_IMAGE_GENERATIONS_ENDPOINT,
            error=e,
        )

    apply_rate_limit_headers(response, context.rate_limit_info)

    return result.model_dump()

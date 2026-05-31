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
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter

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

        usage_log = context.zero_token_usage_log(endpoint=_IMAGE_GENERATIONS_ENDPOINT)

        # Image pricing stores price-per-image in input_price_per_million.
        await context.apply_input_metered_cost(
            db,
            usage_log,
            units=n_images,
            price_divisor=1,
        )

        await log_writer.put(usage_log)

    except HTTPException:
        raise
    except Exception as e:
        await context.log_and_raise_provider_error(
            log_writer,
            endpoint=_IMAGE_GENERATIONS_ENDPOINT,
            error=e,
        )

    context.apply_rate_limit_headers(response)

    return result.model_dump()

from typing import Annotated, Any

from any_llm import AnyLLM, aresponses
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi import Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._budget_checks import validate_scoped_request_budgets
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes._responses_native import (
    log_native_response_usage,
    native_response_call_kwargs,
    native_response_streaming_response,
)
from gateway.api.routes._responses_transform import (
    ResponsesRequest,
    chat_completion_to_response_payload,
    chat_request_from_response_request,
    metadata_from_model_selector,
    response_payload_with_served_metadata,
    served_metadata,
    set_served_headers,
    usage_to_completion_usage,
)
from gateway.api.routes._usage import apply_rate_limit_headers
from gateway.api.routes.chat import (
    chat_completions,
)
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.rate_limit import check_rate_limit
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import get_provider_kwargs
from gateway.services.routing_policy_service import DEFAULT_ROUTING_MODEL

router = APIRouter(prefix="/v1", tags=["responses"])

_MASTER_KEY_USER_REQUIRED = "When using master key, 'user' field is required in request body"
_API_KEY_VALIDATION_FAILED = "API key validation failed"
_API_KEY_NO_USER = "API key has no associated user"


@router.post("/responses", response_model=None)
async def create_response(
    raw_request: Request,
    response: FastAPIResponse,
    background_tasks: BackgroundTasks,
    request_body: ResponsesRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any] | StreamingResponse:
    """OpenAI-compatible Responses endpoint."""
    if not config.is_platform_mode and request_body.model == DEFAULT_ROUTING_MODEL:
        if request_body.stream:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Routing policies do not support streaming responses yet",
            )
        chat_request = chat_request_from_response_request(request_body)
        chat_result = await chat_completions(
            raw_request=raw_request,
            response=response,
            background_tasks=background_tasks,
            request=chat_request,
            db=db,
            config=config,
            log_writer=log_writer,
        )
        if isinstance(chat_result, StreamingResponse):
            return chat_result
        metadata = metadata_from_model_selector(chat_result.model)
        if metadata is not None:
            set_served_headers(response, metadata)
        return chat_completion_to_response_payload(chat_result)

    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None

    user_id = resolve_user_id(
        user_id_from_request=request_body.user,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_MASTER_KEY_USER_REQUIRED,
        ),
        no_api_key_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_API_KEY_VALIDATION_FAILED,
        ),
        no_user_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_API_KEY_NO_USER,
        ),
    )

    rate_limit_info = check_rate_limit(raw_request, user_id)

    await validate_scoped_request_budgets(
        db,
        user_id=user_id,
        model=request_body.model,
        project_id=request_body.project_id,
        tags=request_body.tags,
        strategy=config.budget_strategy,
    )

    provider, model = AnyLLM.split_model_provider(request_body.model)
    provider_class = AnyLLM.get_provider_class(provider)
    if not getattr(provider_class, "SUPPORTS_RESPONSES", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider.value}' does not support the Responses API",
        )

    provider_kwargs = get_provider_kwargs(config, provider)
    call_kwargs, stream = native_response_call_kwargs(request_body, provider_kwargs, provider, model, user_id)

    try:
        if stream:
            call_kwargs["stream"] = True
            stream_result = await aresponses(**call_kwargs)
            return native_response_streaming_response(
                stream_result=stream_result,
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                provider=provider,
                model=model,
                user_id=user_id,
                request_body=request_body,
                rate_limit_info=rate_limit_info,
            )

        result = await aresponses(**call_kwargs)
        usage_data = usage_to_completion_usage(getattr(result, "usage", None))
        await log_native_response_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            provider=provider,
            model=model,
            user_id=user_id,
            request_body=request_body,
            usage_data=usage_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        await log_native_response_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            provider=provider,
            model=model,
            user_id=user_id,
            request_body=request_body,
            error=str(e),
        )
        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    apply_rate_limit_headers(response, rate_limit_info)

    metadata = served_metadata(provider.value, model)
    set_served_headers(response, metadata)
    payload = result.model_dump(exclude_none=True)  # type: ignore[union-attr]
    return response_payload_with_served_metadata(payload, provider=provider.value, requested_model=model)

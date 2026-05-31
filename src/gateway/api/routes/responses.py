from typing import Annotated, Any

from any_llm import AnyLLM, aresponses
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi import Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._provider_context import resolve_openai_provider_request_context
from gateway.api.routes._responses_native import (
    log_native_response_usage,
    native_response_call_kwargs,
    native_response_payload,
    native_response_streaming_response,
)
from gateway.api.routes._responses_transform import (
    ResponsesRequest,
    chat_completion_to_response_payload,
    chat_request_from_response_request,
    metadata_from_model_selector,
    set_served_headers,
    usage_to_completion_usage,
)
from gateway.api.routes.chat import (
    chat_completions,
)
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter
from gateway.services.routing_policy_service import DEFAULT_ROUTING_MODEL

router = APIRouter(prefix="/v1", tags=["responses"])


async def _run_default_routing_response(
    *,
    raw_request: Request,
    response: FastAPIResponse,
    background_tasks: BackgroundTasks,
    request_body: ResponsesRequest,
    db: AsyncSession,
    config: GatewayConfig,
    log_writer: LogWriter,
) -> dict[str, Any] | StreamingResponse:
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


async def _run_provider_native_response(
    *,
    raw_request: Request,
    response: FastAPIResponse,
    request_body: ResponsesRequest,
    auth_result: tuple[APIKey | None, bool],
    db: AsyncSession,
    config: GatewayConfig,
    log_writer: LogWriter,
) -> dict[str, Any] | StreamingResponse:
    context = await resolve_openai_provider_request_context(
        raw_request=raw_request,
        auth_result=auth_result,
        user=request_body.user,
        db=db,
        config=config,
        model=request_body.model,
        project_id=request_body.project_id,
        tags=request_body.tags,
    )

    provider_class = AnyLLM.get_provider_class(context.provider)
    if not getattr(provider_class, "SUPPORTS_RESPONSES", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{context.provider.value}' does not support the Responses API",
        )

    call_kwargs, stream = native_response_call_kwargs(request_body, context)

    try:
        if stream:
            call_kwargs["stream"] = True
            stream_result = await aresponses(**call_kwargs)
            return native_response_streaming_response(
                stream_result=stream_result,
                db=db,
                log_writer=log_writer,
                context=context,
                request_body=request_body,
            )

        result = await aresponses(**call_kwargs)
        usage_data = usage_to_completion_usage(getattr(result, "usage", None))
        await log_native_response_usage(
            db=db,
            log_writer=log_writer,
            context=context,
            request_body=request_body,
            usage_data=usage_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        await log_native_response_usage(
            db=db,
            log_writer=log_writer,
            context=context,
            request_body=request_body,
            error=str(e),
        )
        logger.error("Provider call failed for %s:%s: %s", context.provider, context.model, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    return native_response_payload(result=result, response=response, context=context)


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
        return await _run_default_routing_response(
            raw_request=raw_request,
            response=response,
            background_tasks=background_tasks,
            request_body=request_body,
            db=db,
            config=config,
            log_writer=log_writer,
        )

    return await _run_provider_native_response(
        raw_request=raw_request,
        response=response,
        request_body=request_body,
        auth_result=auth_result,
        db=db,
        config=config,
        log_writer=log_writer,
    )

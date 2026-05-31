"""Batch API endpoints for asynchronous LLM processing."""

import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypeVar

from any_llm import AnyLLM, LLMProvider
from any_llm.api import acancel_batch, acreate_batch, alist_batches, aretrieve_batch, aretrieve_batch_results
from any_llm.exceptions import BatchNotCompleteError, UnsupportedProviderError
from any_llm.types.batch import Batch
from fastapi import APIRouter, Depends, HTTPException, status

from gateway.api.deps import get_config, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._batch_models import BatchRequestItem, CreateBatchRequest
from gateway.api.routes._usage import make_usage_log
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import get_provider_kwargs

__all__ = ["BatchRequestItem", "CreateBatchRequest", "router"]

router = APIRouter(prefix="/v1/batches", tags=["batches"])

T = TypeVar("T")
ErrorLogger = Callable[[str], Awaitable[None]]


async def log_batch_usage(
    log_writer: LogWriter,
    api_key_id: str | None,
    model: str,
    provider: str,
    endpoint: str,
    user_id: str | None = None,
    error: str | None = None,
) -> None:
    """Log batch API usage."""
    usage_log = make_usage_log(
        api_key_id=api_key_id,
        user_id=user_id,
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
    )
    await log_writer.put(usage_log)


def _parse_provider(provider: str) -> LLMProvider:
    """Parse a provider string into an LLMProvider enum, raising 400 on invalid values."""
    try:
        return LLMProvider.from_string(provider)
    except UnsupportedProviderError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


def _split_batch_model(model: str) -> tuple[LLMProvider, str]:
    try:
        return AnyLLM.split_model_provider(model)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {e}",
        ) from e


def _ensure_batch_supported(provider: LLMProvider) -> None:
    provider_class = AnyLLM.get_provider_class(provider)
    if not getattr(provider_class, "SUPPORTS_BATCH", False):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Provider '{provider.value}' does not support batch operations",
        )


def _batch_provider_context(config: GatewayConfig, provider: str) -> tuple[LLMProvider, dict[str, Any]]:
    provider_enum = _parse_provider(provider)
    return provider_enum, get_provider_kwargs(config, provider_enum)


async def _run_batch_operation(
    *,
    action: str,
    provider: str,
    operation: Callable[..., Awaitable[T]],
    call_kwargs: dict[str, Any],
    on_error: ErrorLogger | None = None,
) -> T:
    try:
        return await operation(**call_kwargs)
    except HTTPException:
        raise
    except BatchNotCompleteError:
        raise
    except Exception as e:
        if on_error is not None:
            await on_error(str(e))
        logger.error("Batch %s failed for %s: %s", action, provider, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e


def _batch_response(batch: Batch, provider: str) -> dict[str, Any]:
    response_data = batch.model_dump()
    response_data["provider"] = provider
    return response_data


def _write_batch_input_file(request: CreateBatchRequest, model: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for req_item in request.requests:
            line = {
                "custom_id": req_item.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {**req_item.body, "model": model},
            }
            tmp.write(json.dumps(line) + "\n")
        return tmp.name


def _remove_batch_input_file(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        logger.warning("Failed to remove temp file %s", path)


@router.post("", response_model=None)
async def create_batch(
    request: CreateBatchRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """Create a batch of LLM requests for asynchronous processing."""
    api_key = auth_result[0]
    api_key_id = api_key.id if api_key else None
    user_id = api_key.user_id if api_key else None

    provider, model = _split_batch_model(request.model)
    _ensure_batch_supported(provider)
    provider_kwargs = get_provider_kwargs(config, provider)
    tmp_path = _write_batch_input_file(request, model)

    async def _log_create_error(error: str) -> None:
        await log_batch_usage(
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider.value,
            endpoint="/v1/batches",
            user_id=user_id,
            error=error,
        )

    try:
        batch = await _run_batch_operation(
            action="create",
            provider=provider.value,
            operation=acreate_batch,
            call_kwargs={
                "provider": provider,
                "input_file_path": tmp_path,
                "endpoint": "/v1/chat/completions",
                "completion_window": request.completion_window,
                "metadata": request.metadata,
                **provider_kwargs,
            },
            on_error=_log_create_error,
        )
    finally:
        _remove_batch_input_file(tmp_path)

    await log_batch_usage(
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=model,
        provider=provider.value,
        endpoint="/v1/batches",
        user_id=user_id,
    )

    return _batch_response(batch, provider.value)


@router.get("/{batch_id}", response_model=None)
async def retrieve_batch(
    batch_id: str,
    provider: str,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> dict[str, Any]:
    """Retrieve the status of a batch."""
    provider_enum, provider_kwargs = _batch_provider_context(config, provider)
    batch = await _run_batch_operation(
        action="retrieve",
        provider=provider,
        operation=aretrieve_batch,
        call_kwargs={"provider": provider_enum, "batch_id": batch_id, **provider_kwargs},
    )
    return _batch_response(batch, provider)


@router.post("/{batch_id}/cancel", response_model=None)
async def cancel_batch(
    batch_id: str,
    provider: str,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> dict[str, Any]:
    """Cancel a batch."""
    provider_enum, provider_kwargs = _batch_provider_context(config, provider)
    batch = await _run_batch_operation(
        action="cancel",
        provider=provider,
        operation=acancel_batch,
        call_kwargs={"provider": provider_enum, "batch_id": batch_id, **provider_kwargs},
    )
    return _batch_response(batch, provider)


@router.get("", response_model=None)
async def list_batches(
    provider: str,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    after: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """List batches for a provider."""
    provider_enum, provider_kwargs = _batch_provider_context(config, provider)

    list_kwargs: dict[str, Any] = {"provider": provider_enum, **provider_kwargs}
    if after is not None:
        list_kwargs["after"] = after
    if limit is not None:
        list_kwargs["limit"] = limit

    batches = await _run_batch_operation(
        action="list",
        provider=provider,
        operation=alist_batches,
        call_kwargs=list_kwargs,
    )

    return {"data": [{**batch.model_dump(), "provider": provider} for batch in batches]}


@router.get(
    "/{batch_id}/results",
    response_model=None,
    responses={
        status.HTTP_409_CONFLICT: {"description": "Batch is not yet complete"},
        status.HTTP_502_BAD_GATEWAY: {"description": "LLM provider error"},
    },
)
async def retrieve_batch_results(
    batch_id: str,
    provider: str,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """Retrieve the results of a completed batch."""
    api_key = auth_result[0]
    api_key_id = api_key.id if api_key else None
    user_id = api_key.user_id if api_key else None

    provider_enum, provider_kwargs = _batch_provider_context(config, provider)

    try:
        result = await _run_batch_operation(
            action="results retrieve",
            provider=provider,
            operation=aretrieve_batch_results,
            call_kwargs={"provider": provider_enum, "batch_id": batch_id, **provider_kwargs},
        )
    except BatchNotCompleteError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Batch '{batch_id}' is not yet complete (status: {e.batch_status}). "
                f"Call GET /v1/batches/{batch_id}?provider={provider} to check the current status."
            ),
        ) from e

    # Extract model from the first successful result if available
    batch_model = "batch"
    for item in result.results:
        if item.result is not None:
            batch_model = item.result.model
            break

    await log_batch_usage(
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=batch_model,
        provider=provider,
        endpoint="/v1/batches/results",
        user_id=user_id,
    )

    return {
        "results": [
            {
                "custom_id": item.custom_id,
                "result": item.result.model_dump() if item.result else None,
                "error": {"code": item.error.code, "message": item.error.message} if item.error else None,
            }
            for item in result.results
        ]
    }

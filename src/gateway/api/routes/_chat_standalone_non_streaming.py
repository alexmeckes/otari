import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from fastapi import HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._chat_non_streaming_completion import run_non_streaming_completion
from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._chat_tools import ChatToolSelection
from gateway.api.routes._usage import log_usage, rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import RateLimitInfo
from gateway.services.chat_tool_config import strip_gateway_fields
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_loop import MaxToolIterationsExceeded
from gateway.services.provider_kwargs import get_provider_kwargs
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError


async def run_standalone_non_streaming_chat(
    *,
    request: ChatCompletionRequest,
    response: Response,
    config: GatewayConfig,
    db: AsyncSession | None,
    log_writer: LogWriter,
    api_key_id: str | None,
    user_id: str | None,
    rate_limit_info: RateLimitInfo | None,
    tool_selection: ChatToolSelection,
    max_tool_iterations: int,
    completion_fn: Callable[..., Awaitable[ChatCompletion | AsyncIterator[ChatCompletionChunk]]],
    mcp_client_pool_factory: Callable[[list[McpServerConfig]], Any],
) -> ChatCompletion:
    """Execute a standalone non-streaming chat request against one provider."""
    provider, model = AnyLLM.split_model_provider(request.model)
    provider_kwargs = get_provider_kwargs(config, provider)
    request_fields = strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=tool_selection.tools_extracted,
        remaining_user_tools=tool_selection.remaining_user_tools,
    )
    completion_kwargs = {**provider_kwargs, **request_fields}

    try:
        completion = await run_non_streaming_completion(
            completion_kwargs=completion_kwargs,
            completion_fn=completion_fn,
            mcp_client_pool_factory=mcp_client_pool_factory,
            mcp_server_configs=request.mcp_servers,
            max_tool_iterations=max_tool_iterations,
            tools_header=request.tools_header,
            use_sandbox=tool_selection.use_sandbox,
            sandbox_url=tool_selection.sandbox_url,
            sandbox_tool_entry=tool_selection.sandbox_tool_entry,
            use_web_search=tool_selection.use_web_search,
            web_search_url=tool_selection.web_search_url,
            web_search_tool_entry=tool_selection.web_search_tool_entry,
        )
        if db is not None:
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/chat/completions",
                user_id=user_id,
                project_id=request.project_id,
                tags=request.tags,
                response=completion,
            )
    except HTTPException:
        raise
    except SandboxNotReachableError as exc:
        # Sandbox is gateway-side infra, not an LLM provider. Clearer detail
        # so operators don't chase a provider outage that's really the
        # sandbox container being down.
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
        ) from exc
    except WebSearchNotReachableError as exc:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
        ) from exc
    except MaxToolIterationsExceeded as exc:
        # Gateway-owned cap, not an upstream provider failure. 422 lets
        # callers distinguish a runaway tool loop from a real outage.
        logger.warning("Tool loop iteration cap hit (standalone): cap=%d", max_tool_iterations)
        if db is not None:
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/chat/completions",
                user_id=user_id,
                project_id=request.project_id,
                tags=request.tags,
                error=str(exc),
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        if db is not None:
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/chat/completions",
                user_id=user_id,
                project_id=request.project_id,
                tags=request.tags,
                error=str(exc),
            )

        logger.error("Provider call failed for %s:%s: %s", provider, model, exc)
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM provider timeout",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from exc

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return completion

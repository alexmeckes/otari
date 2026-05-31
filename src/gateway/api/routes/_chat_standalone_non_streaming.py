from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from fastapi import HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._chat_non_streaming_completion import run_non_streaming_completion
from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._chat_standalone_errors import standalone_provider_failure_exception
from gateway.api.routes._chat_tool_backend_errors import (
    chat_tool_backend_failure_exception,
    chat_tool_iteration_cap_exception,
)
from gateway.api.routes._chat_tools import ChatToolSelection
from gateway.api.routes._usage import apply_rate_limit_headers, log_usage, provider_model_label
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

_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"


async def _log_standalone_chat_usage(
    *,
    db: AsyncSession | None,
    log_writer: LogWriter,
    api_key_id: str | None,
    model: str,
    provider: Any,
    user_id: str | None,
    request: ChatCompletionRequest,
    response: ChatCompletion | None = None,
    error: str | None = None,
) -> None:
    if db is None:
        return
    await log_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=model,
        provider=provider,
        endpoint=_CHAT_COMPLETIONS_ENDPOINT,
        user_id=user_id,
        project_id=request.project_id,
        tags=request.tags,
        response=response,
        error=error,
    )


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
    provider_label = provider_model_label(provider, model)
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
        await _log_standalone_chat_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            user_id=user_id,
            request=request,
            response=completion,
        )
    except HTTPException:
        raise
    except SandboxNotReachableError as exc:
        # Sandbox is gateway-side infra, not an LLM provider. Clearer detail
        # so operators don't chase a provider outage that's really the
        # sandbox container being down.
        logger.error("Sandbox unreachable for %s: %s", provider_label, exc)
        raise chat_tool_backend_failure_exception(exc) from exc
    except WebSearchNotReachableError as exc:
        logger.error("Web search backend unreachable for %s: %s", provider_label, exc)
        raise chat_tool_backend_failure_exception(exc) from exc
    except MaxToolIterationsExceeded as exc:
        # Gateway-owned cap, not an upstream provider failure. 422 lets
        # callers distinguish a runaway tool loop from a real outage.
        logger.warning("Tool loop iteration cap hit (standalone): cap=%d", max_tool_iterations)
        await _log_standalone_chat_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            user_id=user_id,
            request=request,
            error=str(exc),
        )
        raise chat_tool_iteration_cap_exception(exc) from exc
    except Exception as exc:
        await _log_standalone_chat_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            user_id=user_id,
            request=request,
            error=str(exc),
        )

        logger.error("Provider call failed for %s: %s", provider_label, exc)
        raise standalone_provider_failure_exception(exc) from exc

    apply_rate_limit_headers(response, rate_limit_info)

    return completion

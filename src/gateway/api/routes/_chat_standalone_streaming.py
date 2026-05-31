from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._chat_standalone_errors import standalone_provider_failure_exception
from gateway.api.routes._chat_streaming_response import build_chat_streaming_response
from gateway.api.routes._chat_tool_backend_errors import chat_tool_backend_failure_exception
from gateway.api.routes._chat_tool_iterations import resolve_max_tool_iterations
from gateway.api.routes._chat_tools import ChatToolSelection
from gateway.api.routes._usage import log_usage
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import RateLimitInfo
from gateway.services.chat_tool_config import (
    build_web_search_backend,
    resolve_sandbox_purpose_hint,
    strip_gateway_fields,
)
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_loop import (
    inject_purpose_hints,
    mcp_tool_loop_stream,
)
from gateway.services.provider_kwargs import get_provider_kwargs
from gateway.services.sandbox_backend import SandboxBackend, SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError


async def run_standalone_streaming_chat(
    *,
    request: ChatCompletionRequest,
    config: GatewayConfig,
    db: AsyncSession | None,
    log_writer: LogWriter,
    api_key_id: str | None,
    user_id: str | None,
    rate_limit_info: RateLimitInfo | None,
    tool_selection: ChatToolSelection,
    completion_fn: Callable[..., Awaitable[ChatCompletion | AsyncIterator[ChatCompletionChunk]]],
    mcp_client_pool_factory: Callable[[list[McpServerConfig]], Any],
) -> StreamingResponse:
    """Create a standalone streaming chat response for one provider attempt."""
    provider, model = AnyLLM.split_model_provider(request.model)
    provider_kwargs = get_provider_kwargs(config, provider)
    max_tool_iterations = resolve_max_tool_iterations(request.max_tool_iterations)

    request_fields = strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=tool_selection.tools_extracted,
        remaining_user_tools=tool_selection.remaining_user_tools,
    )
    completion_kwargs = {**provider_kwargs, **request_fields}
    if completion_kwargs.get("stream_options") is None:
        completion_kwargs["stream_options"] = {"include_usage": True}

    try:
        stream = await _open_standalone_stream(
            request=request,
            completion_kwargs=completion_kwargs,
            tool_selection=tool_selection,
            max_tool_iterations=max_tool_iterations,
            completion_fn=completion_fn,
            mcp_client_pool_factory=mcp_client_pool_factory,
        )
    except HTTPException:
        raise
    except SandboxNotReachableError as exc:
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, exc)
        raise chat_tool_backend_failure_exception(exc) from exc
    except WebSearchNotReachableError as exc:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, exc)
        raise chat_tool_backend_failure_exception(exc) from exc
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
        logger.error("Stream creation failed for %s:%s: %s", provider, model, exc)
        raise standalone_provider_failure_exception(exc) from exc

    return build_chat_streaming_response(
        stream=stream,
        provider=provider,
        model=model,
        platform_mode=False,
        correlation_id=None,
        request_id=None,
        config=config,
        db=db,
        log_writer=log_writer,
        api_key_id=api_key_id,
        user_id=user_id,
        project_id=request.project_id,
        tags=request.tags,
        rate_limit_info=rate_limit_info,
    )


async def _open_standalone_stream(
    *,
    request: ChatCompletionRequest,
    completion_kwargs: dict[str, Any],
    tool_selection: ChatToolSelection,
    max_tool_iterations: int,
    completion_fn: Callable[..., Awaitable[ChatCompletion | AsyncIterator[ChatCompletionChunk]]],
    mcp_client_pool_factory: Callable[[list[McpServerConfig]], Any],
) -> AsyncIterator[ChatCompletionChunk]:
    mcp_server_configs = request.mcp_servers
    if mcp_server_configs:
        pool_configs = mcp_server_configs

        async def _mcp_stream() -> AsyncIterator[ChatCompletionChunk]:
            async with mcp_client_pool_factory(pool_configs) as pool:
                kwargs = {
                    **completion_kwargs,
                    "messages": inject_purpose_hints(
                        completion_kwargs["messages"],
                        pool.purpose_hints(),
                        header=request.tools_header,
                    ),
                }
                async for chunk in mcp_tool_loop_stream(
                    completion_kwargs=kwargs,
                    pool=pool,
                    max_iterations=max_tool_iterations,
                ):
                    yield chunk

        return _mcp_stream()

    if tool_selection.use_sandbox:
        assert tool_selection.sandbox_url is not None
        sandbox_hint = resolve_sandbox_purpose_hint(tool_selection.sandbox_tool_entry)
        sandbox_backend = SandboxBackend(sandbox_url=tool_selection.sandbox_url, purpose_hint=sandbox_hint)
        await sandbox_backend.__aenter__()

        async def _sandbox_stream() -> AsyncIterator[ChatCompletionChunk]:
            try:
                kwargs = {
                    **completion_kwargs,
                    "messages": inject_purpose_hints(
                        completion_kwargs["messages"],
                        sandbox_backend.purpose_hints(),
                        header=request.tools_header,
                    ),
                }
                async for chunk in mcp_tool_loop_stream(
                    completion_kwargs=kwargs,
                    pool=sandbox_backend,  # type: ignore[arg-type]
                    max_iterations=max_tool_iterations,
                ):
                    yield chunk
            finally:
                await sandbox_backend.__aexit__(None, None, None)

        return _sandbox_stream()

    if tool_selection.use_web_search:
        assert tool_selection.web_search_url is not None
        assert tool_selection.web_search_tool_entry is not None
        web_search_backend = build_web_search_backend(
            base_url=tool_selection.web_search_url,
            tool_entry=tool_selection.web_search_tool_entry,
        )
        await web_search_backend.__aenter__()

        async def _web_search_stream() -> AsyncIterator[ChatCompletionChunk]:
            try:
                kwargs = {
                    **completion_kwargs,
                    "messages": inject_purpose_hints(
                        completion_kwargs["messages"],
                        web_search_backend.purpose_hints(),
                        header=request.tools_header,
                    ),
                }
                async for chunk in mcp_tool_loop_stream(
                    completion_kwargs=kwargs,
                    pool=web_search_backend,  # type: ignore[arg-type]
                    max_iterations=max_tool_iterations,
                ):
                    yield chunk
            finally:
                await web_search_backend.__aexit__(None, None, None)

        return _web_search_stream()

    stream = await completion_fn(**completion_kwargs)
    return stream  # type: ignore[return-value]

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Any

from any_llm import LLMProvider, acompletion
from any_llm.types.completion import ChatCompletionChunk
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._chat_request_fields import chat_provider_call_kwargs, chat_provider_request_fields
from gateway.api.routes._chat_stream_options import ensure_stream_usage_options
from gateway.api.routes._chat_streaming_response import build_chat_streaming_response
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import RateLimitInfo
from gateway.services.chat_tool_config import (
    build_web_search_backend,
    resolve_sandbox_purpose_hint,
)
from gateway.services.mcp_client import MCPClientPool
from gateway.services.mcp_loop import DEFAULT_MAX_TOOL_ITERATIONS, mcp_tool_loop_stream, tool_loop_completion_kwargs
from gateway.services.platform_gateway import (
    ResolvedAttempt,
    ResolvedRoute,
    classify_upstream_error,
    report_platform_usage,
)
from gateway.services.sandbox_backend import SandboxBackend
from gateway.streaming import StreamingAttemptFailure, iterate_streaming_attempts

_DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS = 2000
_DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP = 30000
_STREAM_FIRST_CHUNK_TIMEOUT_MS_KEY = "streaming_first_chunk_timeout_ms"
_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP_KEY = "streaming_first_chunk_timeout_ms_tool_loop"


def _first_chunk_timeout_seconds(config: GatewayConfig, *, tool_mode: bool) -> float:
    if tool_mode:
        timeout_ms = config.platform.get(
            _STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP_KEY,
            _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP,
        )
    else:
        timeout_ms = config.platform.get(
            _STREAM_FIRST_CHUNK_TIMEOUT_MS_KEY,
            _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS,
        )
    return int(timeout_ms) / 1000


async def run_streaming_with_fallback(
    *,
    route: ResolvedRoute,
    request: ChatCompletionRequest,
    config: GatewayConfig,
    background_tasks: BackgroundTasks,
    rate_limit_info: RateLimitInfo | None,
    mcp_server_configs: list[McpServerConfig] | None = None,
    use_sandbox: bool = False,
    sandbox_url: str | None = None,
    sandbox_tool_entry: dict[str, Any] | None = None,
    use_web_search: bool = False,
    web_search_url: str | None = None,
    web_search_tool_entry: dict[str, Any] | None = None,
    remaining_user_tools: list[dict[str, Any]] | None = None,
    tools_extracted: bool = False,
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
) -> StreamingResponse:
    """Run a platform streaming request with pre-first-chunk attempt fallback."""
    tool_mode = bool(mcp_server_configs) or use_sandbox or use_web_search
    first_chunk_timeout_seconds = _first_chunk_timeout_seconds(config, tool_mode=tool_mode)

    base_request_fields = chat_provider_request_fields(
        request,
        tools_extracted=tools_extracted,
        remaining_user_tools=remaining_user_tools,
    )

    backend_stack = AsyncExitStack()
    pool_for_loop: Any = None
    try:
        if mcp_server_configs:
            pool_for_loop = await backend_stack.enter_async_context(MCPClientPool(mcp_server_configs))
        elif use_sandbox:
            assert sandbox_url is not None
            sandbox_hint = resolve_sandbox_purpose_hint(sandbox_tool_entry)
            pool_for_loop = await backend_stack.enter_async_context(
                SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint),
            )
        elif use_web_search:
            assert web_search_url is not None
            assert web_search_tool_entry is not None
            pool_for_loop = await backend_stack.enter_async_context(
                build_web_search_backend(base_url=web_search_url, tool_entry=web_search_tool_entry),
            )
    except BaseException:
        await backend_stack.aclose()
        raise

    async def _build_for_attempt(
        attempt: ResolvedAttempt,
    ) -> AsyncIterator[ChatCompletionChunk]:
        provider_kwargs: dict[str, Any] = {"api_key": attempt.api_key}
        if attempt.api_base:
            provider_kwargs["api_base"] = attempt.api_base
        completion_kwargs = chat_provider_call_kwargs(
            provider_kwargs,
            base_request_fields,
            model=attempt.model_selector,
        )
        ensure_stream_usage_options(completion_kwargs)
        if pool_for_loop is None:
            return await acompletion(**completion_kwargs)  # type: ignore[return-value]
        return mcp_tool_loop_stream(
            completion_kwargs=tool_loop_completion_kwargs(
                completion_kwargs,
                pool_for_loop,
                header=request.tools_header,
            ),
            pool=pool_for_loop,
            max_iterations=max_tool_iterations,
        )

    async def _on_attempt_failed(attempt: ResolvedAttempt, failure: StreamingAttemptFailure) -> None:
        background_tasks.add_task(
            report_platform_usage,
            config,
            attempt.attempt_id,
            "error",
            None,
            failure.error_class,
        )
        logger.warning(
            "Streaming attempt failed request_id=%s position=%d provider=%s model=%s error=%s",
            route.request_id,
            attempt.position,
            attempt.provider,
            attempt.model,
            failure.error_class,
        )

    try:
        chosen, stream = await iterate_streaming_attempts(
            attempts=route.attempts,
            build_stream=_build_for_attempt,
            classify_error=classify_upstream_error,
            on_attempt_failed=_on_attempt_failed,
            first_chunk_timeout_seconds=first_chunk_timeout_seconds,
        )
    except BaseException:
        await backend_stack.aclose()
        raise

    if tool_mode:
        logger.info(
            "Tool-loop streaming lock-in request_id=%s position=%d provider=%s model=%s",
            route.request_id,
            chosen.position,
            chosen.provider,
            chosen.model,
        )

    if pool_for_loop is not None:

        async def _stream_with_backend_cleanup() -> AsyncIterator[ChatCompletionChunk]:
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                await backend_stack.aclose()

        stream_to_return: AsyncIterator[ChatCompletionChunk] = _stream_with_backend_cleanup()
    else:
        stream_to_return = stream

    return build_chat_streaming_response(
        stream=stream_to_return,
        provider=LLMProvider(chosen.provider),
        model=chosen.model,
        platform_mode=True,
        correlation_id=chosen.attempt_id,
        request_id=route.request_id,
        config=config,
        db=None,
        log_writer=None,
        api_key_id=None,
        user_id=None,
        project_id=None,
        tags=None,
        rate_limit_info=rate_limit_info,
    )

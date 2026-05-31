from typing import Annotated

from any_llm import acompletion
from any_llm.types.completion import (
    ChatCompletion,
)
from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed, get_log_writer
from gateway.api.routes._chat_context import resolve_chat_mcp_server_ids, resolve_chat_request_context
from gateway.api.routes._chat_platform_non_streaming import run_platform_non_streaming_chat
from gateway.api.routes._chat_platform_streaming import run_platform_streaming_chat
from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._chat_routing import resolve_standalone_chat_routing_plan, run_standalone_routing_plan
from gateway.api.routes._chat_standalone_non_streaming import run_standalone_non_streaming_chat
from gateway.api.routes._chat_standalone_streaming import run_standalone_streaming_chat
from gateway.api.routes._chat_tools import resolve_chat_tool_selection
from gateway.core.config import GatewayConfig
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_client import MCPClientPool
from gateway.services.mcp_loop import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("/completions", response_model=None)
async def chat_completions(
    raw_request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    request: ChatCompletionRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> ChatCompletion | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming responses.
    Handles reasoning content from otari providers.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """
    platform_mode = config.is_platform_mode

    context = await resolve_chat_request_context(
        raw_request=raw_request,
        response=response,
        request=request,
        db=db,
        config=config,
    )
    api_key_id = context.api_key_id
    user_id = context.user_id
    rate_limit_info = context.rate_limit_info
    route = context.route

    await resolve_chat_mcp_server_ids(
        request=request,
        config=config,
        platform_mode=platform_mode,
        user_token=context.user_token,
    )

    tool_selection = resolve_chat_tool_selection(
        tools=request.tools,
        mcp_servers=request.mcp_servers,
    )
    sandbox_tool_entry = tool_selection.sandbox_tool_entry
    sandbox_url = tool_selection.sandbox_url
    use_sandbox = tool_selection.use_sandbox
    web_search_tool_entry = tool_selection.web_search_tool_entry
    web_search_url = tool_selection.web_search_url
    use_web_search = tool_selection.use_web_search
    remaining_user_tools = tool_selection.remaining_user_tools

    routing_plan = await resolve_standalone_chat_routing_plan(
        request=request,
        db=db,
        platform_mode=platform_mode,
    )

    # ------------------------------------------------------------------
    # Streaming path: iterate `route.attempts` before any bytes are flushed,
    # then commit to the first attempt that yields a chunk. Implemented in
    # `run_streaming_with_fallback` via `iterate_streaming_attempts`.
    #
    # Mid-stream failover (after first chunk) is out of scope: recovering
    # would require either silently buffering the prefix (delays first byte)
    # or a client-aware "restart" event (breaks OpenAI-SDK compatibility).
    # Errors after first chunk propagate to the client.
    # ------------------------------------------------------------------
    if request.stream:
        # Platform-mode streaming — tool modes (sandbox / web_search / MCP)
        # also flow through here so they get per-attempt fallback up to the
        # lock-in point (first chunk = first assistant message).
        if platform_mode:
            return await run_platform_streaming_chat(
                request=request,
                background_tasks=background_tasks,
                route=route,
                config=config,
                rate_limit_info=rate_limit_info,
                tool_selection=tool_selection,
            )

        return await run_standalone_streaming_chat(
            request=request,
            config=config,
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            user_id=user_id,
            rate_limit_info=rate_limit_info,
            tool_selection=tool_selection,
            completion_fn=acompletion,
            mcp_client_pool_factory=MCPClientPool,
        )

    # ------------------------------------------------------------------
    # Non-streaming path. Iterates `route.attempts` with pre-lock-in
    # fallback semantics: if a tool-loop attempt fails *before* the model
    # has returned its first assistant message, we fall through to the
    # next attempt. Once locked in (first assistant message received),
    # subsequent failures terminate the request — we never swap providers
    # between tool-use rounds.
    # ------------------------------------------------------------------
    mcp_server_configs = request.mcp_servers
    max_tool_iterations = min(
        request.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
        MAX_TOOL_ITERATIONS_CAP,
    )

    if platform_mode:
        return await run_platform_non_streaming_chat(
            request=request,
            response=response,
            background_tasks=background_tasks,
            route=route,
            config=config,
            rate_limit_info=rate_limit_info,
            tool_selection=tool_selection,
            max_tool_iterations=max_tool_iterations,
            completion_fn=acompletion,
            mcp_client_pool_factory=MCPClientPool,
        )

    if routing_plan is not None:
        assert db is not None
        return await run_standalone_routing_plan(
            plan=routing_plan,
            request=request,
            response=response,
            config=config,
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            user_id=user_id,
            rate_limit_info=rate_limit_info,
            mcp_server_configs=mcp_server_configs,
            max_tool_iterations=max_tool_iterations,
            sandbox_tool_entry=sandbox_tool_entry,
            sandbox_url=sandbox_url,
            use_sandbox=use_sandbox,
            web_search_tool_entry=web_search_tool_entry,
            web_search_url=web_search_url,
            use_web_search=use_web_search,
            remaining_user_tools=remaining_user_tools,
            trace_endpoint=request.route_trace_endpoint,
            completion_fn=acompletion,
            mcp_client_pool_factory=MCPClientPool,
        )

    return await run_standalone_non_streaming_chat(
        request=request,
        response=response,
        config=config,
        db=db,
        log_writer=log_writer,
        api_key_id=api_key_id,
        user_id=user_id,
        rate_limit_info=rate_limit_info,
        tool_selection=tool_selection,
        max_tool_iterations=max_tool_iterations,
        completion_fn=acompletion,
        mcp_client_pool_factory=MCPClientPool,
    )

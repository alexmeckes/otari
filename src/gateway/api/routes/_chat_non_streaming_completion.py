from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk

from gateway.models.mcp import McpServerConfig
from gateway.services.chat_tool_config import build_web_search_backend, resolve_sandbox_purpose_hint
from gateway.services.mcp_loop import mcp_tool_loop, tool_loop_completion_kwargs
from gateway.services.sandbox_backend import SandboxBackend


async def run_non_streaming_completion(
    *,
    completion_kwargs: dict[str, Any],
    completion_fn: Callable[..., Awaitable[ChatCompletion | AsyncIterator[ChatCompletionChunk]]],
    mcp_client_pool_factory: Callable[[list[McpServerConfig]], Any],
    mcp_server_configs: list[McpServerConfig] | None,
    max_tool_iterations: int,
    tools_header: str | None,
    use_sandbox: bool,
    sandbox_url: str | None,
    sandbox_tool_entry: dict[str, Any] | None,
    use_web_search: bool,
    web_search_url: str | None,
    web_search_tool_entry: dict[str, Any] | None,
    on_first_response: Callable[[], None] | None = None,
) -> ChatCompletion:
    """Run one non-streaming completion attempt with gateway tool backends."""
    if mcp_server_configs:
        async with mcp_client_pool_factory(mcp_server_configs) as pool:
            return await mcp_tool_loop(
                completion_kwargs=tool_loop_completion_kwargs(completion_kwargs, pool, header=tools_header),
                pool=pool,
                max_iterations=max_tool_iterations,
                on_first_response=on_first_response,
            )
    if use_sandbox:
        assert sandbox_url is not None
        sandbox_hint = resolve_sandbox_purpose_hint(sandbox_tool_entry)
        async with SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint) as backend:
            return await mcp_tool_loop(
                completion_kwargs=tool_loop_completion_kwargs(completion_kwargs, backend, header=tools_header),
                pool=backend,  # type: ignore[arg-type]
                max_iterations=max_tool_iterations,
                on_first_response=on_first_response,
            )
    if use_web_search:
        assert web_search_url is not None
        assert web_search_tool_entry is not None
        async with build_web_search_backend(
            base_url=web_search_url,
            tool_entry=web_search_tool_entry,
        ) as web_backend:
            return await mcp_tool_loop(
                completion_kwargs=tool_loop_completion_kwargs(completion_kwargs, web_backend, header=tools_header),
                pool=web_backend,  # type: ignore[arg-type]
                max_iterations=max_tool_iterations,
                on_first_response=on_first_response,
            )
    completion = await completion_fn(**completion_kwargs)
    return cast(ChatCompletion, completion)

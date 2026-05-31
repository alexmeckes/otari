from fastapi import BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse

from gateway.api.routes._chat_platform_errors import platform_attempt_failure_exception
from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._chat_streaming_fallback import run_streaming_with_fallback
from gateway.api.routes._chat_tools import ChatToolSelection
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.rate_limit import RateLimitInfo
from gateway.services.mcp_loop import DEFAULT_MAX_TOOL_ITERATIONS, MAX_TOOL_ITERATIONS_CAP
from gateway.services.platform_gateway import ResolvedRoute
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError


async def run_platform_streaming_chat(
    *,
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    route: ResolvedRoute | None,
    config: GatewayConfig,
    rate_limit_info: RateLimitInfo | None,
    tool_selection: ChatToolSelection,
) -> StreamingResponse:
    """Execute platform-mode streaming with pre-first-chunk fallback."""
    if route is None or not route.attempts:
        if route is not None:
            logger.error(
                "Platform returned empty attempts list request_id=%s",
                route.request_id,
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service returned no resolvable provider",
        )

    max_tool_iterations = min(
        request.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
        MAX_TOOL_ITERATIONS_CAP,
    )
    try:
        return await run_streaming_with_fallback(
            route=route,
            request=request,
            config=config,
            background_tasks=background_tasks,
            rate_limit_info=rate_limit_info,
            mcp_server_configs=request.mcp_servers,
            use_sandbox=tool_selection.use_sandbox,
            sandbox_url=tool_selection.sandbox_url,
            sandbox_tool_entry=tool_selection.sandbox_tool_entry,
            use_web_search=tool_selection.use_web_search,
            web_search_url=tool_selection.web_search_url,
            web_search_tool_entry=tool_selection.web_search_tool_entry,
            remaining_user_tools=tool_selection.remaining_user_tools,
            tools_extracted=tool_selection.tools_extracted,
            max_tool_iterations=max_tool_iterations,
        )
    except HTTPException:
        raise
    except SandboxNotReachableError as exc:
        logger.error("Sandbox unreachable request_id=%s: %s", route.request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
        ) from exc
    except WebSearchNotReachableError as exc:
        logger.error("Web search backend unreachable request_id=%s: %s", route.request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
        ) from exc
    except Exception as exc:
        # Every attempt failed before any bytes were flushed.
        logger.error(
            "All streaming attempts failed request_id=%s: %s",
            route.request_id,
            exc,
        )
        raise platform_attempt_failure_exception(exc, attempts_count=len(route.attempts)) from exc

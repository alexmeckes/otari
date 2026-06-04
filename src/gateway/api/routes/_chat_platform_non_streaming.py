from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, NamedTuple

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from fastapi import BackgroundTasks, HTTPException, Response, status

from gateway.api.routes._chat_non_streaming_completion import run_non_streaming_completion
from gateway.api.routes._chat_platform_errors import platform_attempt_failure_exception
from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._chat_request_fields import chat_provider_call_kwargs, chat_provider_request_fields
from gateway.api.routes._chat_tool_backend_errors import chat_tool_iteration_cap_exception
from gateway.api.routes._chat_tools import ChatToolSelection
from gateway.api.routes._usage import apply_rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import RateLimitInfo
from gateway.services.mcp_loop import MaxToolIterationsExceeded
from gateway.services.platform_gateway import (
    ResolvedRoute,
    classify_upstream_error,
    report_platform_usage,
)


class _AttemptFailure(NamedTuple):
    position: int
    provider: str
    model: str
    error_class: str


async def run_platform_non_streaming_chat(
    *,
    request: ChatCompletionRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    route: ResolvedRoute | None,
    config: GatewayConfig,
    rate_limit_info: RateLimitInfo | None,
    tool_selection: ChatToolSelection,
    max_tool_iterations: int,
    completion_fn: Callable[..., Awaitable[ChatCompletion | AsyncIterator[ChatCompletionChunk]]],
    mcp_client_pool_factory: Callable[[list[McpServerConfig]], Any],
) -> ChatCompletion:
    """Execute platform-mode non-streaming attempts with pre-lock-in fallback."""
    if route is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error: missing route context",
        )
    if not route.attempts:
        logger.error(
            "Platform returned empty attempts list request_id=%s",
            route.request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service returned no resolvable provider",
        )

    failures: list[_AttemptFailure] = []
    last_exc: BaseException | None = None
    base_request_fields = chat_provider_request_fields(
        request,
        tools_extracted=tool_selection.tools_extracted,
        remaining_user_tools=tool_selection.remaining_user_tools,
    )
    mcp_server_configs = request.mcp_servers

    for attempt in route.attempts:
        completion_kwargs = chat_provider_call_kwargs(
            attempt.provider_kwargs,
            base_request_fields,
            model=attempt.model_selector,
        )

        locked_in = False

        def _mark_locked_in(_pos: int = attempt.position) -> None:
            nonlocal locked_in
            locked_in = True
            logger.info(
                "Tool-loop lock-in request_id=%s position=%d provider=%s model=%s",
                route.request_id,
                _pos,
                attempt.provider,
                attempt.model,
            )

        try:
            completion = await run_non_streaming_completion(
                completion_kwargs=completion_kwargs,
                completion_fn=completion_fn,
                mcp_client_pool_factory=mcp_client_pool_factory,
                mcp_server_configs=mcp_server_configs,
                max_tool_iterations=max_tool_iterations,
                tools_header=request.tools_header,
                use_sandbox=tool_selection.use_sandbox,
                sandbox_url=tool_selection.sandbox_url,
                sandbox_tool_entry=tool_selection.sandbox_tool_entry,
                use_web_search=tool_selection.use_web_search,
                web_search_url=tool_selection.web_search_url,
                web_search_tool_entry=tool_selection.web_search_tool_entry,
                on_first_response=_mark_locked_in,
            )
        except HTTPException:
            raise
        except MaxToolIterationsExceeded as exc:
            logger.warning(
                "Tool loop iteration cap hit request_id=%s position=%d cap=%d",
                route.request_id,
                attempt.position,
                max_tool_iterations,
            )
            raise chat_tool_iteration_cap_exception(exc) from exc
        except BaseException as exc:
            retryable, error_class = classify_upstream_error(exc)
            background_tasks.add_task(
                report_platform_usage,
                config,
                attempt.attempt_id,
                "error",
                None,
                error_class,
            )
            logger.warning(
                "Provider call failed request_id=%s position=%d provider=%s model=%s "
                "error=%s retryable=%s locked_in=%s",
                route.request_id,
                attempt.position,
                attempt.provider,
                attempt.model,
                error_class,
                retryable,
                locked_in,
            )
            last_exc = exc
            if locked_in:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="LLM provider error",
                ) from exc
            if not retryable:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="LLM provider error",
                ) from exc
            failures.append(_AttemptFailure(attempt.position, attempt.provider, attempt.model, error_class))
            continue

        background_tasks.add_task(
            report_platform_usage,
            config,
            attempt.attempt_id,
            "success",
            completion.usage,
            None,
        )
        response.headers["X-Correlation-ID"] = attempt.attempt_id
        apply_rate_limit_headers(response, rate_limit_info)
        return completion

    logger.error(
        "All upstream attempts failed request_id=%s failures=%s",
        route.request_id,
        failures,
    )
    raise platform_attempt_failure_exception(last_exc, attempts_count=len(route.attempts)) from last_exc

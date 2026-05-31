import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
from any_llm import LLMProvider
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
from gateway.services.platform_gateway import classify_upstream_error
from gateway.services.provider_kwargs import get_provider_kwargs
from gateway.services.routing_policy_service import (
    DEFAULT_ROUTING_MODEL,
    RoutingCandidate,
    RoutingPlan,
    RoutingPolicyError,
    apply_context_policy,
    apply_guardrail_redactions,
    record_route_trace,
    resolve_routing_plan,
)
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError


async def resolve_standalone_chat_routing_plan(
    *,
    request: ChatCompletionRequest,
    db: AsyncSession | None,
    platform_mode: bool,
) -> RoutingPlan | None:
    """Resolve a standalone default_routing chat request into an execution plan."""
    if platform_mode or request.model != DEFAULT_ROUTING_MODEL:
        return None
    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Routing policies do not support streaming chat completions yet",
        )
    assert db is not None
    try:
        return await resolve_routing_plan(
            db,
            request_body=request.model_dump(exclude_unset=True),
            project_id=request.project_id,
            tags=request.tags,
        )
    except RoutingPolicyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


async def run_standalone_routing_plan(
    *,
    plan: RoutingPlan,
    request: ChatCompletionRequest,
    response: Response,
    config: GatewayConfig,
    db: AsyncSession,
    log_writer: LogWriter,
    api_key_id: str | None,
    user_id: str | None,
    rate_limit_info: RateLimitInfo | None,
    tool_selection: ChatToolSelection,
    max_tool_iterations: int,
    trace_endpoint: str,
    completion_fn: Callable[..., Awaitable[ChatCompletion | AsyncIterator[ChatCompletionChunk]]],
    mcp_client_pool_factory: Callable[[list[McpServerConfig]], Any],
) -> ChatCompletion:
    """Execute a standalone routing plan with pre-response fallback."""
    request_fields = strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=tool_selection.tools_extracted,
        remaining_user_tools=tool_selection.remaining_user_tools,
    )
    request_fields, _ = apply_guardrail_redactions(plan.policy.config_ or {}, request_fields)
    request_fields, _ = apply_context_policy(plan.policy.config_ or {}, request_fields)

    attempts: list[dict[str, Any]] = []
    last_exc: BaseException | None = None

    for candidate in plan.candidates:
        provider_kwargs = get_provider_kwargs(config, LLMProvider(candidate.provider))
        completion_kwargs = {**provider_kwargs, **request_fields, "model": candidate.model}
        started_at = time.perf_counter()
        attempt_record: dict[str, Any] = {
            "position": candidate.position,
            "provider": candidate.provider,
            "model": candidate.provider_model,
            "model_key": candidate.model,
            "tier": candidate.tier,
        }

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
        except HTTPException:
            raise
        except SandboxNotReachableError as exc:
            await _record_routing_attempt_error(
                db=db,
                log_writer=log_writer,
                plan=plan,
                api_key_id=api_key_id,
                user_id=user_id,
                candidate=candidate,
                attempts=attempts,
                attempt_record=attempt_record,
                started_at=started_at,
                exc=exc,
                error_class="sandbox_unreachable",
                final=True,
                trace_endpoint=trace_endpoint,
            )
            logger.error("Sandbox unreachable for routed model %s: %s", candidate.model, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
            ) from exc
        except WebSearchNotReachableError as exc:
            await _record_routing_attempt_error(
                db=db,
                log_writer=log_writer,
                plan=plan,
                api_key_id=api_key_id,
                user_id=user_id,
                candidate=candidate,
                attempts=attempts,
                attempt_record=attempt_record,
                started_at=started_at,
                exc=exc,
                error_class="web_search_unreachable",
                final=True,
                trace_endpoint=trace_endpoint,
            )
            logger.error("Web search backend unreachable for routed model %s: %s", candidate.model, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
            ) from exc
        except MaxToolIterationsExceeded as exc:
            await _record_routing_attempt_error(
                db=db,
                log_writer=log_writer,
                plan=plan,
                api_key_id=api_key_id,
                user_id=user_id,
                candidate=candidate,
                attempts=attempts,
                attempt_record=attempt_record,
                started_at=started_at,
                exc=exc,
                error_class="max_tool_iterations",
                final=True,
                trace_endpoint=trace_endpoint,
            )
            logger.warning("Tool loop iteration cap hit (routed): cap=%d", max_tool_iterations)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except BaseException as exc:
            _retryable, error_class = classify_upstream_error(exc)
            last_exc = exc
            await _record_routing_attempt_error(
                db=db,
                log_writer=log_writer,
                plan=plan,
                api_key_id=api_key_id,
                user_id=user_id,
                candidate=candidate,
                attempts=attempts,
                attempt_record=attempt_record,
                started_at=started_at,
                exc=exc,
                error_class=error_class,
                final=False,
                trace_endpoint=trace_endpoint,
            )
            logger.warning(
                "Routed provider call failed policy_id=%s provider=%s model=%s error_class=%s error=%s",
                plan.policy.policy_id,
                candidate.provider,
                candidate.provider_model,
                error_class,
                exc,
            )
            continue

        attempt_record.update(
            {
                "status": "success",
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            }
        )
        attempts.append(attempt_record)
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=candidate.provider_model,
            provider=candidate.provider,
            endpoint=trace_endpoint,
            user_id=user_id,
            project_id=plan.project_id,
            tags=plan.tags,
            response=completion,
        )
        trace_id = await record_route_trace(
            db,
            plan=plan,
            api_key_id=api_key_id,
            user_id=user_id,
            status="success",
            attempts=attempts,
            error_message=None,
            selected_candidate=candidate,
            endpoint=trace_endpoint,
        )
        _set_routing_response_headers(response=response, plan=plan, trace_id=trace_id, routed_model=candidate.model)
        if rate_limit_info:
            for key, value in rate_limit_headers(rate_limit_info).items():
                response.headers[key] = value
        return completion

    trace_id = await record_route_trace(
        db,
        plan=plan,
        api_key_id=api_key_id,
        user_id=user_id,
        status="error",
        attempts=attempts,
        error_message=str(last_exc) if last_exc else "No routed candidates were attempted",
        selected_candidate=None,
        endpoint=trace_endpoint,
    )
    _set_routing_response_headers(response=response, plan=plan, trace_id=trace_id)

    if last_exc is not None and isinstance(last_exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="All routed upstream providers timed out",
        ) from last_exc
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="All routed upstream providers failed",
    ) from last_exc


def _set_routing_response_headers(
    *,
    response: Response,
    plan: RoutingPlan,
    trace_id: str,
    routed_model: str | None = None,
) -> None:
    response.headers["X-Route-Trace-ID"] = trace_id
    response.headers["X-Routing-Policy-ID"] = plan.policy.policy_id
    response.headers["X-Routing-Strategy"] = plan.strategy
    response.headers["X-Routing-Tier"] = plan.target_tier
    response.headers["X-Routing-Fallback-Enabled"] = str(plan.fallback_enabled).lower()
    response.headers["X-Routing-Policy-Source"] = plan.policy_source
    if routed_model is not None:
        response.headers["X-Routed-Model"] = routed_model


async def _record_routing_attempt_error(
    *,
    db: AsyncSession,
    log_writer: LogWriter,
    plan: RoutingPlan,
    api_key_id: str | None,
    user_id: str | None,
    candidate: RoutingCandidate,
    attempts: list[dict[str, Any]],
    attempt_record: dict[str, Any],
    started_at: float,
    exc: BaseException,
    error_class: str,
    final: bool,
    trace_endpoint: str,
) -> str | None:
    """Record usage and trace data for a routed attempt error."""
    await log_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=candidate.provider_model,
        provider=candidate.provider,
        endpoint=trace_endpoint,
        user_id=user_id,
        project_id=plan.project_id,
        tags=plan.tags,
        error=str(exc),
    )
    attempt_record.update(
        {
            "status": "error",
            "error_class": error_class,
            "error_message": str(exc),
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
    )
    attempts.append(attempt_record)
    if not final:
        return None
    return await record_route_trace(
        db,
        plan=plan,
        api_key_id=api_key_id,
        user_id=user_id,
        status="error",
        attempts=attempts,
        error_message=str(exc),
        selected_candidate=candidate,
        endpoint=trace_endpoint,
    )

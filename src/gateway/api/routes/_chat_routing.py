import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, NoReturn

from any_llm import LLMProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from fastapi import HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._chat_non_streaming_completion import run_non_streaming_completion
from gateway.api.routes._chat_provider_errors import is_provider_timeout
from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._chat_request_fields import chat_provider_call_kwargs, chat_provider_request_fields
from gateway.api.routes._chat_tool_backend_errors import chat_tool_backend_failure, chat_tool_iteration_cap_failure
from gateway.api.routes._chat_tools import ChatToolSelection
from gateway.api.routes._usage import apply_rate_limit_headers, log_usage
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import RateLimitInfo
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_loop import MaxToolIterationsExceeded
from gateway.services.platform_gateway import classify_upstream_error
from gateway.services.provider_kwargs import get_provider_kwargs
from gateway.services.routing_context_policy import apply_context_policy
from gateway.services.routing_guardrail_redactions import apply_guardrail_redactions
from gateway.services.routing_policy_service import (
    DEFAULT_ROUTING_MODEL,
    RoutingCandidate,
    RoutingPlan,
    RoutingPolicyError,
    record_route_trace,
    resolve_routing_plan,
)
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError


@dataclass(frozen=True)
class _RoutingExecutionContext:
    db: AsyncSession
    log_writer: LogWriter
    plan: RoutingPlan
    api_key_id: str | None
    user_id: str | None
    trace_endpoint: str


@dataclass(frozen=True)
class _RoutingAttemptContext:
    candidate: RoutingCandidate
    attempts: list[dict[str, Any]]
    record: dict[str, Any]
    started_at: float


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
    request_fields = chat_provider_request_fields(
        request,
        tools_extracted=tool_selection.tools_extracted,
        remaining_user_tools=tool_selection.remaining_user_tools,
    )
    policy_config = plan.policy.config_dict()
    request_fields, _ = apply_guardrail_redactions(policy_config, request_fields)
    request_fields, _ = apply_context_policy(policy_config, request_fields)

    attempts: list[dict[str, Any]] = []
    last_exc: BaseException | None = None
    execution = _RoutingExecutionContext(
        db=db,
        log_writer=log_writer,
        plan=plan,
        api_key_id=api_key_id,
        user_id=user_id,
        trace_endpoint=trace_endpoint,
    )

    for candidate in plan.candidates:
        provider_kwargs = get_provider_kwargs(config, LLMProvider(candidate.provider))
        completion_kwargs = chat_provider_call_kwargs(provider_kwargs, request_fields, model=candidate.model)
        attempt = _RoutingAttemptContext(
            candidate=candidate,
            attempts=attempts,
            record={
                "position": candidate.position,
                "provider": candidate.provider,
                "model": candidate.provider_model,
                "model_key": candidate.model,
                "tier": candidate.tier,
            },
            started_at=time.perf_counter(),
        )

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
            failure = chat_tool_backend_failure(exc)
            await _raise_final_routing_attempt_error(
                execution=execution,
                attempt=attempt,
                exc=exc,
                error_class=failure.error_class,
                status_code=failure.status_code,
                detail=failure.detail,
                log=logger.error,
                log_message="Sandbox unreachable for routed model %s: %s",
                log_args=(candidate.model, exc),
            )
        except WebSearchNotReachableError as exc:
            failure = chat_tool_backend_failure(exc)
            await _raise_final_routing_attempt_error(
                execution=execution,
                attempt=attempt,
                exc=exc,
                error_class=failure.error_class,
                status_code=failure.status_code,
                detail=failure.detail,
                log=logger.error,
                log_message="Web search backend unreachable for routed model %s: %s",
                log_args=(candidate.model, exc),
            )
        except MaxToolIterationsExceeded as exc:
            failure = chat_tool_iteration_cap_failure(exc)
            await _raise_final_routing_attempt_error(
                execution=execution,
                attempt=attempt,
                exc=exc,
                error_class=failure.error_class,
                status_code=failure.status_code,
                detail=failure.detail,
                log=logger.warning,
                log_message="Tool loop iteration cap hit (routed): cap=%d",
                log_args=(max_tool_iterations,),
            )
        except BaseException as exc:
            _retryable, error_class = classify_upstream_error(exc)
            last_exc = exc
            await _record_routing_attempt_error(
                execution=execution,
                attempt=attempt,
                exc=exc,
                error_class=error_class,
                final=False,
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

        await _record_routing_attempt_success(
            execution=execution,
            response=response,
            rate_limit_info=rate_limit_info,
            attempt=attempt,
            completion=completion,
        )
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

    if is_provider_timeout(last_exc):
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


async def _record_routing_attempt_success(
    *,
    execution: _RoutingExecutionContext,
    response: Response,
    rate_limit_info: RateLimitInfo | None,
    attempt: _RoutingAttemptContext,
    completion: ChatCompletion,
) -> None:
    _finalize_routing_attempt_record(attempt, status="success")
    await log_usage(
        db=execution.db,
        log_writer=execution.log_writer,
        api_key_id=execution.api_key_id,
        model=attempt.candidate.provider_model,
        provider=attempt.candidate.provider,
        endpoint=execution.trace_endpoint,
        user_id=execution.user_id,
        project_id=execution.plan.project_id,
        tags=execution.plan.tags,
        response=completion,
    )
    trace_id = await record_route_trace(
        execution.db,
        plan=execution.plan,
        api_key_id=execution.api_key_id,
        user_id=execution.user_id,
        status="success",
        attempts=attempt.attempts,
        error_message=None,
        selected_candidate=attempt.candidate,
        endpoint=execution.trace_endpoint,
    )
    _set_routing_response_headers(
        response=response,
        plan=execution.plan,
        trace_id=trace_id,
        routed_model=attempt.candidate.model,
    )
    apply_rate_limit_headers(response, rate_limit_info)


async def _raise_final_routing_attempt_error(
    *,
    execution: _RoutingExecutionContext,
    attempt: _RoutingAttemptContext,
    exc: BaseException,
    error_class: str,
    status_code: int,
    detail: str,
    log: Callable[..., None],
    log_message: str,
    log_args: tuple[Any, ...],
) -> NoReturn:
    await _record_routing_attempt_error(
        execution=execution,
        attempt=attempt,
        exc=exc,
        error_class=error_class,
        final=True,
    )
    log(log_message, *log_args)
    raise HTTPException(status_code=status_code, detail=detail) from exc


async def _record_routing_attempt_error(
    *,
    execution: _RoutingExecutionContext,
    attempt: _RoutingAttemptContext,
    exc: BaseException,
    error_class: str,
    final: bool,
) -> str | None:
    """Record usage and trace data for a routed attempt error."""
    await log_usage(
        db=execution.db,
        log_writer=execution.log_writer,
        api_key_id=execution.api_key_id,
        model=attempt.candidate.provider_model,
        provider=attempt.candidate.provider,
        endpoint=execution.trace_endpoint,
        user_id=execution.user_id,
        project_id=execution.plan.project_id,
        tags=execution.plan.tags,
        error=str(exc),
    )
    _finalize_routing_attempt_record(
        attempt,
        status="error",
        error_class=error_class,
        error_message=str(exc),
    )
    if not final:
        return None
    return await record_route_trace(
        execution.db,
        plan=execution.plan,
        api_key_id=execution.api_key_id,
        user_id=execution.user_id,
        status="error",
        attempts=attempt.attempts,
        error_message=str(exc),
        selected_candidate=attempt.candidate,
        endpoint=execution.trace_endpoint,
    )


def _finalize_routing_attempt_record(
    attempt: _RoutingAttemptContext,
    *,
    status: str,
    error_class: str | None = None,
    error_message: str | None = None,
) -> None:
    attempt.record.update(
        {
            "status": status,
            "duration_ms": round((time.perf_counter() - attempt.started_at) * 1000, 2),
        }
    )
    if error_class is not None:
        attempt.record["error_class"] = error_class
    if error_message is not None:
        attempt.record["error_message"] = error_message
    attempt.attempts.append(attempt.record)

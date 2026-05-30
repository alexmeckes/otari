import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack
from typing import Annotated, Any, NamedTuple

import httpx
from any_llm import AnyLLM, LLMProvider, acompletion
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._chat_streaming_response import build_chat_streaming_response
from gateway.api.routes._chat_tools import resolve_chat_tool_selection
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes._usage import log_usage, rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.budget_service import validate_project_budget, validate_tag_budgets, validate_user_budget
from gateway.services.chat_tool_config import (
    build_web_search_backend,
    resolve_sandbox_purpose_hint,
    strip_gateway_fields,
)
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_client import MCPClientPool
from gateway.services.mcp_loop import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
    inject_purpose_hints,
    mcp_tool_loop,
    mcp_tool_loop_stream,
)
from gateway.services.platform_gateway import (
    ResolvedAttempt,
    ResolvedRoute,
    classify_upstream_error,
    extract_platform_user_token,
    report_platform_usage,
    resolve_platform_credentials,
    resolve_platform_mcp_servers,
)
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
from gateway.services.sandbox_backend import SandboxBackend, SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError
from gateway.streaming import (
    StreamingAttemptFailure,
    iterate_streaming_attempts,
)

router = APIRouter(prefix="/v1/chat", tags=["chat"])

# Streaming first-chunk timeouts (platform-mode fallback). Plain LLM streams
# rarely take long to produce a first token, so a tight cap keeps failed-
# attempt latency low. Tool-loop streams may reason before emitting tokens
# or a tool_call (especially with extended thinking), so they get more
# headroom. Both are operator-tunable via `config.platform`.
_DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS = 2000
_DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP = 30000
_STREAM_FIRST_CHUNK_TIMEOUT_MS_KEY = "streaming_first_chunk_timeout_ms"
_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP_KEY = "streaming_first_chunk_timeout_ms_tool_loop"


async def _run_non_streaming_completion(
    *,
    completion_kwargs: dict[str, Any],
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
        async with MCPClientPool(mcp_server_configs) as pool:
            mcp_kwargs = {
                **completion_kwargs,
                "messages": inject_purpose_hints(
                    completion_kwargs["messages"],
                    pool.purpose_hints(),
                    header=tools_header,
                ),
            }
            return await mcp_tool_loop(
                completion_kwargs=mcp_kwargs,
                pool=pool,
                max_iterations=max_tool_iterations,
                on_first_response=on_first_response,
            )
    if use_sandbox:
        assert sandbox_url is not None
        sandbox_hint = resolve_sandbox_purpose_hint(sandbox_tool_entry)
        async with SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint) as backend:
            sandbox_kwargs = {
                **completion_kwargs,
                "messages": inject_purpose_hints(
                    completion_kwargs["messages"],
                    backend.purpose_hints(),
                    header=tools_header,
                ),
            }
            return await mcp_tool_loop(
                completion_kwargs=sandbox_kwargs,
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
            web_kwargs = {
                **completion_kwargs,
                "messages": inject_purpose_hints(
                    completion_kwargs["messages"],
                    web_backend.purpose_hints(),
                    header=tools_header,
                ),
            }
            return await mcp_tool_loop(
                completion_kwargs=web_kwargs,
                pool=web_backend,  # type: ignore[arg-type]
                max_iterations=max_tool_iterations,
                on_first_response=on_first_response,
            )
    return await acompletion(**completion_kwargs)  # type: ignore[return-value]


async def _run_standalone_routing_plan(
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
    mcp_server_configs: list[McpServerConfig] | None,
    max_tool_iterations: int,
    sandbox_tool_entry: dict[str, Any] | None,
    sandbox_url: str | None,
    use_sandbox: bool,
    web_search_tool_entry: dict[str, Any] | None,
    web_search_url: str | None,
    use_web_search: bool,
    remaining_user_tools: list[dict[str, Any]] | None,
    trace_endpoint: str,
) -> ChatCompletion:
    """Execute a standalone routing plan with pre-response fallback."""
    request_fields = strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
        remaining_user_tools=remaining_user_tools,
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
            completion = await _run_non_streaming_completion(
                completion_kwargs=completion_kwargs,
                mcp_server_configs=mcp_server_configs,
                max_tool_iterations=max_tool_iterations,
                tools_header=request.tools_header,
                use_sandbox=use_sandbox,
                sandbox_url=sandbox_url,
                sandbox_tool_entry=sandbox_tool_entry,
                use_web_search=use_web_search,
                web_search_url=web_search_url,
                web_search_tool_entry=web_search_tool_entry,
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
        response.headers["X-Route-Trace-ID"] = trace_id
        response.headers["X-Routing-Policy-ID"] = plan.policy.policy_id
        response.headers["X-Routed-Model"] = candidate.model
        response.headers["X-Routing-Strategy"] = plan.strategy
        response.headers["X-Routing-Tier"] = plan.target_tier
        response.headers["X-Routing-Fallback-Enabled"] = str(plan.fallback_enabled).lower()
        response.headers["X-Routing-Policy-Source"] = plan.policy_source
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
    response.headers["X-Route-Trace-ID"] = trace_id
    response.headers["X-Routing-Policy-ID"] = plan.policy.policy_id
    response.headers["X-Routing-Strategy"] = plan.strategy
    response.headers["X-Routing-Tier"] = plan.target_tier
    response.headers["X-Routing-Fallback-Enabled"] = str(plan.fallback_enabled).lower()
    response.headers["X-Routing-Policy-Source"] = plan.policy_source

    if last_exc is not None and isinstance(last_exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="All routed upstream providers timed out",
        ) from last_exc
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="All routed upstream providers failed",
    ) from last_exc


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
    api_key: APIKey | None = None
    api_key_id: str | None = None
    user_id: str | None = None
    rate_limit_info: RateLimitInfo | None = None
    platform_mode = config.is_platform_mode
    route: ResolvedRoute | None = None
    routing_plan: RoutingPlan | None = None
    user_token: str | None = None  # set inside the platform_mode branch; referenced again later

    if platform_mode:
        user_token = extract_platform_user_token(raw_request)
        start_time = time.perf_counter()
        route = await resolve_platform_credentials(
            config=config,
            user_token=user_token,
            model_selector=request.model,
        )
        resolve_latency_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Otari-Request-ID"] = route.request_id
        logger.info(
            "Platform resolve succeeded request_id=%s attempts=%d fallback_enabled=%s resolve_latency_ms=%.2f",
            route.request_id,
            len(route.attempts),
            route.fallback_enabled,
            resolve_latency_ms,
        )
    else:
        if db is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session unavailable",
            )

        api_key, is_master_key = await verify_api_key_or_master_key(raw_request, db, config)
        api_key_id = api_key.id if api_key else None
        user_id = resolve_user_id(
            user_id_from_request=request.user,
            api_key=api_key,
            is_master_key=is_master_key,
            master_key_error=HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="When using master key, 'user' field is required in request body",
            ),
            no_api_key_error=HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API key validation failed",
            ),
            no_user_error=HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API key has no associated user",
            ),
        )

        rate_limit_info = check_rate_limit(raw_request, user_id)
        _ = await validate_user_budget(db, user_id, request.model, strategy=config.budget_strategy)
        if request.project_id is not None:
            _ = await validate_project_budget(db, request.project_id, request.model, strategy=config.budget_strategy)
        _ = await validate_tag_budgets(db, request.tags, request.model, strategy=config.budget_strategy)
        if config.budget_strategy == "for_update":
            await db.rollback()

    # Workspace-scoped MCP server references (platform mode only). Callers
    # pass `mcp_server_ids: [uuid, ...]` instead of inlining each config; we
    # resolve them against the platform's `/gateway/mcp-servers/resolve`
    # endpoint and combine with any inline `mcp_servers` so the downstream
    # MCP loop sees a single list. In standalone mode there's no platform
    # to consult, so we reject the field with a 400 rather than silently
    # ignoring it.
    if request.mcp_server_ids and not platform_mode:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mcp_server_ids is only available in platform mode",
        )
    if platform_mode and request.mcp_server_ids:
        assert user_token is not None  # guaranteed by the platform_mode branch above
        resolved_mcp_servers = await resolve_platform_mcp_servers(
            config=config,
            user_token=user_token,
            mcp_server_ids=request.mcp_server_ids,
        )
        request.mcp_servers = (request.mcp_servers or []) + resolved_mcp_servers

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
    tools_extracted = tool_selection.tools_extracted

    if not platform_mode and request.model == DEFAULT_ROUTING_MODEL:
        if request.stream:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Routing policies do not support streaming chat completions yet",
            )
        assert db is not None
        try:
            routing_plan = await resolve_routing_plan(
                db,
                request_body=request.model_dump(exclude_unset=True),
                project_id=request.project_id,
                tags=request.tags,
            )
        except RoutingPolicyError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    # ------------------------------------------------------------------
    # Streaming path: iterate `route.attempts` before any bytes are flushed,
    # then commit to the first attempt that yields a chunk. Implemented in
    # `_run_streaming_with_fallback` via `iterate_streaming_attempts`.
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
            stream_mcp_configs = request.mcp_servers
            stream_max_tool_iterations = min(
                request.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
                MAX_TOOL_ITERATIONS_CAP,
            )
            try:
                return await _run_streaming_with_fallback(
                    route=route,
                    request=request,
                    response=response,
                    config=config,
                    background_tasks=background_tasks,
                    rate_limit_info=rate_limit_info,
                    mcp_server_configs=stream_mcp_configs,
                    use_sandbox=use_sandbox,
                    sandbox_url=sandbox_url,
                    sandbox_tool_entry=sandbox_tool_entry,
                    use_web_search=use_web_search,
                    web_search_url=web_search_url,
                    web_search_tool_entry=web_search_tool_entry,
                    remaining_user_tools=remaining_user_tools,
                    tools_extracted=tools_extracted,
                    max_tool_iterations=stream_max_tool_iterations,
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
                if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=(
                            "LLM provider timeout" if len(route.attempts) <= 1 else "All upstream providers timed out"
                        ),
                    ) from exc
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=("LLM provider error" if len(route.attempts) <= 1 else "All upstream providers failed"),
                ) from exc

        # Standalone path: single attempt, no fallback (no `route.attempts`).
        # Platform-mode requests (including tool modes) take the multi-attempt
        # branch above; this block runs only when `model` is a literal
        # ``provider:model`` selector and no routing policy applies.
        provider, model = AnyLLM.split_model_provider(request.model)
        provider_kwargs = get_provider_kwargs(config, provider)

        mcp_server_configs = request.mcp_servers
        max_tool_iterations = min(
            request.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
            MAX_TOOL_ITERATIONS_CAP,
        )

        request_fields = strip_gateway_fields(
            request.model_dump(exclude_unset=True),
            tools_extracted=tools_extracted,
            remaining_user_tools=remaining_user_tools,
        )
        completion_kwargs = {**provider_kwargs, **request_fields}
        if completion_kwargs.get("stream_options") is None:
            completion_kwargs["stream_options"] = {"include_usage": True}

        try:
            if mcp_server_configs:
                # Bind the truthy value to a non-Optional local so mypy can
                # narrow inside the nested `_mcp_stream` closure.
                pool_configs = mcp_server_configs

                async def _mcp_stream() -> AsyncIterator[ChatCompletionChunk]:
                    async with MCPClientPool(pool_configs) as pool:
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

                stream: AsyncIterator[ChatCompletionChunk] = _mcp_stream()
            elif use_sandbox:
                # SandboxBackend duck-types as MCPClientPool — same tool-loop helper.
                # Eagerly open the backend *before* constructing the
                # StreamingResponse so that a failure to reach the sandbox
                # surfaces synchronously as an HTTP error. A lazy `async with`
                # inside the generator would only run after the response
                # was committed, and SandboxNotReachableError would land in
                # the SSE channel after a 200 OK header — confusing for
                # clients that expected a normal HTTP failure.
                assert sandbox_url is not None
                sandbox_hint = resolve_sandbox_purpose_hint(sandbox_tool_entry)
                sandbox_backend = SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint)
                await sandbox_backend.__aenter__()  # may raise SandboxNotReachableError

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

                stream = _sandbox_stream()
            elif use_web_search:
                # Same eager-open rationale as the sandbox path above.
                assert web_search_url is not None
                assert web_search_tool_entry is not None
                web_search_backend = build_web_search_backend(
                    base_url=web_search_url,
                    tool_entry=web_search_tool_entry,
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

                stream = _web_search_stream()
            else:
                stream = await acompletion(**completion_kwargs)  # type: ignore[assignment]
        except HTTPException:
            raise
        except SandboxNotReachableError as exc:
            # The sandbox is part of the gateway's own infra, not the LLM
            # provider — surface a clearer status so operators don't chase
            # a "provider outage" that's actually the sandbox container
            # being down. 502 keeps "upstream dependency failed" semantics.
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
            if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="LLM provider timeout",
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM provider error",
            ) from exc

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
        if route is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error: missing route context",
            )
        platform_route = route
        attempts_to_try = platform_route.attempts
        if not attempts_to_try:
            logger.error(
                "Platform returned empty attempts list request_id=%s",
                platform_route.request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Authorization service returned no resolvable provider",
            )
    else:
        if routing_plan is None:
            provider, model = AnyLLM.split_model_provider(request.model)
            provider_kwargs = get_provider_kwargs(config, provider)
        attempts_to_try = []  # standalone path doesn't use the attempts list

    class _AttemptFailure(NamedTuple):
        position: int
        provider: str
        model: str
        error_class: str

    failures: list[_AttemptFailure] = []
    last_exc: BaseException | None = None

    if platform_mode:
        base_request_fields = strip_gateway_fields(
            request.model_dump(exclude_unset=True),
            tools_extracted=tools_extracted,
            remaining_user_tools=remaining_user_tools,
        )
        for attempt in attempts_to_try:
            attempt_provider = LLMProvider(attempt.provider)
            attempt_model = attempt.model
            attempt_kwargs: dict[str, Any] = {"api_key": attempt.api_key}
            if attempt.api_base:
                attempt_kwargs["api_base"] = attempt.api_base

            completion_kwargs = {
                **attempt_kwargs,
                **base_request_fields,
                "model": f"{attempt_provider.value}:{attempt_model}",
            }

            # Per-attempt lock-in flag. Flipped the moment the upstream
            # returns its first assistant message. After that, any failure
            # terminates the request — fallback would replay a provider-
            # specific transcript on a different provider, which has
            # undefined semantics.
            locked_in = False

            def _mark_locked_in(_pos: int = attempt.position) -> None:
                nonlocal locked_in
                locked_in = True
                logger.info(
                    "Tool-loop lock-in request_id=%s position=%d provider=%s model=%s",
                    platform_route.request_id,
                    _pos,
                    attempt.provider,
                    attempt.model,
                )

            try:
                completion = await _run_non_streaming_completion(
                    completion_kwargs=completion_kwargs,
                    mcp_server_configs=mcp_server_configs,
                    max_tool_iterations=max_tool_iterations,
                    tools_header=request.tools_header,
                    use_sandbox=use_sandbox,
                    sandbox_url=sandbox_url,
                    sandbox_tool_entry=sandbox_tool_entry,
                    use_web_search=use_web_search,
                    web_search_url=web_search_url,
                    web_search_tool_entry=web_search_tool_entry,
                    on_first_response=_mark_locked_in,
                )
            except HTTPException:
                raise
            except MaxToolIterationsExceeded as exc:
                # The gateway's own iteration cap was hit, not an upstream
                # failure. Surface a distinct 422 so callers can tell a
                # runaway tool loop apart from a provider outage.
                logger.warning(
                    "Tool loop iteration cap hit request_id=%s position=%d cap=%d",
                    platform_route.request_id,
                    attempt.position,
                    max_tool_iterations,
                )
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
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
                    platform_route.request_id,
                    attempt.position,
                    attempt.provider,
                    attempt.model,
                    error_class,
                    retryable,
                    locked_in,
                )
                last_exc = exc
                # Locked-in: at least one tool-loop round produced an
                # assistant message on this attempt. Subsequent failures
                # cannot be transparently retried on another provider — the
                # conversation now contains provider-specific tool_call ids
                # and reasoning blocks. Surface immediately.
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

            # Success on this attempt.
            background_tasks.add_task(
                report_platform_usage,
                config,
                attempt.attempt_id,
                "success",
                completion.usage,
                None,
            )
            response.headers["X-Correlation-ID"] = attempt.attempt_id
            if rate_limit_info:
                for key, value in rate_limit_headers(rate_limit_info).items():
                    response.headers[key] = value
            return completion

        # All attempts exhausted with retryable errors.
        logger.error(
            "All upstream attempts failed request_id=%s failures=%s",
            platform_route.request_id,
            failures,
        )
        is_single_attempt = len(attempts_to_try) <= 1
        if last_exc is not None and isinstance(last_exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
            detail = "LLM provider timeout" if is_single_attempt else "All upstream providers timed out"
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=detail,
            ) from last_exc
        detail = "LLM provider error" if is_single_attempt else "All upstream providers failed"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        ) from last_exc

    if routing_plan is not None:
        assert db is not None
        return await _run_standalone_routing_plan(
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
        )

    # Standalone path (no platform / no fallback). Same MCP / sandbox /
    # web_search semantics as platform mode above: if `mcp_servers` is set,
    # the request goes through the MCP tool-use loop; if `tools` includes a
    # code_execution entry, the request goes through the SandboxBackend
    # tool-use loop; if `tools` includes a web_search entry, the request
    # goes through the WebSearchBackend tool-use loop; otherwise a single
    # ``acompletion`` call.
    request_fields = strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=tools_extracted,
        remaining_user_tools=remaining_user_tools,
    )
    completion_kwargs = {**provider_kwargs, **request_fields}

    try:
        completion = await _run_non_streaming_completion(
            completion_kwargs=completion_kwargs,
            mcp_server_configs=mcp_server_configs,
            max_tool_iterations=max_tool_iterations,
            tools_header=request.tools_header,
            use_sandbox=use_sandbox,
            sandbox_url=sandbox_url,
            sandbox_tool_entry=sandbox_tool_entry,
            use_web_search=use_web_search,
            web_search_url=web_search_url,
            web_search_tool_entry=web_search_tool_entry,
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
    except MaxToolIterationsExceeded as e:
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
                error=str(e),
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
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
                error=str(e),
            )

        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        if isinstance(e, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM provider timeout",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return completion


async def _run_streaming_with_fallback(
    *,
    route: ResolvedRoute,
    request: ChatCompletionRequest,
    response: Response,
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
    """Iterate route.attempts for a streaming request, falling through on any
    attempt that fails before its first chunk arrives.

    Once an attempt yields its first chunk, we commit and start flushing to
    the client — errors past that point propagate to the SSE channel as
    today. This is the streaming analogue of the non-streaming
    pre-lock-in-fallback flow in the request handler above.

    Tool-loop modes (sandbox / web_search / MCP) are layered on top using
    the same pre-first-chunk fallback semantics: the upstream
    ``acompletion(stream=True)`` call inside ``mcp_tool_loop_stream`` runs
    lazily when ``iterate_streaming_attempts`` pulls the first chunk; if
    that call fails (or any retryable error fires before a chunk arrives)
    we move to the next attempt with a clean conversation slate.
    """
    tool_mode = bool(mcp_server_configs) or use_sandbox or use_web_search

    base_request_fields = strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=tools_extracted,
        remaining_user_tools=remaining_user_tools,
    )

    # Tool-mode streams need more headroom on first-chunk wait: the model
    # may reason briefly before emitting tokens or a tool_call, especially
    # with extended thinking. Keep the existing tight default for plain
    # streams so failed-attempt latency stays low when no tools are in play.
    if tool_mode:
        first_chunk_timeout_seconds = (
            int(
                config.platform.get(
                    _STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP_KEY,
                    _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP,
                )
            )
            / 1000
        )
    else:
        first_chunk_timeout_seconds = (
            int(
                config.platform.get(
                    _STREAM_FIRST_CHUNK_TIMEOUT_MS_KEY,
                    _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS,
                )
            )
            / 1000
        )

    # Open the tool backend(s) once before iterating attempts. Eager open
    # lets gateway-side dependency failures (sandbox unreachable, web
    # search backend unreachable) surface as a normal HTTP error instead
    # of as a mid-SSE error event after the 200 OK header. The exit stack
    # is closed when streaming completes (success or error path inside
    # the wrapped iterator) or, if no attempt commits, in the outer
    # except handler below.
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
        # Eager-open failure (e.g. SandboxNotReachableError) — propagate so
        # the route handler maps it to the existing HTTP status. Nothing to
        # clean up on the stack yet because the entry failed.
        await backend_stack.aclose()
        raise

    async def _build_for_attempt(
        attempt: ResolvedAttempt,
    ) -> AsyncIterator[ChatCompletionChunk]:
        attempt_provider = LLMProvider(attempt.provider)
        provider_kwargs: dict[str, Any] = {"api_key": attempt.api_key}
        if attempt.api_base:
            provider_kwargs["api_base"] = attempt.api_base
        completion_kwargs = {
            **provider_kwargs,
            **base_request_fields,
            "model": f"{attempt_provider.value}:{attempt.model}",
        }
        if completion_kwargs.get("stream_options") is None:
            completion_kwargs["stream_options"] = {"include_usage": True}
        if pool_for_loop is None:
            return await acompletion(**completion_kwargs)  # type: ignore[return-value]
        kwargs = {
            **completion_kwargs,
            "messages": inject_purpose_hints(
                completion_kwargs["messages"],
                pool_for_loop.purpose_hints(),
                header=request.tools_header,
            ),
        }
        return mcp_tool_loop_stream(
            completion_kwargs=kwargs,
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
        # No attempt yielded a first chunk — close the tool backend before
        # propagating the failure. The route handler maps it to HTTP.
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
        db=None,  # platform mode doesn't use the local DB
        log_writer=None,  # unused when db is None
        api_key_id=None,
        user_id=None,
        project_id=None,
        tags=None,
        rate_limit_info=rate_limit_info,
    )

import time
from dataclasses import dataclass

from fastapi import HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import verify_api_key_or_master_key
from gateway.api.routes._budget_checks import validate_scoped_request_budgets
from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._helpers import resolve_user_id
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.platform_gateway import (
    ResolvedRoute,
    extract_platform_user_token,
    resolve_platform_credentials,
    resolve_platform_mcp_servers,
)


@dataclass(frozen=True)
class ChatRequestContext:
    api_key_id: str | None = None
    user_id: str | None = None
    rate_limit_info: RateLimitInfo | None = None
    route: ResolvedRoute | None = None
    user_token: str | None = None


async def resolve_chat_request_context(
    *,
    raw_request: Request,
    response: Response,
    request: ChatCompletionRequest,
    db: AsyncSession | None,
    config: GatewayConfig,
) -> ChatRequestContext:
    """Resolve platform or standalone request context before provider execution."""
    if config.is_platform_mode:
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
        return ChatRequestContext(route=route, user_token=user_token)

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
    await validate_scoped_request_budgets(
        db,
        user_id=user_id,
        model=request.model,
        project_id=request.project_id,
        tags=request.tags,
        strategy=config.budget_strategy,
    )

    return ChatRequestContext(
        api_key_id=api_key_id,
        user_id=user_id,
        rate_limit_info=rate_limit_info,
    )


async def resolve_chat_mcp_server_ids(
    *,
    request: ChatCompletionRequest,
    config: GatewayConfig,
    platform_mode: bool,
    user_token: str | None,
) -> None:
    """Resolve platform-scoped MCP server ids into inline server configs."""
    if request.mcp_server_ids and not platform_mode:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mcp_server_ids is only available in platform mode",
        )

    if platform_mode and request.mcp_server_ids:
        assert user_token is not None
        resolved_mcp_servers = await resolve_platform_mcp_servers(
            config=config,
            user_token=user_token,
            mcp_server_ids=request.mcp_server_ids,
        )
        request.mcp_servers = (request.mcp_servers or []) + resolved_mcp_servers

"""Platform-mode gateway calls for credentials, MCP servers, and usage reports."""

import asyncio
import uuid
from typing import Any, NoReturn

import httpx
from any_llm.types.completion import CompletionUsage
from fastapi import HTTPException, Request, status
from pydantic import BaseModel

from gateway.core.config import GatewayConfig
from gateway.models.mcp import McpServerConfig
from gateway.services.routing_policy_shape import split_model_selector as _split_model_selector

_USAGE_NON_RETRYABLE_STATUS_CODES = {401, 404, 409, 422}
_PLATFORM_RESOLUTION_PASSTHROUGH_STATUS_CODES = {401, 402, 403, 404, 429}

# Status codes that cause the gateway to move on to the next attempt in a
# multi-attempt route. 401/403 are included because users configure multi-attempt
# routing policies on the platform precisely to handle credential outages.
_FALLBACK_RETRYABLE_STATUS_CODES = {401, 403, 408, 429, 500, 502, 503, 504}
_FALLBACK_NON_RETRYABLE_STATUS_CODES = {400, 422}


class ResolvedAttempt(BaseModel):
    """A single resolution attempt returned by the platform."""

    attempt_id: str
    position: int
    provider: str
    model: str
    api_base: str | None = None
    api_key: str
    managed: bool

    @property
    def model_selector(self) -> str:
        return f"{self.provider}:{self.model}"


class ResolvedRoute(BaseModel):
    """The full resolution plan returned by the platform."""

    request_id: str
    fallback_enabled: bool
    attempts: list[ResolvedAttempt]


def extract_platform_user_token(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    token = auth_header[7:].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    return token


def _platform_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _platform_base_url_or_raise(config: GatewayConfig) -> str:
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Platform mode is misconfigured",
        )
    return str(platform_base_url)


def _platform_user_headers(config: GatewayConfig, user_token: str) -> dict[str, str]:
    return {
        "X-Gateway-Token": config.platform_token or "",
        "X-User-Token": user_token,
    }


def _platform_resolve_timeout_seconds(config: GatewayConfig) -> float:
    return int(config.platform.get("resolve_timeout_ms", 5000)) / 1000


def _safe_detail_from_platform(response: httpx.Response, fallback: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return fallback

    detail = payload.get("detail") if isinstance(payload, dict) else None
    return detail if isinstance(detail, str) else fallback


def _raise_platform_resolution_error(response: httpx.Response, passthrough_fallback: str) -> NoReturn:
    if response.status_code in _PLATFORM_RESOLUTION_PASSTHROUGH_STATUS_CODES:
        detail = _safe_detail_from_platform(response, passthrough_fallback)
        headers: dict[str, str] | None = None
        if response.status_code == 429 and response.headers.get("Retry-After"):
            headers = {"Retry-After": response.headers["Retry-After"]}
        raise HTTPException(status_code=response.status_code, detail=detail, headers=headers)

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Authorization service unavailable",
    )


async def _post_platform(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_seconds: float,
) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        return await client.post(url, headers=headers, json=body)


async def resolve_platform_credentials(
    config: GatewayConfig,
    user_token: str,
    model_selector: str,
) -> ResolvedRoute:
    provider, model_name = _split_model_selector(model_selector)
    platform_base_url = _platform_base_url_or_raise(config)
    resolve_url = _platform_url(platform_base_url, "/gateway/provider-keys/resolve")
    resolve_headers = _platform_user_headers(config, user_token)
    resolve_body: dict[str, Any] = {"model": model_name}
    if provider:
        resolve_body["provider"] = provider

    try:
        response = await _post_platform(
            url=resolve_url,
            headers=resolve_headers,
            body=resolve_body,
            timeout_seconds=_platform_resolve_timeout_seconds(config),
        )
    except (httpx.TimeoutException, httpx.NetworkError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        ) from None

    if response.status_code == 200:
        payload = response.json()
        return parse_resolve_payload(payload)

    _raise_platform_resolution_error(response, "Authorization request rejected")


def parse_resolve_payload(payload: dict[str, Any]) -> ResolvedRoute:
    """Build a ResolvedRoute from the current or legacy platform payload shape."""
    attempts_payload = payload.get("attempts")
    if attempts_payload is not None:
        attempts = [
            ResolvedAttempt(
                attempt_id=str(att["attempt_id"]),
                position=int(att["position"]),
                provider=str(att["provider"]),
                model=str(att["model"]),
                api_base=att.get("api_base"),
                api_key=str(att["api_key"]),
                managed=bool(att.get("managed", False)),
            )
            for att in attempts_payload
        ]
        return ResolvedRoute(
            request_id=str(payload["request_id"]),
            fallback_enabled=bool(payload.get("fallback_enabled", False)),
            attempts=attempts,
        )

    correlation_id = str(payload["correlation_id"])
    return ResolvedRoute(
        request_id=correlation_id,
        fallback_enabled=False,
        attempts=[
            ResolvedAttempt(
                attempt_id=correlation_id,
                position=0,
                provider=str(payload["provider"]),
                model=str(payload["model"]),
                api_base=payload.get("api_base"),
                api_key=str(payload["api_key"]),
                managed=bool(payload.get("managed", False)),
            )
        ],
    )


def classify_upstream_error(exc: BaseException) -> tuple[bool, str]:
    """Return whether an upstream provider error can fall back, plus its class."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return True, "timeout"
    if isinstance(exc, httpx.NetworkError):
        return True, "conn_err"

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            status_code = getattr(resp, "status_code", None)

    if isinstance(status_code, int):
        if status_code in _FALLBACK_NON_RETRYABLE_STATUS_CODES:
            return False, f"http_{status_code}"
        if status_code in _FALLBACK_RETRYABLE_STATUS_CODES or 500 <= status_code <= 599:
            return True, f"http_{status_code}"
        return False, f"http_{status_code}"

    return False, "unknown"


async def resolve_platform_mcp_servers(
    config: GatewayConfig,
    user_token: str,
    mcp_server_ids: list[uuid.UUID],
) -> list[McpServerConfig]:
    """Swap workspace-scoped MCP server ids for inline configs by calling the platform."""
    platform_base_url = _platform_base_url_or_raise(config)
    resolve_url = _platform_url(platform_base_url, "/gateway/mcp-servers/resolve")
    headers = _platform_user_headers(config, user_token)
    body: dict[str, Any] = {"mcp_server_ids": [str(uid) for uid in mcp_server_ids]}

    try:
        response = await _post_platform(
            url=resolve_url,
            headers=headers,
            body=body,
            timeout_seconds=_platform_resolve_timeout_seconds(config),
        )
    except (httpx.TimeoutException, httpx.NetworkError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        ) from None

    if response.status_code == 200:
        payload = response.json()
        return [
            McpServerConfig(
                name=s["name"],
                url=s["url"],
                authorization_token=s.get("authorization_token"),
                purpose_hint=s.get("purpose_hint"),
                allowed_tools=s.get("allowed_tools"),
            )
            for s in payload.get("servers", [])
        ]

    _raise_platform_resolution_error(response, "MCP server resolution failed")


async def report_platform_usage(
    config: GatewayConfig,
    correlation_id: str,
    outcome: str,
    usage: CompletionUsage | None,
    error_class: str | None = None,
) -> None:
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        return

    timeout_ms = int(config.platform.get("usage_timeout_ms", 5000))
    max_retries = int(config.platform.get("usage_max_retries", 3))
    usage_url = _platform_url(platform_base_url, "/gateway/usage")
    headers = {"X-Gateway-Token": config.platform_token or ""}

    payload: dict[str, Any] = {"correlation_id": correlation_id, "status": outcome}
    if outcome == "success":
        token_usage = usage or CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        payload["usage"] = {
            "prompt_tokens": token_usage.prompt_tokens,
            "completion_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens,
        }
    elif error_class is not None:
        payload["error_class"] = error_class

    delay_seconds = 0.25
    for attempt in range(1, max_retries + 1):
        should_retry = False
        try:
            response = await _post_platform(
                url=usage_url,
                headers=headers,
                body=payload,
                timeout_seconds=timeout_ms / 1000,
            )
            if response.status_code == 204:
                return
            if response.status_code in _USAGE_NON_RETRYABLE_STATUS_CODES:
                return
            should_retry = response.status_code >= 500
        except (httpx.TimeoutException, httpx.NetworkError):
            should_retry = True

        if not should_retry or attempt == max_retries:
            return

        await asyncio.sleep(delay_seconds)
        delay_seconds *= 2

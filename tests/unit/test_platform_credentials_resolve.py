from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from gateway.core.config import GatewayConfig
from gateway.services import platform_gateway


def _request_with_authorization(value: str | None) -> Request:
    headers = []
    if value is not None:
        headers.append((b"authorization", value.encode("utf-8")))
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers})


def test_extract_platform_user_token_trims_bearer_value() -> None:
    request = _request_with_authorization("Bearer  user-token  ")

    assert platform_gateway.extract_platform_user_token(request) == "user-token"


@pytest.mark.parametrize("authorization", [None, "Basic user-token", "Bearer   "])
def test_extract_platform_user_token_rejects_missing_or_blank_bearer_value(authorization: str | None) -> None:
    with pytest.raises(HTTPException) as exc_info:
        platform_gateway.extract_platform_user_token(_request_with_authorization(authorization))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing authentication token"


@pytest.mark.asyncio
async def test_resolve_platform_credentials_uses_shared_request_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        captured["timeout_seconds"] = timeout_seconds
        return httpx.Response(
            200,
            json={
                "request_id": "request-1",
                "fallback_enabled": False,
                "attempts": [
                    {
                        "attempt_id": "attempt-1",
                        "position": 0,
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "api_key": "sk-test",
                        "managed": True,
                    },
                ],
            },
        )

    monkeypatch.setattr(platform_gateway, "_post_platform", fake_post_platform)
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    route = await platform_gateway.resolve_platform_credentials(
        GatewayConfig(
            mode="platform",
            platform={"base_url": "https://platform.local/api/v1/", "resolve_timeout_ms": 2500},
        ),
        "user-token",
        "openai/gpt-4o-mini",
    )

    assert captured == {
        "url": "https://platform.local/api/v1/gateway/provider-keys/resolve",
        "headers": {
            "X-Gateway-Token": "gw_test_token",
            "X-User-Token": "user-token",
        },
        "body": {
            "model": "gpt-4o-mini",
            "provider": "openai",
        },
        "timeout_seconds": 2.5,
    }
    assert route.request_id == "request-1"
    assert route.attempts[0].model == "gpt-4o-mini"
    assert route.attempts[0].model_selector == "openai:gpt-4o-mini"


def test_resolved_attempt_provider_kwargs_include_optional_api_base() -> None:
    attempt = platform_gateway.ResolvedAttempt(
        attempt_id="attempt-1",
        position=0,
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        api_base="https://api.openai.com/v1",
        managed=True,
    )

    assert attempt.provider_kwargs == {
        "api_key": "sk-test",
        "api_base": "https://api.openai.com/v1",
    }


def test_resolved_attempt_provider_kwargs_omit_missing_api_base() -> None:
    attempt = platform_gateway.ResolvedAttempt(
        attempt_id="attempt-1",
        position=0,
        provider="anthropic",
        model="claude-3-5-sonnet",
        api_key="sk-test",
        managed=True,
    )

    assert attempt.provider_kwargs == {"api_key": "sk-test"}


def test_parse_resolve_payload_maps_legacy_payload_to_single_attempt() -> None:
    route = platform_gateway.parse_resolve_payload(
        {
            "correlation_id": "corr-1",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet",
            "api_key": "sk-test",
            "api_base": "https://anthropic.local",
            "managed": True,
        }
    )

    assert route.request_id == "corr-1"
    assert route.fallback_enabled is False
    assert len(route.attempts) == 1
    attempt = route.attempts[0]
    assert attempt.attempt_id == "corr-1"
    assert attempt.position == 0
    assert attempt.provider == "anthropic"
    assert attempt.model == "claude-3-5-sonnet"
    assert attempt.api_key == "sk-test"
    assert attempt.api_base == "https://anthropic.local"
    assert attempt.managed is True

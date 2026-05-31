from typing import Any

import httpx
import pytest

from gateway.core.config import GatewayConfig
from gateway.services import platform_gateway


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

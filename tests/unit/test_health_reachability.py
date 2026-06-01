from typing import Any

import httpx
import pytest

from gateway.api.routes import health
from gateway.core.config import GatewayConfig


@pytest.mark.asyncio
async def test_platform_reachability_uses_default_health_path_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> httpx.Response:
            captured["url"] = url
            return httpx.Response(204)

    monkeypatch.setattr(health.httpx, "AsyncClient", FakeAsyncClient)

    assert await health._check_platform_reachability(
        GatewayConfig(platform={"base_url": "https://platform.local/api/v1/"})
    )
    assert captured == {
        "timeout": 5,
        "url": "https://platform.local/api/v1/utils/health-check/",
    }


@pytest.mark.asyncio
async def test_platform_reachability_uses_configured_health_path_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> httpx.Response:
            captured["url"] = url
            return httpx.Response(200)

    monkeypatch.setattr(health.httpx, "AsyncClient", FakeAsyncClient)

    assert await health._check_platform_reachability(
        GatewayConfig(
            platform={
                "base_url": "https://platform.local/api/v1/",
                "health_path": "/healthz",
                "resolve_timeout_ms": "1500",
            }
        )
    )
    assert captured == {
        "timeout": 1.5,
        "url": "https://platform.local/api/v1/healthz",
    }


@pytest.mark.asyncio
async def test_platform_reachability_returns_false_without_base_url() -> None:
    assert await health._check_platform_reachability(GatewayConfig(platform={})) is False

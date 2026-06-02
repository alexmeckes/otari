import httpx
import pytest
from any_llm.types.completion import CompletionUsage

from gateway.core.config import PLATFORM_TOKEN_ENV_VARS, GatewayConfig
from gateway.services import platform_gateway
from gateway.services.platform_gateway import (
    _platform_gateway_headers,
    report_platform_usage,
)


async def _reported_usage_body(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    usage: CompletionUsage | None,
    *,
    error_class: str | None = None,
) -> dict[str, object]:
    bodies: list[dict[str, object]] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
        timeout_seconds: float,
    ) -> httpx.Response:
        bodies.append(body)
        return httpx.Response(204)

    monkeypatch.setattr(platform_gateway, "_post_platform", fake_post_platform)
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw-test-token")

    await report_platform_usage(
        GatewayConfig(platform={"base_url": "https://platform.local/api/v1"}),
        "corr-1",
        outcome,
        usage,
        error_class=error_class,
    )

    assert len(bodies) == 1
    return bodies[0]


async def _reported_usage_calls(
    monkeypatch: pytest.MonkeyPatch,
    config: GatewayConfig,
    *,
    status_code: int = 204,
) -> tuple[list[dict[str, object]], list[float]]:
    calls: list[dict[str, object]] = []
    sleeps: list[float] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
        timeout_seconds: float,
    ) -> httpx.Response:
        calls.append(
            {
                "url": url,
                "headers": headers,
                "body": body,
                "timeout_seconds": timeout_seconds,
            }
        )
        return httpx.Response(status_code)

    async def fake_sleep(delay_seconds: float) -> None:
        sleeps.append(delay_seconds)

    monkeypatch.setattr(platform_gateway, "_post_platform", fake_post_platform)
    monkeypatch.setattr(platform_gateway.asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw-test-token")

    await report_platform_usage(
        config,
        "corr-1",
        "success",
        CompletionUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )

    return calls, sleeps


@pytest.mark.asyncio
async def test_report_platform_usage_includes_success_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    body = await _reported_usage_body(
        monkeypatch,
        "success",
        CompletionUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )

    assert body == {
        "correlation_id": "corr-1",
        "status": "success",
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
    }


@pytest.mark.asyncio
async def test_report_platform_usage_defaults_missing_success_usage_to_zero_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = await _reported_usage_body(monkeypatch, "success", None)

    assert body == {
        "correlation_id": "corr-1",
        "status": "success",
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@pytest.mark.asyncio
async def test_report_platform_usage_includes_error_class_for_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    body = await _reported_usage_body(monkeypatch, "error", None, error_class="http_503")

    assert body == {
        "correlation_id": "corr-1",
        "status": "error",
        "error_class": "http_503",
    }


def test_platform_gateway_headers_include_gateway_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw-test-token")

    assert _platform_gateway_headers(GatewayConfig()) == {"X-Gateway-Token": "gw-test-token"}


def test_platform_gateway_headers_default_missing_token_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in PLATFORM_TOKEN_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    assert _platform_gateway_headers(GatewayConfig()) == {"X-Gateway-Token": ""}


@pytest.mark.asyncio
async def test_report_platform_usage_uses_default_request_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, sleeps = await _reported_usage_calls(
        monkeypatch,
        GatewayConfig(platform={"base_url": "https://platform.local/api/v1/"}),
        status_code=500,
    )

    assert len(calls) == 3
    assert calls[0]["url"] == "https://platform.local/api/v1/gateway/usage"
    assert calls[0]["headers"] == {"X-Gateway-Token": "gw-test-token"}
    assert calls[0]["timeout_seconds"] == 5
    assert sleeps == [0.25, 0.5]


@pytest.mark.asyncio
async def test_report_platform_usage_uses_configured_request_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, sleeps = await _reported_usage_calls(
        monkeypatch,
        GatewayConfig(
            platform={
                "base_url": "https://platform.local/api/v1",
                "usage_timeout_ms": 2500,
                "usage_max_retries": 7,
            }
        ),
        status_code=500,
    )

    assert len(calls) == 7
    assert calls[0]["url"] == "https://platform.local/api/v1/gateway/usage"
    assert calls[0]["headers"] == {"X-Gateway-Token": "gw-test-token"}
    assert calls[0]["timeout_seconds"] == 2.5
    assert sleeps == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


@pytest.mark.asyncio
async def test_report_platform_usage_returns_without_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, sleeps = await _reported_usage_calls(monkeypatch, GatewayConfig(platform={}))

    assert calls == []
    assert sleeps == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_attempts"),
    [
        (204, 1),
        (401, 1),
        (404, 1),
        (409, 1),
        (422, 1),
        (400, 1),
        (429, 1),
        (500, 3),
        (503, 3),
    ],
)
async def test_report_platform_usage_retries_only_server_statuses(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected_attempts: int,
) -> None:
    calls, sleeps = await _reported_usage_calls(
        monkeypatch,
        GatewayConfig(platform={"base_url": "https://platform.local/api/v1", "usage_max_retries": 3}),
        status_code=status_code,
    )

    assert len(calls) == expected_attempts
    expected_sleeps = [0.25, 0.5] if expected_attempts == 3 else []
    assert sleeps == expected_sleeps

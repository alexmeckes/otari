import httpx
import pytest
from any_llm.types.completion import CompletionUsage

from gateway.core.config import PLATFORM_TOKEN_ENV_VARS, GatewayConfig
from gateway.services import platform_gateway
from gateway.services.platform_gateway import (
    _platform_gateway_headers,
    _platform_usage_payload,
    _platform_usage_report_request,
    report_platform_usage,
)


def test_platform_usage_payload_includes_success_usage() -> None:
    payload = _platform_usage_payload(
        "corr-1",
        "success",
        CompletionUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )

    assert payload == {
        "correlation_id": "corr-1",
        "status": "success",
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
    }


def test_platform_usage_payload_defaults_missing_success_usage_to_zero_tokens() -> None:
    payload = _platform_usage_payload("corr-1", "success", None)

    assert payload == {
        "correlation_id": "corr-1",
        "status": "success",
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def test_platform_usage_payload_includes_error_class_for_errors() -> None:
    payload = _platform_usage_payload("corr-1", "error", None, error_class="http_503")

    assert payload == {
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


def test_platform_usage_report_request_uses_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw-test-token")

    usage_request = _platform_usage_report_request(
        GatewayConfig(platform={"base_url": "https://platform.local/api/v1/"})
    )

    assert usage_request is not None
    assert usage_request.url == "https://platform.local/api/v1/gateway/usage"
    assert usage_request.headers == {"X-Gateway-Token": "gw-test-token"}
    assert usage_request.timeout_seconds == 5
    assert usage_request.max_retries == 3


def test_platform_usage_report_request_uses_configured_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw-test-token")

    usage_request = _platform_usage_report_request(
        GatewayConfig(
            platform={
                "base_url": "https://platform.local/api/v1",
                "usage_timeout_ms": 2500,
                "usage_max_retries": 7,
            }
        )
    )

    assert usage_request is not None
    assert usage_request.url == "https://platform.local/api/v1/gateway/usage"
    assert usage_request.headers == {"X-Gateway-Token": "gw-test-token"}
    assert usage_request.timeout_seconds == 2.5
    assert usage_request.max_retries == 7


def test_platform_usage_report_request_returns_none_without_base_url() -> None:
    assert _platform_usage_report_request(GatewayConfig(platform={})) is None


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
        GatewayConfig(platform={"base_url": "https://platform.local/api/v1", "usage_max_retries": 3}),
        "corr-1",
        "success",
        CompletionUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )

    assert len(calls) == expected_attempts
    expected_sleeps = [0.25, 0.5] if expected_attempts == 3 else []
    assert sleeps == expected_sleeps

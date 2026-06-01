import pytest
from any_llm.types.completion import CompletionUsage

from gateway.core.config import PLATFORM_TOKEN_ENV_VARS, GatewayConfig
from gateway.services.platform_gateway import (
    _platform_gateway_headers,
    _platform_usage_payload,
    _platform_usage_report_request,
    _should_retry_usage_report_status,
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


def test_platform_usage_retry_status_accepts_no_content_without_retry() -> None:
    assert _should_retry_usage_report_status(204) is False


def test_platform_usage_retry_status_does_not_retry_non_retryable_statuses() -> None:
    for status_code in (401, 404, 409, 422):
        assert _should_retry_usage_report_status(status_code) is False


def test_platform_usage_retry_status_retries_server_errors() -> None:
    for status_code in (500, 503):
        assert _should_retry_usage_report_status(status_code) is True


def test_platform_usage_retry_status_does_not_retry_other_non_server_errors() -> None:
    for status_code in (400, 429):
        assert _should_retry_usage_report_status(status_code) is False

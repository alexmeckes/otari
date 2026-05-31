import httpx
import pytest
from fastapi import status

from gateway.api.routes._chat_platform_errors import platform_attempt_failure_exception


@pytest.mark.parametrize(
    ("attempts_count", "expected_detail"),
    [
        (1, "LLM provider timeout"),
        (2, "All upstream providers timed out"),
    ],
)
def test_platform_attempt_failure_exception_maps_timeouts(attempts_count: int, expected_detail: str) -> None:
    exc = platform_attempt_failure_exception(httpx.ReadTimeout("provider timed out"), attempts_count=attempts_count)

    assert exc.status_code == status.HTTP_504_GATEWAY_TIMEOUT
    assert exc.detail == expected_detail


@pytest.mark.parametrize(
    ("attempts_count", "expected_detail"),
    [
        (1, "LLM provider error"),
        (2, "All upstream providers failed"),
    ],
)
def test_platform_attempt_failure_exception_maps_provider_errors(
    attempts_count: int,
    expected_detail: str,
) -> None:
    exc = platform_attempt_failure_exception(RuntimeError("provider failed"), attempts_count=attempts_count)

    assert exc.status_code == status.HTTP_502_BAD_GATEWAY
    assert exc.detail == expected_detail


def test_platform_attempt_failure_exception_treats_missing_cause_as_provider_error() -> None:
    exc = platform_attempt_failure_exception(None, attempts_count=2)

    assert exc.status_code == status.HTTP_502_BAD_GATEWAY
    assert exc.detail == "All upstream providers failed"

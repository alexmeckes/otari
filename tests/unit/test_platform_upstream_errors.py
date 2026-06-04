import httpx
import pytest

from gateway.services.platform_gateway import classify_upstream_error


class _StatusCodeError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _ResponseStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"response {status_code}")
        self.response = httpx.Response(status_code)


def test_classify_upstream_error_treats_timeouts_as_retryable() -> None:
    assert classify_upstream_error(httpx.ReadTimeout("timed out")) == (True, "timeout")


def test_classify_upstream_error_treats_network_errors_as_retryable() -> None:
    assert classify_upstream_error(httpx.ConnectError("connection failed")) == (True, "conn_err")


@pytest.mark.parametrize("status_code", [401, 403, 408, 429, 500, 502, 503, 504, 599])
def test_classify_upstream_error_treats_retryable_status_codes_as_retryable(status_code: int) -> None:
    assert classify_upstream_error(_StatusCodeError(status_code)) == (True, f"http_{status_code}")


def test_classify_upstream_error_uses_exception_status_code() -> None:
    assert classify_upstream_error(_StatusCodeError(429)) == (True, "http_429")
    assert classify_upstream_error(_StatusCodeError(400)) == (False, "http_400")


def test_classify_upstream_error_uses_response_status_code() -> None:
    assert classify_upstream_error(_ResponseStatusError(503)) == (True, "http_503")
    assert classify_upstream_error(_ResponseStatusError(422)) == (False, "http_422")


def test_classify_upstream_error_treats_unknown_errors_as_non_retryable() -> None:
    assert classify_upstream_error(RuntimeError("provider failed")) == (False, "unknown")

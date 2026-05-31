import httpx
from fastapi import status

from gateway.api.routes._chat_standalone_errors import standalone_provider_failure_exception


def test_standalone_provider_failure_exception_maps_timeouts() -> None:
    exc = standalone_provider_failure_exception(httpx.ConnectTimeout("provider timed out"))

    assert exc.status_code == status.HTTP_504_GATEWAY_TIMEOUT
    assert exc.detail == "LLM provider timeout"


def test_standalone_provider_failure_exception_maps_provider_errors() -> None:
    exc = standalone_provider_failure_exception(RuntimeError("provider failed"))

    assert exc.status_code == status.HTTP_502_BAD_GATEWAY
    assert exc.detail == "LLM provider error"

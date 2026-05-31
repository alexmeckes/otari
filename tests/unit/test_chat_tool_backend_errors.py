from fastapi import status

from gateway.api.routes._chat_tool_backend_errors import (
    chat_tool_backend_failure,
    chat_tool_backend_failure_exception,
)
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError


def test_chat_tool_backend_failure_maps_sandbox_unreachable() -> None:
    failure = chat_tool_backend_failure(SandboxNotReachableError("sandbox failed"))

    assert failure.status_code == status.HTTP_502_BAD_GATEWAY
    assert failure.detail == "code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL"
    assert failure.error_class == "sandbox_unreachable"


def test_chat_tool_backend_failure_maps_web_search_unreachable() -> None:
    failure = chat_tool_backend_failure(WebSearchNotReachableError("web_search failed"))

    assert failure.status_code == status.HTTP_502_BAD_GATEWAY
    assert failure.detail == "web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL"
    assert failure.error_class == "web_search_unreachable"


def test_chat_tool_backend_failure_exception_uses_shared_metadata() -> None:
    exc = chat_tool_backend_failure_exception(SandboxNotReachableError("sandbox failed"))

    assert exc.status_code == status.HTTP_502_BAD_GATEWAY
    assert exc.detail == "code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL"

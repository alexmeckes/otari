from dataclasses import dataclass

from fastapi import HTTPException, status

from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError


@dataclass(frozen=True)
class ChatToolBackendFailure:
    status_code: int
    detail: str
    error_class: str


def chat_tool_backend_failure(exc: SandboxNotReachableError | WebSearchNotReachableError) -> ChatToolBackendFailure:
    if isinstance(exc, SandboxNotReachableError):
        return ChatToolBackendFailure(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
            error_class="sandbox_unreachable",
        )
    return ChatToolBackendFailure(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
        error_class="web_search_unreachable",
    )


def chat_tool_backend_failure_exception(
    exc: SandboxNotReachableError | WebSearchNotReachableError,
) -> HTTPException:
    failure = chat_tool_backend_failure(exc)
    return HTTPException(
        status_code=failure.status_code,
        detail=failure.detail,
    )

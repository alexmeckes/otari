from fastapi import HTTPException, status

from gateway.api.routes._chat_provider_errors import is_provider_timeout


def standalone_provider_failure_exception(exc: BaseException) -> HTTPException:
    """Map standalone provider failures to the public HTTP response."""
    if is_provider_timeout(exc):
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM provider timeout",
        )
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="LLM provider error",
    )

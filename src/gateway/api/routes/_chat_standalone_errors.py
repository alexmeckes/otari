import asyncio

import httpx
from fastapi import HTTPException, status


def standalone_provider_failure_exception(exc: BaseException) -> HTTPException:
    """Map standalone provider failures to the public HTTP response."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM provider timeout",
        )
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="LLM provider error",
    )

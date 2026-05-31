import asyncio

import httpx
from fastapi import HTTPException, status


def platform_attempt_failure_exception(exc: BaseException | None, *, attempts_count: int) -> HTTPException:
    """Map platform chat attempt failures to the public HTTP response."""
    is_single_attempt = attempts_count <= 1
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        detail = "LLM provider timeout" if is_single_attempt else "All upstream providers timed out"
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=detail,
        )

    detail = "LLM provider error" if is_single_attempt else "All upstream providers failed"
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=detail,
    )

import asyncio

import httpx


def is_provider_timeout(exc: BaseException | None) -> bool:
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException))

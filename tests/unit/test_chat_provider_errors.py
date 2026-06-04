import asyncio

import httpx

from gateway.api.routes._chat_provider_errors import is_provider_timeout


def test_is_provider_timeout_accepts_asyncio_timeout() -> None:
    assert is_provider_timeout(asyncio.TimeoutError())


def test_is_provider_timeout_accepts_builtin_timeout() -> None:
    assert is_provider_timeout(TimeoutError())


def test_is_provider_timeout_accepts_httpx_timeout() -> None:
    assert is_provider_timeout(httpx.ReadTimeout("timed out"))


def test_is_provider_timeout_rejects_missing_and_non_timeout_errors() -> None:
    assert not is_provider_timeout(None)
    assert not is_provider_timeout(RuntimeError("failed"))

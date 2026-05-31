from dataclasses import dataclass, field

import pytest
from fastapi import HTTPException, Response

from gateway.api.routes._provider_context import OpenAIProviderRequestContext
from gateway.models.entities import UsageLog
from gateway.rate_limit import RateLimitInfo


@dataclass
class StubLogWriter:
    logs: list[UsageLog] = field(default_factory=list)

    async def put(self, log: UsageLog) -> None:
        self.logs.append(log)


def _context(rate_limit_info: RateLimitInfo | None = None) -> OpenAIProviderRequestContext:
    return OpenAIProviderRequestContext(
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=rate_limit_info,
        provider="openai",
        model="gpt-4o-mini",
        provider_kwargs={"api_key": "sk-test"},
    )


def test_provider_context_call_kwargs_includes_provider_defaults() -> None:
    call_kwargs = _context().call_kwargs(input="hello")

    assert call_kwargs == {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "input": "hello",
        "api_key": "sk-test",
    }


def test_provider_context_rate_limit_headers_are_empty_when_disabled() -> None:
    response = Response()
    context = _context()

    assert context.rate_limit_headers() == {}
    context.apply_rate_limit_headers(response)
    assert not any(header.startswith("x-ratelimit") for header in response.headers)


def test_provider_context_rate_limit_headers_include_limit_state() -> None:
    response = Response()
    context = _context(RateLimitInfo(limit=10, remaining=7, reset=123.4))

    assert context.rate_limit_headers() == {
        "X-RateLimit-Limit": "10",
        "X-RateLimit-Remaining": "7",
        "X-RateLimit-Reset": "123",
    }
    context.apply_rate_limit_headers(response)
    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Remaining"] == "7"
    assert response.headers["X-RateLimit-Reset"] == "123"


def test_provider_context_usage_log_includes_identity_fields() -> None:
    usage_log = _context().usage_log(
        endpoint="/v1/test",
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
        tags={"kind": "unit"},
    )

    assert usage_log.api_key_id == "key-1"
    assert usage_log.user_id == "user-1"
    assert usage_log.model == "gpt-4o-mini"
    assert usage_log.provider == "openai"
    assert usage_log.endpoint == "/v1/test"
    assert usage_log.prompt_tokens == 1
    assert usage_log.completion_tokens == 2
    assert usage_log.total_tokens == 3
    assert usage_log.tags == {"kind": "unit"}


@pytest.mark.asyncio
async def test_provider_context_log_usage_error_writes_identity_fields() -> None:
    writer = StubLogWriter()

    await _context().log_usage_error(writer, endpoint="/v1/test", error=RuntimeError("provider down"))

    assert len(writer.logs) == 1
    log = writer.logs[0]
    assert log.api_key_id == "key-1"
    assert log.user_id == "user-1"
    assert log.model == "gpt-4o-mini"
    assert log.provider == "openai"
    assert log.endpoint == "/v1/test"
    assert log.status == "error"
    assert log.error_message == "provider down"


@pytest.mark.asyncio
async def test_provider_context_log_and_raise_provider_error_logs_then_raises() -> None:
    writer = StubLogWriter()

    with pytest.raises(HTTPException) as exc_info:
        await _context().log_and_raise_provider_error(
            writer,
            endpoint="/v1/test",
            error=RuntimeError("provider down"),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "The request could not be completed by the provider"
    assert len(writer.logs) == 1
    assert writer.logs[0].status == "error"
    assert writer.logs[0].error_message == "provider down"

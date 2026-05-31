from typing import Any

import pytest
from any_llm import AnyLLM
from any_llm.types.completion import CompletionUsage
from fastapi import Response

from gateway.api.routes._provider_context import OpenAIProviderRequestContext
from gateway.api.routes._responses_native import (
    log_native_response_usage,
    native_response_call_kwargs,
    native_response_payload,
    native_response_streaming_response,
)
from gateway.api.routes._responses_transform import ResponsesRequest
from gateway.rate_limit import RateLimitInfo


def _context(rate_limit_info: RateLimitInfo | None = None) -> OpenAIProviderRequestContext:
    provider, model = AnyLLM.split_model_provider("openai:gpt-4o-mini")
    return OpenAIProviderRequestContext(
        api_key_id="key-1",
        user_id="resolved-user",
        rate_limit_info=rate_limit_info,
        provider=provider,
        model=model,
        provider_kwargs={"api_key": "sk-test", "timeout": 30},
    )


def test_native_response_call_kwargs_uses_provider_context() -> None:
    context = _context()
    request_body = ResponsesRequest(
        model="openai:gpt-4o",
        input="Hello",
        stream=True,
        user="request-user",
        project_id="proj-1",
        tags={"team": "platform"},
        temperature=0.2,
        metadata={"trace": "native"},
    )

    call_kwargs, stream = native_response_call_kwargs(request_body, context)

    assert stream is True
    assert call_kwargs == {
        "api_key": "sk-test",
        "timeout": 30,
        "temperature": 0.2,
        "metadata": {"trace": "native"},
        "user": "resolved-user",
        "model": "gpt-4o-mini",
        "provider": context.provider,
        "input_data": "Hello",
    }


@pytest.mark.asyncio
async def test_log_native_response_usage_uses_provider_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_log_usage(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("gateway.api.routes._responses_native.log_usage", fake_log_usage)

    usage = CompletionUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7)
    request_body = ResponsesRequest(
        model="openai:gpt-4o-mini",
        input="Hello",
        user="request-user",
        project_id="proj-1",
        tags={"surface": "responses"},
    )

    await log_native_response_usage(
        db=object(),  # type: ignore[arg-type]
        log_writer=object(),  # type: ignore[arg-type]
        context=_context(),
        request_body=request_body,
        usage_data=usage,
    )

    assert captured["api_key_id"] == "key-1"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["provider"] == _context().provider
    assert captured["endpoint"] == "/v1/responses"
    assert captured["user_id"] == "resolved-user"
    assert captured["project_id"] == "proj-1"
    assert captured["tags"] == {"surface": "responses"}
    assert captured["usage_override"] == usage
    assert captured["error"] is None


def test_native_response_streaming_response_uses_context_headers() -> None:
    response = native_response_streaming_response(
        stream_result=object(),
        db=object(),  # type: ignore[arg-type]
        log_writer=object(),  # type: ignore[arg-type]
        context=_context(RateLimitInfo(limit=10, remaining=8, reset=123.4)),
        request_body=ResponsesRequest(model="openai:gpt-4o-mini", input="Hello"),
    )

    assert response.headers["X-Response-Model"] == "openai/gpt-4o-mini"
    assert response.headers["X-Response-Vendor"] == "openai"
    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Remaining"] == "8"
    assert response.headers["X-RateLimit-Reset"] == "123"


def test_native_response_payload_applies_headers_and_served_metadata() -> None:
    class FakeResult:
        def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
            assert exclude_none is True
            return {
                "id": "resp_native",
                "model": "gpt-4o-mini",
                "output": [],
            }

    response = Response()

    payload = native_response_payload(
        result=FakeResult(),
        response=response,
        context=_context(RateLimitInfo(limit=10, remaining=7, reset=123.4)),
    )

    assert payload == {
        "id": "resp_native",
        "model": "openai/gpt-4o-mini",
        "output": [],
        "vendor": "openai",
    }
    assert response.headers["X-Response-Model"] == "openai/gpt-4o-mini"
    assert response.headers["X-Response-Vendor"] == "openai"
    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Remaining"] == "7"
    assert response.headers["X-RateLimit-Reset"] == "123"

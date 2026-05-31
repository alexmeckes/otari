from types import SimpleNamespace
from typing import Any, cast

import pytest
from any_llm.types.completion import CompletionUsage
from any_llm.types.messages import (
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageUsage,
    TextBlock,
)
from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes import messages
from gateway.api.routes._message_models import MessagesRequest
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey
from gateway.rate_limit import RateLimitInfo
from gateway.services.log_writer import LogWriter


def _messages_request(metadata: dict[str, Any] | None = None) -> MessagesRequest:
    return MessagesRequest(
        model="anthropic:claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        metadata=metadata,
    )


def _api_key(*, user_id: str | None = "api-user") -> APIKey:
    return cast(APIKey, SimpleNamespace(id="key-1", user_id=user_id))


def _message_response() -> MessageResponse:
    return MessageResponse(
        id="msg_test123",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="Hello!")],
        model="claude-3-5-sonnet",
        stop_reason="end_turn",
        usage=MessageUsage(input_tokens=10, output_tokens=5),
    )


def _provider_call_context() -> messages.MessageProviderCallContext:
    return messages.MessageProviderCallContext(
        provider="anthropic",
        model="claude-3-5-sonnet",
        call_kwargs={},
    )


def _execution_context(
    *,
    message_context: messages.MessageRequestContext | None = None,
    provider_call_context: messages.MessageProviderCallContext | None = None,
) -> messages.MessageExecutionContext:
    return messages.MessageExecutionContext(
        message_context=message_context or messages.MessageRequestContext(api_key_id="key-1", user_id="user-1"),
        rate_limit_info=RateLimitInfo(limit=10, remaining=8, reset=123.4),
        provider_call_context=provider_call_context or _provider_call_context(),
    )


def test_resolve_message_request_context_uses_master_key_metadata_user() -> None:
    context = messages._resolve_message_request_context(
        _messages_request({"user_id": "metadata-user"}),
        (None, True),
    )

    assert context.api_key_id is None
    assert context.user_id == "metadata-user"


def test_resolve_message_request_context_uses_api_key_fallback() -> None:
    context = messages._resolve_message_request_context(
        _messages_request(),
        (_api_key(), False),
    )

    assert context.api_key_id == "key-1"
    assert context.user_id == "api-user"


def test_resolve_message_request_context_metadata_overrides_api_key_user() -> None:
    context = messages._resolve_message_request_context(
        _messages_request({"user_id": "metadata-user"}),
        (_api_key(), False),
    )

    assert context.api_key_id == "key-1"
    assert context.user_id == "metadata-user"


def test_resolve_message_request_context_preserves_master_key_error_shape() -> None:
    with pytest.raises(HTTPException) as exc_info:
        messages._resolve_message_request_context(_messages_request(), (None, True))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "When using master key, 'metadata.user_id' is required in request body",
        },
    }


def test_message_provider_call_context_preserves_request_field_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[GatewayConfig, Any]] = []
    config = cast(GatewayConfig, object())
    request = _messages_request()
    request.temperature = 0.2

    def fake_get_provider_kwargs(config_arg: GatewayConfig, provider: Any) -> dict[str, Any]:
        calls.append((config_arg, provider))
        return {
            "api_key": "sk-test",
            "model": "provider-default",
            "temperature": 0.9,
            "top_p": 0.4,
        }

    monkeypatch.setattr(messages, "get_provider_kwargs", fake_get_provider_kwargs)

    context = messages._message_provider_call_context(request, config)

    assert calls == [(config, context.provider)]
    assert context.model == "claude-3-5-sonnet"
    assert context.call_kwargs["api_key"] == "sk-test"
    assert context.call_kwargs["model"] == "anthropic:claude-3-5-sonnet"
    assert context.call_kwargs["temperature"] == 0.2
    assert context.call_kwargs["top_p"] == 0.4
    assert context.call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert context.call_kwargs["max_tokens"] == 1024


def test_message_call_kwargs_adds_stream_without_mutating_context() -> None:
    context = messages.MessageProviderCallContext(
        provider="anthropic",
        model="claude-3-5-sonnet",
        call_kwargs={"model": "anthropic:claude-3-5-sonnet", "api_key": "sk-test"},
    )

    call_kwargs = messages._message_call_kwargs(context, stream=True)

    assert call_kwargs == {
        "model": "anthropic:claude-3-5-sonnet",
        "api_key": "sk-test",
        "stream": True,
    }
    assert context.call_kwargs == {"model": "anthropic:claude-3-5-sonnet", "api_key": "sk-test"}


def test_message_call_kwargs_preserves_explicit_non_streaming_flag() -> None:
    context = messages.MessageProviderCallContext(
        provider="anthropic",
        model="claude-3-5-sonnet",
        call_kwargs={"model": "anthropic:claude-3-5-sonnet", "stream": False},
    )

    assert messages._message_call_kwargs(context, stream=False) == {
        "model": "anthropic:claude-3-5-sonnet",
        "stream": False,
    }


def test_message_execution_context_formats_label_and_rate_limit_headers() -> None:
    context = _execution_context()
    response = Response()

    assert context.provider_label == "anthropic:claude-3-5-sonnet"
    assert context.rate_limit_headers() == {
        "X-RateLimit-Limit": "10",
        "X-RateLimit-Remaining": "8",
        "X-RateLimit-Reset": "123",
    }

    context.apply_rate_limit_headers(response)

    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Remaining"] == "8"
    assert response.headers["X-RateLimit-Reset"] == "123"


def test_message_execution_context_ignores_missing_rate_limit_info() -> None:
    context = messages.MessageExecutionContext(
        message_context=messages.MessageRequestContext(api_key_id="key-1", user_id="user-1"),
        rate_limit_info=None,
        provider_call_context=_provider_call_context(),
    )
    response = Response()

    assert context.rate_limit_headers() == {}

    context.apply_rate_limit_headers(response)

    assert "X-RateLimit-Limit" not in response.headers


@pytest.mark.asyncio
async def test_message_execution_context_resolves_rate_limit_budget_and_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []
    raw_request = cast(Request, object())
    db = cast(AsyncSession, object())
    config = cast(GatewayConfig, SimpleNamespace(budget_strategy="fail_closed"))
    request = _messages_request({"user_id": "metadata-user"})
    rate_limit_info = RateLimitInfo(limit=10, remaining=7, reset=123.4)

    def fake_check_rate_limit(raw_request_arg: Request, user_id: str) -> RateLimitInfo:
        calls.append(("rate_limit", raw_request_arg, user_id))
        return rate_limit_info

    async def fake_validate_user_request_budget(
        db_arg: AsyncSession,
        user_id: str,
        model: str,
        *,
        strategy: str,
    ) -> None:
        calls.append(("budget", db_arg, user_id, model, strategy))

    def fake_get_provider_kwargs(config_arg: GatewayConfig, provider: Any) -> dict[str, Any]:
        calls.append(("provider_kwargs", config_arg, provider))
        return {"api_key": "sk-test"}

    monkeypatch.setattr(messages, "check_rate_limit", fake_check_rate_limit)
    monkeypatch.setattr(messages, "validate_user_request_budget", fake_validate_user_request_budget)
    monkeypatch.setattr(messages, "get_provider_kwargs", fake_get_provider_kwargs)

    context = await messages._message_execution_context(
        raw_request=raw_request,
        request=request,
        auth_result=(_api_key(), False),
        db=db,
        config=config,
    )

    assert context.message_context == messages.MessageRequestContext(api_key_id="key-1", user_id="metadata-user")
    assert context.rate_limit_info is rate_limit_info
    assert context.provider_call_context.provider == "anthropic"
    assert context.provider_call_context.model == "claude-3-5-sonnet"
    assert context.provider_call_context.call_kwargs["api_key"] == "sk-test"
    assert context.provider_call_context.call_kwargs["model"] == "anthropic:claude-3-5-sonnet"
    assert calls == [
        ("rate_limit", raw_request, "metadata-user"),
        ("budget", db, "metadata-user", "anthropic:claude-3-5-sonnet", "fail_closed"),
        ("provider_kwargs", config, context.provider_call_context.provider),
    ]


@pytest.mark.asyncio
async def test_message_provider_response_calls_non_streaming_provider_and_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    response = Response()
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())
    provider_context = messages.MessageProviderCallContext(
        provider="anthropic",
        model="claude-3-5-sonnet",
        call_kwargs={"model": "anthropic:claude-3-5-sonnet", "api_key": "sk-test"},
    )
    provider_result = _message_response()
    payload = {"ok": True}
    execution_context = _execution_context(provider_call_context=provider_context)

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        calls.append(("amessages", kwargs))
        return provider_result

    async def fake_message_response_payload(**kwargs: Any) -> dict[str, Any]:
        calls.append(("payload", kwargs))
        return payload

    monkeypatch.setattr(messages, "amessages", fake_amessages)
    monkeypatch.setattr(messages, "_message_response_payload", fake_message_response_payload)

    result = await messages._message_provider_response(
        request=_messages_request(),
        response=response,
        db=db,
        log_writer=log_writer,
        execution_context=execution_context,
    )

    assert result == payload
    assert calls == [
        (
            "amessages",
            {"model": "anthropic:claude-3-5-sonnet", "api_key": "sk-test"},
        ),
        (
            "payload",
            {
                "result": provider_result,
                "response": response,
                "db": db,
                "log_writer": log_writer,
                "execution_context": execution_context,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_message_provider_response_calls_streaming_provider_and_response_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    response = Response()
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())
    provider_context = messages.MessageProviderCallContext(
        provider="anthropic",
        model="claude-3-5-sonnet",
        call_kwargs={"model": "anthropic:claude-3-5-sonnet"},
    )
    request = _messages_request()
    request.stream = True
    stream_result = object()
    streaming_response = cast(StreamingResponse, object())
    execution_context = _execution_context(provider_call_context=provider_context)

    async def fake_amessages(**kwargs: Any) -> object:
        calls.append(("amessages", kwargs))
        return stream_result

    def fake_message_streaming_response(**kwargs: Any) -> StreamingResponse:
        calls.append(("streaming_response", kwargs))
        return streaming_response

    monkeypatch.setattr(messages, "amessages", fake_amessages)
    monkeypatch.setattr(messages, "_message_streaming_response", fake_message_streaming_response)

    result = await messages._message_provider_response(
        request=request,
        response=response,
        db=db,
        log_writer=log_writer,
        execution_context=execution_context,
    )

    assert result is streaming_response
    assert provider_context.call_kwargs == {"model": "anthropic:claude-3-5-sonnet"}
    assert calls == [
        (
            "amessages",
            {"model": "anthropic:claude-3-5-sonnet", "stream": True},
        ),
        (
            "streaming_response",
            {
                "stream_result": stream_result,
                "db": db,
                "log_writer": log_writer,
                "execution_context": execution_context,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_log_message_usage_forwards_success_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    async def fake_log_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(messages, "log_usage", fake_log_usage)

    await messages._log_message_usage(
        db=db,
        log_writer=log_writer,
        execution_context=_execution_context(),
        usage_data=usage,
    )

    assert calls == [
        {
            "db": db,
            "log_writer": log_writer,
            "api_key_id": "key-1",
            "model": "claude-3-5-sonnet",
            "provider": "anthropic",
            "endpoint": "/v1/messages",
            "user_id": "user-1",
            "usage_override": usage,
            "error": None,
        }
    ]


@pytest.mark.asyncio
async def test_message_response_payload_logs_usage_sets_headers_and_serializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())
    response = Response()

    async def fake_log_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(messages, "log_usage", fake_log_usage)

    payload = await messages._message_response_payload(
        result=_message_response(),
        response=response,
        db=db,
        log_writer=log_writer,
        execution_context=_execution_context(),
    )

    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Remaining"] == "8"
    assert response.headers["X-RateLimit-Reset"] == "123"
    assert payload == {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-3-5-sonnet",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    assert calls == [
        {
            "db": db,
            "log_writer": log_writer,
            "api_key_id": "key-1",
            "model": "claude-3-5-sonnet",
            "provider": "anthropic",
            "endpoint": "/v1/messages",
            "user_id": "user-1",
            "usage_override": CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            "error": None,
        }
    ]


@pytest.mark.asyncio
async def test_log_message_usage_forwards_error_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())

    async def fake_log_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(messages, "log_usage", fake_log_usage)

    await messages._log_message_usage(
        db=db,
        log_writer=log_writer,
        execution_context=_execution_context(
            message_context=messages.MessageRequestContext(api_key_id=None, user_id="user-1"),
        ),
        error="provider down",
    )

    assert calls == [
        {
            "db": db,
            "log_writer": log_writer,
            "api_key_id": None,
            "model": "claude-3-5-sonnet",
            "provider": "anthropic",
            "endpoint": "/v1/messages",
            "user_id": "user-1",
            "usage_override": None,
            "error": "provider down",
        }
    ]


@pytest.mark.asyncio
async def test_log_and_raise_message_provider_error_preserves_logging_and_error_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_calls: list[dict[str, Any]] = []
    logger_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())
    provider_error = RuntimeError("provider down")

    async def fake_log_usage(**kwargs: Any) -> None:
        usage_calls.append(kwargs)

    def fake_logger_error(*args: Any, **kwargs: Any) -> None:
        logger_calls.append((args, kwargs))

    monkeypatch.setattr(messages, "log_usage", fake_log_usage)
    monkeypatch.setattr(messages.logger, "error", fake_logger_error)

    with pytest.raises(HTTPException) as exc_info:
        await messages._log_and_raise_message_provider_error(
            db=db,
            log_writer=log_writer,
            execution_context=_execution_context(),
            error=provider_error,
        )

    assert exc_info.value.__cause__ is provider_error
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": "The request could not be completed by the provider",
        },
    }
    assert usage_calls == [
        {
            "db": db,
            "log_writer": log_writer,
            "api_key_id": "key-1",
            "model": "claude-3-5-sonnet",
            "provider": "anthropic",
            "endpoint": "/v1/messages",
            "user_id": "user-1",
            "usage_override": None,
            "error": "provider down",
        }
    ]
    assert logger_calls == [
        (
            ("Provider call failed for %s:%s: %s", "anthropic", "claude-3-5-sonnet", provider_error),
            {},
        )
    ]


@pytest.mark.asyncio
async def test_message_streaming_response_logs_usage_and_sets_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())

    async def fake_log_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    async def fake_stream() -> Any:
        yield MessageDeltaEvent(
            type="message_delta",
            delta=MessageDelta(),
            usage=MessageDeltaUsage(input_tokens=2, output_tokens=5),
        )

    monkeypatch.setattr(messages, "log_usage", fake_log_usage)

    response = messages._message_streaming_response(
        stream_result=fake_stream(),
        db=db,
        log_writer=log_writer,
        execution_context=_execution_context(),
    )

    chunks = [chunk async for chunk in response.body_iterator]

    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Remaining"] == "8"
    assert response.headers["X-RateLimit-Reset"] == "123"
    assert chunks[-1] == "event: done\ndata: {}\n\n"
    assert calls == [
        {
            "db": db,
            "log_writer": log_writer,
            "api_key_id": "key-1",
            "model": "claude-3-5-sonnet",
            "provider": "anthropic",
            "endpoint": "/v1/messages",
            "user_id": "user-1",
            "usage_override": CompletionUsage(prompt_tokens=2, completion_tokens=5, total_tokens=7),
            "error": None,
        }
    ]

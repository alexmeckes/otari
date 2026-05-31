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
from fastapi import HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes import messages
from gateway.api.routes._message_models import MessagesRequest
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
        message_context=messages.MessageRequestContext(api_key_id="key-1", user_id="user-1"),
        model="claude-3-5-sonnet",
        provider="anthropic",
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
    message_context = messages.MessageRequestContext(api_key_id="key-1", user_id="user-1")

    async def fake_log_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(messages, "log_usage", fake_log_usage)

    payload = await messages._message_response_payload(
        result=_message_response(),
        response=response,
        db=db,
        log_writer=log_writer,
        message_context=message_context,
        model="claude-3-5-sonnet",
        provider="anthropic",
        rate_limit_info=RateLimitInfo(limit=10, remaining=8, reset=123.4),
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
        message_context=messages.MessageRequestContext(api_key_id=None, user_id="user-1"),
        model="claude-3-5-sonnet",
        provider="anthropic",
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
async def test_message_streaming_response_logs_usage_and_sets_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())
    message_context = messages.MessageRequestContext(api_key_id="key-1", user_id="user-1")

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
        message_context=message_context,
        model="claude-3-5-sonnet",
        provider="anthropic",
        rate_limit_info=RateLimitInfo(limit=10, remaining=8, reset=123.4),
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

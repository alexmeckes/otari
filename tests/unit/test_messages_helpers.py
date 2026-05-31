from types import SimpleNamespace
from typing import Any, cast

import pytest
from any_llm.types.completion import CompletionUsage
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes import messages
from gateway.api.routes._message_models import MessagesRequest
from gateway.models.entities import APIKey
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
        api_key_id="key-1",
        model="claude-3-5-sonnet",
        provider="anthropic",
        user_id="user-1",
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
        api_key_id=None,
        model="claude-3-5-sonnet",
        provider="anthropic",
        user_id="user-1",
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

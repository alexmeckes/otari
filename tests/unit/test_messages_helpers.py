from typing import Any, cast

import pytest
from any_llm.types.completion import CompletionUsage
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes import messages
from gateway.services.log_writer import LogWriter


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

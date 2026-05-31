from typing import Any, cast

import pytest
from any_llm.types.completion import ChatCompletion
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes import _chat_standalone_non_streaming as standalone_chat
from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.services.log_writer import LogWriter


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="openai:gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        project_id="project-1",
        tags={"team": "platform"},
    )


@pytest.mark.asyncio
async def test_log_standalone_chat_usage_forwards_success_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())
    completion = cast(ChatCompletion, object())

    async def fake_log_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(standalone_chat, "log_usage", fake_log_usage)

    await standalone_chat._log_standalone_chat_usage(
        db=db,
        log_writer=log_writer,
        api_key_id="key-1",
        model="gpt-4o-mini",
        provider="openai",
        user_id="user-1",
        request=_request(),
        response=completion,
    )

    assert calls == [
        {
            "db": db,
            "log_writer": log_writer,
            "api_key_id": "key-1",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "endpoint": "/v1/chat/completions",
            "user_id": "user-1",
            "project_id": "project-1",
            "tags": {"team": "platform"},
            "response": completion,
            "error": None,
        }
    ]


@pytest.mark.asyncio
async def test_log_standalone_chat_usage_forwards_error_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())

    async def fake_log_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(standalone_chat, "log_usage", fake_log_usage)

    await standalone_chat._log_standalone_chat_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=None,
        model="gpt-4o-mini",
        provider="openai",
        user_id=None,
        request=_request(),
        error="provider down",
    )

    assert calls == [
        {
            "db": db,
            "log_writer": log_writer,
            "api_key_id": None,
            "model": "gpt-4o-mini",
            "provider": "openai",
            "endpoint": "/v1/chat/completions",
            "user_id": None,
            "project_id": "project-1",
            "tags": {"team": "platform"},
            "response": None,
            "error": "provider down",
        }
    ]


@pytest.mark.asyncio
async def test_log_standalone_chat_usage_skips_when_database_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_log_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(standalone_chat, "log_usage", fake_log_usage)

    await standalone_chat._log_standalone_chat_usage(
        db=None,
        log_writer=cast(LogWriter, object()),
        api_key_id="key-1",
        model="gpt-4o-mini",
        provider="openai",
        user_id="user-1",
        request=_request(),
        error="provider down",
    )

    assert calls == []

from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import pytest
from any_llm import LLMProvider
from any_llm.types.completion import CompletionUsage
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes import _chat_streaming_response as streaming_response
from gateway.core.config import GatewayConfig
from gateway.services.log_writer import LogWriter


@pytest.mark.asyncio
async def test_schedule_platform_streaming_usage_forwards_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    config = GatewayConfig(mode="platform", platform={"base_url": "http://platform.test/api/v1"})

    async def fake_report_platform_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(streaming_response, "report_platform_usage", fake_report_platform_usage)

    task = streaming_response._schedule_platform_streaming_usage(
        config=config,
        correlation_id="corr-1",
        outcome="success",
        usage=usage,
    )
    await task

    assert calls == [
        {
            "config": config,
            "correlation_id": "corr-1",
            "outcome": "success",
            "usage": usage,
        }
    ]


@pytest.mark.asyncio
async def test_log_standalone_streaming_usage_forwards_success_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())
    usage = CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)

    async def fake_log_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(streaming_response, "log_usage", fake_log_usage)

    await streaming_response._log_standalone_streaming_usage(
        db=db,
        log_writer=log_writer,
        api_key_id="key-1",
        model="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
        user_id="user-1",
        project_id="project-1",
        tags={"team": "platform"},
        usage_data=usage,
    )

    assert calls == [
        {
            "db": db,
            "log_writer": log_writer,
            "api_key_id": "key-1",
            "model": "gpt-4o-mini",
            "provider": LLMProvider.OPENAI,
            "endpoint": "/v1/chat/completions",
            "user_id": "user-1",
            "project_id": "project-1",
            "tags": {"team": "platform"},
            "usage_override": usage,
            "error": None,
        }
    ]


@pytest.mark.asyncio
async def test_log_standalone_streaming_usage_forwards_error_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    db = cast(AsyncSession, object())
    log_writer = cast(LogWriter, object())

    async def fake_log_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(streaming_response, "log_usage", fake_log_usage)

    await streaming_response._log_standalone_streaming_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=None,
        model="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
        user_id=None,
        project_id=None,
        tags=None,
        error="stream failed",
    )

    assert calls == [
        {
            "db": db,
            "log_writer": log_writer,
            "api_key_id": None,
            "model": "gpt-4o-mini",
            "provider": LLMProvider.OPENAI,
            "endpoint": "/v1/chat/completions",
            "user_id": None,
            "project_id": None,
            "tags": None,
            "usage_override": None,
            "error": "stream failed",
        }
    ]


@pytest.mark.asyncio
async def test_log_standalone_streaming_usage_skips_without_database_or_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_log_usage(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(streaming_response, "log_usage", fake_log_usage)

    await streaming_response._log_standalone_streaming_usage(
        db=None,
        log_writer=cast(LogWriter, object()),
        api_key_id="key-1",
        model="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
        user_id="user-1",
        project_id="project-1",
        tags={"team": "platform"},
        error="stream failed",
    )
    await streaming_response._log_standalone_streaming_usage(
        db=cast(AsyncSession, object()),
        log_writer=None,
        api_key_id="key-1",
        model="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
        user_id="user-1",
        project_id="project-1",
        tags={"team": "platform"},
        error="stream failed",
    )

    assert calls == []


def test_build_chat_streaming_response_uses_provider_model_label(monkeypatch: pytest.MonkeyPatch) -> None:
    labels: list[str] = []
    formatters: list[Callable[[Any], str]] = []

    def fake_streaming_generator(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        labels.append(kwargs["label"])
        formatters.append(kwargs["format_chunk"])

        async def _events() -> AsyncIterator[str]:
            yield "data: [DONE]\n\n"

        return _events()

    async def empty_stream() -> AsyncIterator[Any]:
        return
        yield

    monkeypatch.setattr(streaming_response, "streaming_generator", fake_streaming_generator)

    streaming_response.build_chat_streaming_response(
        stream=empty_stream(),
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        platform_mode=False,
        correlation_id=None,
        request_id=None,
        config=GatewayConfig(),
        db=None,
        log_writer=None,
        api_key_id=None,
        user_id=None,
        project_id=None,
        tags=None,
        rate_limit_info=None,
    )

    assert labels == ["openai:gpt-4o-mini"]

    class FakeChunk:
        def model_dump_json(self) -> str:
            return '{"id":"chunk-1"}'

    assert formatters[0](FakeChunk()) == 'data: {"id":"chunk-1"}\n\n'

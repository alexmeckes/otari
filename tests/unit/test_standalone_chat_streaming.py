from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from fastapi import HTTPException

from gateway.api.routes import _chat_standalone_streaming as standalone_streaming
from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._chat_tools import ChatToolSelection
from gateway.core.config import GatewayConfig
from gateway.services.log_writer import LogWriter


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="openai:gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        project_id="project-1",
        tags={"team": "platform"},
    )


def _tool_selection() -> ChatToolSelection:
    return ChatToolSelection(
        sandbox_tool_entry=None,
        sandbox_url=None,
        use_sandbox=False,
        web_search_tool_entry=None,
        web_search_url=None,
        use_web_search=False,
        remaining_user_tools=None,
    )


@pytest.mark.asyncio
async def test_standalone_streaming_chat_logs_stream_creation_error_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("provider down")
    logger_calls: list[tuple[Any, ...]] = []

    def fake_logger_error(*args: Any) -> None:
        logger_calls.append(args)

    async def failing_completion(**kwargs: Any) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        raise error

    monkeypatch.setattr(standalone_streaming.logger, "error", fake_logger_error)

    with pytest.raises(HTTPException) as exc_info:
        await standalone_streaming.run_standalone_streaming_chat(
            request=_request(),
            config=GatewayConfig(),
            db=None,
            log_writer=cast(LogWriter, object()),
            api_key_id=None,
            user_id=None,
            rate_limit_info=None,
            tool_selection=_tool_selection(),
            completion_fn=failing_completion,
            mcp_client_pool_factory=lambda configs: object(),
        )

    assert exc_info.value.status_code == 502
    assert logger_calls == [
        ("Stream creation failed for %s: %s", "openai:gpt-4o-mini", error),
    ]

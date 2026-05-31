from dataclasses import dataclass, field

import pytest

from gateway.api.routes._usage import log_usage_error, make_usage_log
from gateway.models.entities import UsageLog


@dataclass
class StubLogWriter:
    logs: list[UsageLog] = field(default_factory=list)

    async def put(self, log: UsageLog) -> None:
        self.logs.append(log)


@pytest.mark.asyncio
async def test_log_usage_error_writes_error_log() -> None:
    writer = StubLogWriter()
    error = RuntimeError("provider down")

    await log_usage_error(
        writer,
        api_key_id="key-1",
        user_id="user-1",
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/test",
        error=error,
    )

    assert len(writer.logs) == 1
    log = writer.logs[0]
    assert log.api_key_id == "key-1"
    assert log.user_id == "user-1"
    assert log.model == "gpt-4o"
    assert log.provider == "openai"
    assert log.endpoint == "/v1/test"
    assert log.status == "error"
    assert log.error_message == "provider down"


def test_make_usage_log_sets_common_fields() -> None:
    log = make_usage_log(
        api_key_id="key-1",
        user_id="user-1",
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/test",
        prompt_tokens=3,
        completion_tokens=4,
        total_tokens=7,
        tags={"team": "platform"},
    )

    assert log.id
    assert log.api_key_id == "key-1"
    assert log.user_id == "user-1"
    assert log.model == "gpt-4o"
    assert log.provider == "openai"
    assert log.endpoint == "/v1/test"
    assert log.status == "success"
    assert log.prompt_tokens == 3
    assert log.completion_tokens == 4
    assert log.total_tokens == 7
    assert log.tags == {"team": "platform"}

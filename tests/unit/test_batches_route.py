"""Unit tests for batch route Pydantic request models."""

from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from gateway.api.routes.batches import BatchRequestItem, CreateBatchRequest, log_batch_usage
from gateway.models.entities import UsageLog


@dataclass
class StubLogWriter:
    logs: list[UsageLog] = field(default_factory=list)

    async def put(self, log: UsageLog) -> None:
        self.logs.append(log)


class TestBatchRequestItem:
    def test_valid_item(self) -> None:
        item = BatchRequestItem(
            custom_id="req-1",
            body={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100},
        )
        assert item.custom_id == "req-1"
        assert "messages" in item.body

    def test_missing_custom_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="custom_id"):
            BatchRequestItem(body={"messages": []})  # type: ignore[call-arg]


class TestCreateBatchRequest:
    def test_valid_request(self) -> None:
        request = CreateBatchRequest(
            model="openai:gpt-4o-mini",
            requests=[
                BatchRequestItem(
                    custom_id="req-1",
                    body={"messages": [{"role": "user", "content": "Hello"}]},
                ),
            ],
        )
        assert request.model == "openai:gpt-4o-mini"
        assert len(request.requests) == 1
        assert request.completion_window == "24h"
        assert request.metadata is None

    def test_empty_requests_rejected(self) -> None:
        with pytest.raises(ValidationError, match="List should have at least 1 item"):
            CreateBatchRequest(model="openai:gpt-4o-mini", requests=[])

    def test_too_many_requests_rejected(self) -> None:
        items = [BatchRequestItem(custom_id=f"req-{i}", body={}) for i in range(10_001)]
        with pytest.raises(ValidationError, match="List should have at most 10000 items"):
            CreateBatchRequest(model="openai:gpt-4o-mini", requests=items)

    def test_missing_model_rejected(self) -> None:
        with pytest.raises(ValidationError, match="model"):
            CreateBatchRequest(
                requests=[BatchRequestItem(custom_id="req-1", body={})],
            )  # type: ignore[call-arg]

    def test_optional_metadata(self) -> None:
        request = CreateBatchRequest(
            model="openai:gpt-4o-mini",
            requests=[BatchRequestItem(custom_id="req-1", body={})],
            metadata={"team": "ml-ops"},
        )
        assert request.metadata == {"team": "ml-ops"}

    def test_custom_completion_window(self) -> None:
        request = CreateBatchRequest(
            model="openai:gpt-4o-mini",
            requests=[BatchRequestItem(custom_id="req-1", body={})],
            completion_window="48h",
        )
        assert request.completion_window == "48h"


@pytest.mark.asyncio
async def test_log_batch_usage_uses_common_usage_log_shape() -> None:
    writer = StubLogWriter()

    await log_batch_usage(
        writer,
        api_key_id="key-1",
        user_id="user-1",
        model="gpt-4o-mini",
        provider="openai",
        endpoint="/v1/batches",
        error="provider down",
    )

    assert len(writer.logs) == 1
    log = writer.logs[0]
    assert log.id
    assert log.timestamp is not None
    assert log.api_key_id == "key-1"
    assert log.user_id == "user-1"
    assert log.model == "gpt-4o-mini"
    assert log.provider == "openai"
    assert log.endpoint == "/v1/batches"
    assert log.status == "error"
    assert log.error_message == "provider down"
    assert log.tags == {}

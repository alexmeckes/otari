from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import HTTPException, Response

from gateway.api.routes._provider_context import OpenAIProviderRequestContext
from gateway.models.entities import UsageLog
from gateway.rate_limit import RateLimitInfo


@dataclass
class StubLogWriter:
    logs: list[UsageLog] = field(default_factory=list)

    async def put(self, log: UsageLog) -> None:
        self.logs.append(log)


def _context(rate_limit_info: RateLimitInfo | None = None) -> OpenAIProviderRequestContext:
    return OpenAIProviderRequestContext(
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=rate_limit_info,
        provider="openai",
        model="gpt-4o-mini",
        provider_kwargs={"api_key": "sk-test"},
    )


def test_provider_context_call_kwargs_includes_provider_defaults() -> None:
    call_kwargs = _context().call_kwargs(input="hello")

    assert call_kwargs == {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "input": "hello",
        "api_key": "sk-test",
    }


def test_provider_context_call_kwargs_filters_optional_values_after_provider_defaults() -> None:
    context = OpenAIProviderRequestContext(
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=None,
        provider="openai",
        model="gpt-4o-mini",
        provider_kwargs={"api_key": "sk-test", "timeout": 5},
    )

    call_kwargs = context.call_kwargs(
        input="hello",
        timeout=1,
        optional={
            "timeout": 10,
            "temperature": None,
            "response_format": "json",
        },
    )

    assert call_kwargs == {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "input": "hello",
        "api_key": "sk-test",
        "timeout": 10,
        "response_format": "json",
    }


def test_provider_context_rate_limit_headers_are_empty_when_disabled() -> None:
    response = Response()
    context = _context()

    assert context.rate_limit_headers() == {}
    context.apply_rate_limit_headers(response)
    assert not any(header.startswith("x-ratelimit") for header in response.headers)


def test_provider_context_rate_limit_headers_include_limit_state() -> None:
    response = Response()
    context = _context(RateLimitInfo(limit=10, remaining=7, reset=123.4))

    assert context.rate_limit_headers() == {
        "X-RateLimit-Limit": "10",
        "X-RateLimit-Remaining": "7",
        "X-RateLimit-Reset": "123",
    }
    context.apply_rate_limit_headers(response)
    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Remaining"] == "7"
    assert response.headers["X-RateLimit-Reset"] == "123"


def test_provider_context_usage_log_includes_identity_fields() -> None:
    usage_log = _context().usage_log(
        endpoint="/v1/test",
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
        tags={"kind": "unit"},
    )

    assert usage_log.api_key_id == "key-1"
    assert usage_log.user_id == "user-1"
    assert usage_log.model == "gpt-4o-mini"
    assert usage_log.provider == "openai"
    assert usage_log.endpoint == "/v1/test"
    assert usage_log.prompt_tokens == 1
    assert usage_log.completion_tokens == 2
    assert usage_log.total_tokens == 3
    assert usage_log.tags == {"kind": "unit"}


@pytest.mark.asyncio
async def test_provider_context_log_zero_token_usage_writes_log() -> None:
    writer = StubLogWriter()

    usage_log = await _context().log_zero_token_usage(writer, endpoint="/v1/test")

    assert writer.logs == [usage_log]
    assert usage_log.api_key_id == "key-1"
    assert usage_log.user_id == "user-1"
    assert usage_log.model == "gpt-4o-mini"
    assert usage_log.provider == "openai"
    assert usage_log.endpoint == "/v1/test"
    assert usage_log.prompt_tokens == 0
    assert usage_log.completion_tokens == 0
    assert usage_log.total_tokens == 0


@pytest.mark.asyncio
async def test_provider_context_log_input_metered_usage_applies_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = StubLogWriter()

    class Pricing:
        input_price_per_million = 2.0

    async def fake_find_model_pricing(*args: Any, **kwargs: Any) -> Pricing:
        return Pricing()

    monkeypatch.setattr("gateway.api.routes._provider_context.find_model_pricing", fake_find_model_pricing)

    usage_log = await _context().log_input_metered_usage(
        object(),  # type: ignore[arg-type]
        writer,
        endpoint="/v1/test",
        prompt_tokens=250_000,
        total_tokens=250_000,
        cost_units=250_000,
        apply_cost=True,
    )

    assert writer.logs == [usage_log]
    assert usage_log.prompt_tokens == 250_000
    assert usage_log.completion_tokens == 0
    assert usage_log.total_tokens == 250_000
    assert usage_log.cost == 0.5


@pytest.mark.asyncio
async def test_provider_context_log_input_metered_usage_skips_cost_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = StubLogWriter()

    async def fail_find_model_pricing(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("pricing lookup should be skipped")

    monkeypatch.setattr("gateway.api.routes._provider_context.find_model_pricing", fail_find_model_pricing)

    usage_log = await _context().log_input_metered_usage(
        object(),  # type: ignore[arg-type]
        writer,
        endpoint="/v1/test",
        prompt_tokens=None,
        total_tokens=None,
        cost_units=None,
        apply_cost=False,
    )

    assert writer.logs == [usage_log]
    assert usage_log.prompt_tokens is None
    assert usage_log.completion_tokens == 0
    assert usage_log.total_tokens is None
    assert usage_log.cost is None


@pytest.mark.asyncio
async def test_provider_context_log_input_metered_usage_uses_missing_default_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = StubLogWriter()
    calls: list[tuple[str, str]] = []

    async def fake_find_model_pricing(*args: Any, **kwargs: Any) -> None:
        return None

    def fake_log_missing_pricing(provider: str, model: str) -> None:
        calls.append((provider, model))

    monkeypatch.setattr("gateway.api.routes._provider_context.find_model_pricing", fake_find_model_pricing)
    monkeypatch.setattr("gateway.api.routes._provider_context.log_missing_pricing", fake_log_missing_pricing)

    usage_log = await _context().log_input_metered_usage(
        object(),  # type: ignore[arg-type]
        writer,
        endpoint="/v1/test",
        prompt_tokens=None,
        total_tokens=None,
        cost_units=1,
        apply_cost=True,
        missing_cost=0.0,
        warn_missing_pricing=False,
    )

    assert writer.logs == [usage_log]
    assert usage_log.cost == 0.0
    assert calls == []


@pytest.mark.asyncio
async def test_provider_context_log_usage_error_writes_identity_fields() -> None:
    writer = StubLogWriter()

    await _context().log_usage_error(writer, endpoint="/v1/test", error=RuntimeError("provider down"))

    assert len(writer.logs) == 1
    log = writer.logs[0]
    assert log.api_key_id == "key-1"
    assert log.user_id == "user-1"
    assert log.model == "gpt-4o-mini"
    assert log.provider == "openai"
    assert log.endpoint == "/v1/test"
    assert log.status == "error"
    assert log.error_message == "provider down"


@pytest.mark.asyncio
async def test_provider_context_log_and_raise_provider_error_logs_then_raises() -> None:
    writer = StubLogWriter()

    with pytest.raises(HTTPException) as exc_info:
        await _context().log_and_raise_provider_error(
            writer,
            endpoint="/v1/test",
            error=RuntimeError("provider down"),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "The request could not be completed by the provider"
    assert len(writer.logs) == 1
    assert writer.logs[0].status == "error"
    assert writer.logs[0].error_message == "provider down"


@pytest.mark.asyncio
async def test_provider_context_apply_input_metered_cost_defaults_to_per_million(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_log = _context().usage_log(endpoint="/v1/test")

    class Pricing:
        input_price_per_million = 2.0

    async def fake_find_model_pricing(*args: Any, **kwargs: Any) -> Pricing:
        return Pricing()

    monkeypatch.setattr("gateway.api.routes._provider_context.find_model_pricing", fake_find_model_pricing)

    await _context().apply_input_metered_cost(
        object(),  # type: ignore[arg-type]
        usage_log,
        units=250_000,
    )

    assert usage_log.cost == 0.5


@pytest.mark.asyncio
async def test_provider_context_apply_input_metered_cost_keeps_zero_absent_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_log = _context().usage_log(endpoint="/v1/test")

    class Pricing:
        input_price_per_million = 2.0

    async def fake_find_model_pricing(*args: Any, **kwargs: Any) -> Pricing:
        return Pricing()

    monkeypatch.setattr("gateway.api.routes._provider_context.find_model_pricing", fake_find_model_pricing)

    await _context().apply_input_metered_cost(
        object(),  # type: ignore[arg-type]
        usage_log,
        units=0,
        require_positive_units=True,
    )

    assert usage_log.cost is None


@pytest.mark.asyncio
async def test_provider_context_apply_input_metered_cost_logs_missing_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_log = _context().usage_log(endpoint="/v1/test")
    calls: list[tuple[str, str]] = []

    async def fake_find_model_pricing(*args: Any, **kwargs: Any) -> None:
        return None

    def fake_log_missing_pricing(provider: str, model: str) -> None:
        calls.append((provider, model))

    monkeypatch.setattr("gateway.api.routes._provider_context.find_model_pricing", fake_find_model_pricing)
    monkeypatch.setattr("gateway.api.routes._provider_context.log_missing_pricing", fake_log_missing_pricing)

    await _context().apply_input_metered_cost(
        object(),  # type: ignore[arg-type]
        usage_log,
        units=100,
    )

    assert usage_log.cost is None
    assert calls == [("openai", "gpt-4o-mini")]


@pytest.mark.asyncio
async def test_provider_context_apply_input_metered_cost_sets_scaled_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_log = _context().usage_log(endpoint="/v1/test")

    class Pricing:
        input_price_per_million = 2.0

    async def fake_find_model_pricing(*args: Any, **kwargs: Any) -> Pricing:
        return Pricing()

    monkeypatch.setattr("gateway.api.routes._provider_context.find_model_pricing", fake_find_model_pricing)

    await _context().apply_input_metered_cost(
        object(),  # type: ignore[arg-type]
        usage_log,
        units=3,
        price_divisor=1,
    )

    assert usage_log.cost == 6.0


@pytest.mark.asyncio
async def test_provider_context_apply_input_metered_cost_uses_missing_default_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_log = _context().usage_log(endpoint="/v1/test")
    calls: list[tuple[str, str]] = []

    async def fake_find_model_pricing(*args: Any, **kwargs: Any) -> None:
        return None

    def fake_log_missing_pricing(provider: str, model: str) -> None:
        calls.append((provider, model))

    monkeypatch.setattr("gateway.api.routes._provider_context.find_model_pricing", fake_find_model_pricing)
    monkeypatch.setattr("gateway.api.routes._provider_context.log_missing_pricing", fake_log_missing_pricing)

    await _context().apply_input_metered_cost(
        object(),  # type: ignore[arg-type]
        usage_log,
        units=1,
        missing_cost=0.0,
        warn_missing_pricing=False,
    )

    assert usage_log.cost == 0.0
    assert calls == []

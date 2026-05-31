from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException, Response

from gateway.api.routes._provider_context import OpenAIProviderRequestContext, resolve_openai_provider_request_context
from gateway.models.entities import UsageLog
from gateway.rate_limit import RateLimitInfo


@dataclass
class StubLogWriter:
    logs: list[UsageLog] = field(default_factory=list)

    async def put(self, log: UsageLog) -> None:
        self.logs.append(log)


@dataclass(frozen=True)
class StubPricing:
    input_price_per_million: float = 2.0


_FIND_MODEL_PRICING = "gateway.api.routes._provider_context.find_model_pricing"
_GET_PROVIDER_KWARGS = "gateway.api.routes._provider_context.get_provider_kwargs"
_LOG_MISSING_PRICING = "gateway.api.routes._provider_context.log_missing_pricing"
_CHECK_RATE_LIMIT = "gateway.api.routes._provider_context.check_rate_limit"
_VALIDATE_SCOPED_BUDGETS = "gateway.api.routes._provider_context.validate_scoped_request_budgets"


def _context(rate_limit_info: RateLimitInfo | None = None) -> OpenAIProviderRequestContext:
    return OpenAIProviderRequestContext(
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=rate_limit_info,
        provider="openai",
        model="gpt-4o-mini",
        provider_kwargs={"api_key": "sk-test"},
    )


def _patch_model_pricing(monkeypatch: pytest.MonkeyPatch, pricing: StubPricing | None = None) -> None:
    async def fake_find_model_pricing(*args: Any, **kwargs: Any) -> StubPricing | None:
        return pricing

    monkeypatch.setattr(_FIND_MODEL_PRICING, fake_find_model_pricing)


def _patch_model_pricing_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_find_model_pricing(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("pricing lookup should be skipped")

    monkeypatch.setattr(_FIND_MODEL_PRICING, fail_find_model_pricing)


def _patch_missing_pricing_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_log_missing_pricing(provider: str, model: str) -> None:
        calls.append((provider, model))

    monkeypatch.setattr(_LOG_MISSING_PRICING, fake_log_missing_pricing)
    return calls


def test_provider_context_call_kwargs_includes_provider_defaults() -> None:
    call_kwargs = _context().call_kwargs(input="hello")

    assert call_kwargs == {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "input": "hello",
        "api_key": "sk-test",
    }


def test_provider_context_formats_provider_label() -> None:
    assert _context().provider_label == "openai:gpt-4o-mini"


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
async def test_resolve_provider_context_validates_scoped_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    db = object()
    tags = {"team": "platform"}

    async def fake_validate_scoped_request_budgets(
        db_arg: Any,
        *,
        user_id: str,
        model: str,
        project_id: str | None,
        tags: dict[str, Any] | None,
        strategy: str,
    ) -> None:
        captured.update(
            {
                "db": db_arg,
                "user_id": user_id,
                "model": model,
                "project_id": project_id,
                "tags": tags,
                "strategy": strategy,
            }
        )

    monkeypatch.setattr(_VALIDATE_SCOPED_BUDGETS, fake_validate_scoped_request_budgets)
    monkeypatch.setattr(_CHECK_RATE_LIMIT, lambda raw_request, user_id: RateLimitInfo(10, 9, 123.0))
    monkeypatch.setattr(_GET_PROVIDER_KWARGS, lambda config, provider: {"api_key": "sk-test"})

    context = await resolve_openai_provider_request_context(
        raw_request=object(),  # type: ignore[arg-type]
        auth_result=(None, True),
        user="user-1",
        db=db,  # type: ignore[arg-type]
        config=SimpleNamespace(budget_strategy="cas"),  # type: ignore[arg-type]
        model="openai:gpt-4o-mini",
        project_id="proj-1",
        tags=tags,
    )

    assert captured == {
        "db": db,
        "user_id": "user-1",
        "model": "openai:gpt-4o-mini",
        "project_id": "proj-1",
        "tags": tags,
        "strategy": "cas",
    }
    assert context.api_key_id is None
    assert context.user_id == "user-1"
    assert context.provider.value == "openai"
    assert context.model == "gpt-4o-mini"
    assert context.provider_kwargs == {"api_key": "sk-test"}
    assert context.rate_limit_info == RateLimitInfo(10, 9, 123.0)


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

    _patch_model_pricing(monkeypatch, StubPricing())

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

    _patch_model_pricing_failure(monkeypatch)

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
    calls = _patch_missing_pricing_calls(monkeypatch)

    _patch_model_pricing(monkeypatch)

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
async def test_provider_context_log_input_metered_usage_keeps_zero_absent_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = StubLogWriter()

    _patch_model_pricing(monkeypatch, StubPricing())

    usage_log = await _context().log_input_metered_usage(
        object(),  # type: ignore[arg-type]
        writer,
        endpoint="/v1/test",
        prompt_tokens=0,
        total_tokens=0,
        cost_units=0,
        apply_cost=True,
        require_positive_units=True,
    )

    assert writer.logs == [usage_log]
    assert usage_log.cost is None


@pytest.mark.asyncio
async def test_provider_context_log_input_metered_usage_logs_missing_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = StubLogWriter()
    calls = _patch_missing_pricing_calls(monkeypatch)

    _patch_model_pricing(monkeypatch)

    usage_log = await _context().log_input_metered_usage(
        object(),  # type: ignore[arg-type]
        writer,
        endpoint="/v1/test",
        prompt_tokens=100,
        total_tokens=100,
        cost_units=100,
        apply_cost=True,
    )

    assert writer.logs == [usage_log]
    assert usage_log.cost is None
    assert calls == [("openai", "gpt-4o-mini")]


@pytest.mark.asyncio
async def test_provider_context_log_input_metered_usage_sets_scaled_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = StubLogWriter()

    _patch_model_pricing(monkeypatch, StubPricing())

    usage_log = await _context().log_input_metered_usage(
        object(),  # type: ignore[arg-type]
        writer,
        endpoint="/v1/test",
        prompt_tokens=0,
        total_tokens=0,
        cost_units=3,
        apply_cost=True,
        price_divisor=1,
    )

    assert writer.logs == [usage_log]
    assert usage_log.cost == 6.0


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

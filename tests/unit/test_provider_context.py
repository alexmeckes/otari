from gateway.api.routes._provider_context import OpenAIProviderRequestContext


def _context() -> OpenAIProviderRequestContext:
    return OpenAIProviderRequestContext(
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=None,
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

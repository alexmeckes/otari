from gateway.api.routes._completion_usage import completion_usage_from_token_counts


def test_completion_usage_from_token_counts_derives_total_when_absent() -> None:
    usage = completion_usage_from_token_counts(input_tokens=7, output_tokens=3)

    assert usage.prompt_tokens == 7
    assert usage.completion_tokens == 3
    assert usage.total_tokens == 10


def test_completion_usage_from_token_counts_preserves_provider_total() -> None:
    usage = completion_usage_from_token_counts(input_tokens=7, output_tokens=3, total_tokens=12)

    assert usage.prompt_tokens == 7
    assert usage.completion_tokens == 3
    assert usage.total_tokens == 12

from any_llm.types.completion import CompletionUsage

from gateway.services.platform_gateway import _platform_usage_payload


def test_platform_usage_payload_includes_success_usage() -> None:
    payload = _platform_usage_payload(
        "corr-1",
        "success",
        CompletionUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )

    assert payload == {
        "correlation_id": "corr-1",
        "status": "success",
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
    }


def test_platform_usage_payload_defaults_missing_success_usage_to_zero_tokens() -> None:
    payload = _platform_usage_payload("corr-1", "success", None)

    assert payload == {
        "correlation_id": "corr-1",
        "status": "success",
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def test_platform_usage_payload_includes_error_class_for_errors() -> None:
    payload = _platform_usage_payload("corr-1", "error", None, error_class="http_503")

    assert payload == {
        "correlation_id": "corr-1",
        "status": "error",
        "error_class": "http_503",
    }

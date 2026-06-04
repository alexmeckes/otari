import pytest

from gateway.services.routing_trace_attempts import (
    attempt_duration_ms,
    attempt_model_key,
    attempt_outcome,
    attempt_provider,
)


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        ({"model_key": "openai:gpt-4o", "provider": "ignored", "model": "ignored"}, "openai:gpt-4o"),
        ({"provider": "anthropic", "model": "claude-3-5-haiku"}, "anthropic:claude-3-5-haiku"),
        ({"model_key": "", "provider": "openai", "model": "gpt-4o-mini"}, "openai:gpt-4o-mini"),
        ({"provider": "openai"}, None),
        ({"model": "gpt-4o"}, None),
        ({"model_key": 12, "provider": "openai", "model": 7}, None),
    ],
)
def test_attempt_model_key(attempt: dict[str, object], expected: str | None) -> None:
    assert attempt_model_key(attempt) == expected


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        ({"duration_ms": 42}, 42.0),
        ({"duration_ms": 0}, 0.0),
        ({"duration_ms": 12.5}, 12.5),
        ({"duration_ms": -1}, None),
        ({"duration_ms": True}, None),
        ({"duration_ms": "42"}, None),
        ({}, None),
    ],
)
def test_attempt_duration_ms(attempt: dict[str, object], expected: float | None) -> None:
    assert attempt_duration_ms(attempt) == expected


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        ({"provider": "openai"}, "openai"),
        ({"provider": ""}, None),
        ({"provider": 1}, None),
        ({}, None),
    ],
)
def test_attempt_provider(attempt: dict[str, object], expected: str | None) -> None:
    assert attempt_provider(attempt) == expected


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        ({"status": "success"}, "success"),
        ({"status": "error"}, "error"),
        ({"status": "skipped"}, None),
        ({"status": True}, None),
        ({}, None),
    ],
)
def test_attempt_outcome(attempt: dict[str, object], expected: str | None) -> None:
    assert attempt_outcome(attempt) == expected

from gateway.services.routing_guardrail_helpers import (
    guardrail_violation,
    guardrails_config,
    named_patterns,
    string_list,
)


def test_string_list_preserves_existing_list_coercion() -> None:
    assert string_list(" alpha ") == ["alpha"]
    assert string_list([" beta ", "", None, 7]) == ["beta", "None", "7"]
    assert string_list({"not": "a-list"}) == []


def test_guardrails_config_returns_mapping_only() -> None:
    guardrails = {"enabled": True}
    assert guardrails_config({"guardrails": guardrails}) == guardrails
    assert guardrails_config({"guardrails": ["enabled"]}) == {}
    assert guardrails_config({}) == {}


def test_named_patterns_accepts_strings_and_named_pattern_objects() -> None:
    patterns = named_patterns(
        [
            "secret",
            {"name": "emailish", "pattern": r"@example\.com"},
            {"pattern": "token"},
            {"name": "missing"},
        ]
    )

    assert [name for name, _ in patterns] == ["pattern_1", "emailish", "pattern_3"]
    assert patterns[0][1].search("SECRET")
    assert patterns[1][1].search("a@example.com")


def test_named_patterns_skips_invalid_regex() -> None:
    assert named_patterns([{"name": "bad", "pattern": "["}]) == []


def test_guardrail_violation_shape() -> None:
    assert guardrail_violation("pii", "email") == {"type": "pii", "rule": "email"}

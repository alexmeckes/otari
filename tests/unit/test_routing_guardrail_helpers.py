from gateway.services.routing_guardrail_helpers import (
    _compiled_named_pattern,
    _compiled_named_pattern_item,
    _named_pattern_item,
    guardrail_config_value,
    guardrail_violation,
    guardrails_config,
    named_patterns,
    pii_patterns_from_config,
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


def test_guardrail_config_value_preserves_dict_and_scalar_shorthand_behavior() -> None:
    assert guardrail_config_value({"enabled": True}, "enabled", scalar_value=False) is True
    assert guardrail_config_value({"types": ["email"]}, "missing", scalar_value=["ssn"]) is None
    assert guardrail_config_value(True, "enabled", scalar_value=True) is True
    assert guardrail_config_value("pii", "types") is None


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


def test_named_patterns_trims_names_and_uses_default_for_blank_names() -> None:
    patterns = named_patterns(
        [
            {"name": " secret_name ", "pattern": "secret"},
            {"name": " ", "pattern": "token"},
        ]
    )

    assert [name for name, _ in patterns] == ["secret_name", "pattern_2"]
    assert patterns[0][1].search("SECRET")
    assert patterns[1][1].search("TOKEN")


def test_named_patterns_skips_blank_pattern_values() -> None:
    patterns = named_patterns(
        [
            {"name": "blank", "pattern": "   "},
            {"name": "missing"},
            {"name": "numeric", "pattern": 123},
            {"name": "kept", "pattern": " secret "},
        ]
    )

    assert [name for name, _ in patterns] == ["kept"]
    assert patterns[0][1].search(" secret ")
    assert not patterns[0][1].search("SECRET")


def test_named_patterns_skips_invalid_regex() -> None:
    assert named_patterns([{"name": "bad", "pattern": "["}]) == []


def test_named_pattern_item_normalizes_name_and_preserves_regex_text() -> None:
    assert _named_pattern_item(" secret ", 1) == ("pattern_1", " secret ")
    assert _named_pattern_item({"name": " token ", "pattern": " token "}, 2) == ("token", " token ")
    assert _named_pattern_item({"name": " ", "pattern": " token "}, 3) == ("pattern_3", " token ")
    assert _named_pattern_item({"name": "blank", "pattern": "   "}, 4) is None
    assert _named_pattern_item({"name": "numeric", "pattern": 123}, 5) is None


def test_compiled_named_pattern_preserves_name_and_ignores_case() -> None:
    compiled = _compiled_named_pattern("secret", "secret")

    assert compiled is not None
    name, pattern = compiled
    assert name == "secret"
    assert pattern.search("SECRET")
    assert _compiled_named_pattern("bad", "[") is None


def test_compiled_named_pattern_item_compiles_or_skips_items() -> None:
    compiled = _compiled_named_pattern_item({"name": " token ", "pattern": "token"}, 1)

    assert compiled is not None
    name, pattern = compiled
    assert name == "token"
    assert pattern.search("TOKEN")
    assert _compiled_named_pattern_item({"name": "blank", "pattern": "   "}, 2) is None
    assert _compiled_named_pattern_item({"name": "bad", "pattern": "["}, 3) is None


def test_guardrail_violation_shape() -> None:
    assert guardrail_violation("pii", "email") == {"type": "pii", "rule": "email"}


def test_pii_patterns_from_config_returns_enabled_types() -> None:
    patterns = pii_patterns_from_config({"enabled": True, "types": ["email", "missing"]})

    assert [name for name, _ in patterns] == ["email"]
    assert patterns[0][1].search("ada@example.com")


def test_pii_patterns_from_config_uses_fallback_types_for_boolean_config() -> None:
    patterns = pii_patterns_from_config(True, fallback_types=["ssn"])

    assert [name for name, _ in patterns] == ["ssn"]
    assert patterns[0][1].search("123-45-6789")


def test_pii_patterns_from_config_returns_empty_when_disabled() -> None:
    assert pii_patterns_from_config({"enabled": False, "types": ["email"]}) == []

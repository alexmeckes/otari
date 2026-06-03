from gateway.services.routing_guardrail_redactions import (
    _missing_redaction_rules_trace,
    _redact_mapping,
    _redact_messages,
    _redact_request_fields,
    _redact_string,
    _redaction_replacement,
    _redaction_rules,
    _redaction_trace,
    _redactions_config,
    _redactions_enabled,
    apply_guardrail_redactions,
)


def test_redaction_rules_collects_pii_and_named_patterns() -> None:
    rules, pattern_count = _redaction_rules(
        {
            "pii": True,
            "pii_types": ["email"],
            "patterns": [{"name": "token", "pattern": r"token-[0-9]+"}],
        }
    )

    assert [(kind, rule) for kind, rule, _pattern in rules] == [
        ("pii", "email"),
        ("pattern", "token"),
    ]
    assert pattern_count == 1


def test_redactions_config_extracts_nested_redactions_mapping() -> None:
    redactions = {"patterns": [{"name": "token", "pattern": r"token-[0-9]+"}]}

    assert _redactions_config({"guardrails": {"redactions": redactions}}) == redactions
    assert _redactions_config({}) == {}
    assert _redactions_config({"guardrails": ["redactions"]}) == {}
    assert _redactions_config({"guardrails": {"redactions": ["enabled"]}}) == {}


def test_redact_string_replaces_matches_and_updates_counts() -> None:
    rules, _pattern_count = _redaction_rules({"patterns": [{"name": "token", "pattern": r"token-[0-9]+"}]})
    counts: dict[tuple[str, str], int] = {}

    redacted = _redact_string("token-123 and token-456", rules=rules, replacement="[MASKED]", counts=counts)

    assert redacted == "[MASKED] and [MASKED]"
    assert counts == {("pattern", "token"): 2}
    assert _redact_string("nothing to mask", rules=rules, replacement="[MASKED]", counts=counts) == "nothing to mask"
    assert counts == {("pattern", "token"): 2}


def test_redact_mapping_recurses_nested_values_without_mutating_input() -> None:
    rules, _pattern_count = _redaction_rules({"patterns": [{"name": "token", "pattern": r"token-[0-9]+"}]})
    counts: dict[tuple[str, str], int] = {}
    value = {
        "items": ["token-123", {"nested": "token-456"}],
        "unchanged": 3,
    }

    redacted = _redact_mapping(value, rules=rules, replacement="[MASKED]", counts=counts)

    assert redacted == {
        "items": ["[MASKED]", {"nested": "[MASKED]"}],
        "unchanged": 3,
    }
    assert value == {
        "items": ["token-123", {"nested": "token-456"}],
        "unchanged": 3,
    }
    assert counts == {("pattern", "token"): 2}


def test_redaction_replacement_preserves_strings_and_defaults_other_values() -> None:
    assert _redaction_replacement({"replacement": "[MASKED]"}) == "[MASKED]"
    assert _redaction_replacement({}) == "[REDACTED]"
    assert _redaction_replacement({"replacement": None}) == "[REDACTED]"
    assert _redaction_replacement({"replacement": 123}) == "[REDACTED]"


def test_redactions_enabled_uses_explicit_flag_and_config_presence() -> None:
    assert _redactions_enabled({}) is False
    assert _redactions_enabled({"patterns": [{"name": "token", "pattern": r"token-[0-9]+"}]}) is True
    assert _redactions_enabled({"enabled": True}) is True
    assert _redactions_enabled(
        {
            "enabled": False,
            "patterns": [{"name": "token", "pattern": r"token-[0-9]+"}],
        }
    ) is False


def test_redact_messages_redacts_content_and_preserves_other_items() -> None:
    rules, _pattern_count = _redaction_rules({"patterns": [{"name": "token", "pattern": r"token-[0-9]+"}]})
    counts: dict[tuple[str, str], int] = {}
    messages = [
        {"role": "user", "content": "token-123"},
        {"role": "assistant", "tool_calls": []},
        "raw",
    ]

    redacted = _redact_messages(messages, rules=rules, replacement="[MASKED]", counts=counts)

    assert redacted == [
        {"role": "user", "content": "[MASKED]"},
        {"role": "assistant", "tool_calls": []},
        "raw",
    ]
    assert messages[0]["content"] == "token-123"
    assert counts == {("pattern", "token"): 1}
    assert _redact_messages("not-a-list", rules=rules, replacement="[MASKED]", counts=counts) is None


def test_redact_request_fields_redacts_supported_fields_only() -> None:
    rules, _pattern_count = _redaction_rules({"patterns": [{"name": "token", "pattern": r"token-[0-9]+"}]})
    counts: dict[tuple[str, str], int] = {}
    body = {
        "input": {"note": "token-123"},
        "instructions": "token-456",
        "other": "token-789",
    }

    _redact_request_fields(body, rules=rules, replacement="[MASKED]", counts=counts)

    assert body == {
        "input": {"note": "[MASKED]"},
        "instructions": "[MASKED]",
        "other": "token-789",
    }
    assert counts == {("pattern", "token"): 2}


def test_missing_redaction_rules_trace_marks_skipped() -> None:
    assert _missing_redaction_rules_trace("[MASKED]") == {
        "enabled": True,
        "status": "skipped",
        "reason": "missing_rules",
        "replacement": "[MASKED]",
    }


def test_redaction_trace_sorts_counts_and_marks_status() -> None:
    trace = _redaction_trace(
        replacement="[MASKED]",
        counts={("pii", "email"): 2, ("pattern", "token"): 1},
        pattern_count=1,
    )

    assert trace == {
        "enabled": True,
        "status": "redacted",
        "replacement": "[MASKED]",
        "total_replacements": 3,
        "counts": [
            {"type": "pattern", "rule": "token", "count": 1},
            {"type": "pii", "rule": "email", "count": 2},
        ],
        "pattern_count": 1,
    }
    assert _redaction_trace(replacement="[MASKED]", counts={}, pattern_count=0) == {
        "enabled": True,
        "status": "unchanged",
        "replacement": "[MASKED]",
        "total_replacements": 0,
        "counts": [],
        "pattern_count": 0,
    }


def test_apply_guardrail_redactions_uses_configured_rules_and_replacement() -> None:
    body, trace = apply_guardrail_redactions(
        {
            "guardrails": {
                "redactions": {
                    "replacement": "[MASKED]",
                    "pii": True,
                    "pii_types": ["email"],
                    "patterns": [{"name": "token", "pattern": r"token-[0-9]+"}],
                }
            }
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Email ada@example.com about token-123.",
                }
            ],
            "input": {"note": "token-456"},
            "instructions": "Contact bob@example.com",
        },
    )

    assert body["messages"][0]["content"] == "Email [MASKED] about [MASKED]."
    assert body["input"] == {"note": "[MASKED]"}
    assert body["instructions"] == "Contact [MASKED]"
    assert trace == {
        "enabled": True,
        "status": "redacted",
        "replacement": "[MASKED]",
        "total_replacements": 4,
        "counts": [
            {"type": "pattern", "rule": "token", "count": 2},
            {"type": "pii", "rule": "email", "count": 2},
        ],
        "pattern_count": 1,
    }


def test_apply_guardrail_redactions_reports_missing_rules_with_default_replacement() -> None:
    body, trace = apply_guardrail_redactions(
        {"guardrails": {"redactions": {"enabled": True}}},
        {"messages": [{"role": "user", "content": "hello"}]},
    )

    assert body == {"messages": [{"role": "user", "content": "hello"}]}
    assert trace == {
        "enabled": True,
        "status": "skipped",
        "reason": "missing_rules",
        "replacement": "[REDACTED]",
    }

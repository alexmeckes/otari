from gateway.services.routing_guardrail_redactions import (
    _redact_list,
    _redact_mapping,
    _redact_string,
    _redaction_rules,
    _RedactionContext,
    apply_guardrail_redactions,
)


def _redaction_context(replacement: str = "[MASKED]") -> _RedactionContext:
    rules, _pattern_count = _redaction_rules({"patterns": [{"name": "token", "pattern": r"token-[0-9]+"}]})
    return _RedactionContext(rules=rules, replacement=replacement, counts={})


def test_redaction_rules_collects_pii_and_named_patterns() -> None:
    rules, pattern_count = _redaction_rules(
        {
            "pii": True,
            "pii_types": ["email"],
            "patterns": [
                {"name": "token", "pattern": r"token-[0-9]+"},
                {"name": "secret", "pattern": r"secret-[a-z]+"},
            ],
        }
    )

    assert [(kind, rule) for kind, rule, _pattern in rules] == [
        ("pii", "email"),
        ("pattern", "token"),
        ("pattern", "secret"),
    ]
    assert pattern_count == 2
    assert rules[1][2].pattern == r"token-[0-9]+"
    assert _redaction_rules({"patterns": "not-a-list"}) == ([], 0)
    assert _redaction_rules({"pii": False, "pii_types": ["email"]}) == ([], 0)


def test_redact_string_replaces_matches_and_updates_counts() -> None:
    context = _redaction_context()

    redacted = _redact_string("token-123 and token-456", context=context)

    assert redacted == "[MASKED] and [MASKED]"
    assert context.counts == {("pattern", "token"): 2}
    assert _redact_string("nothing to mask", context=context) == "nothing to mask"
    assert context.counts == {("pattern", "token"): 2}


def test_redact_mapping_recurses_nested_values_without_mutating_input() -> None:
    context = _redaction_context()
    value = {
        "items": ["token-123", {"nested": "token-456"}],
        "unchanged": 3,
    }

    redacted = _redact_mapping(value, context=context)

    assert redacted == {
        "items": ["[MASKED]", {"nested": "[MASKED]"}],
        "unchanged": 3,
    }
    assert value == {
        "items": ["token-123", {"nested": "token-456"}],
        "unchanged": 3,
    }
    assert context.counts == {("pattern", "token"): 2}


def test_redact_list_recurses_nested_values_without_mutating_input() -> None:
    context = _redaction_context()
    value = ["token-123", {"nested": "token-456"}, 3]

    redacted = _redact_list(value, context=context)

    assert redacted == ["[MASKED]", {"nested": "[MASKED]"}, 3]
    assert value == ["token-123", {"nested": "token-456"}, 3]
    assert context.counts == {("pattern", "token"): 2}


def test_apply_guardrail_redactions_uses_configured_rules_and_replacement() -> None:
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": "Email ada@example.com about token-123.",
            },
            {"role": "assistant", "tool_calls": []},
            "raw",
        ],
        "input": {"note": "token-456"},
        "instructions": "Contact bob@example.com",
        "other": "token-789",
    }

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
        request_body,
    )

    assert body["messages"] == [
        {"role": "user", "content": "Email [MASKED] about [MASKED]."},
        {"role": "assistant", "tool_calls": []},
        "raw",
    ]
    assert body["input"] == {"note": "[MASKED]"}
    assert body["instructions"] == "Contact [MASKED]"
    assert body["other"] == "token-789"
    assert request_body["messages"][0]["content"] == "Email ada@example.com about token-123."
    assert request_body["input"] == {"note": "token-456"}
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


def test_apply_guardrail_redactions_preserves_non_list_messages() -> None:
    body, trace = apply_guardrail_redactions(
        {
            "guardrails": {
                "redactions": {
                    "replacement": "[MASKED]",
                    "patterns": [{"name": "token", "pattern": r"token-[0-9]+"}],
                }
            }
        },
        {"messages": "token-123", "input": "token-456"},
    )

    assert body == {"messages": "token-123", "input": "[MASKED]"}
    assert trace == {
        "enabled": True,
        "status": "redacted",
        "replacement": "[MASKED]",
        "total_replacements": 1,
        "counts": [{"type": "pattern", "rule": "token", "count": 1}],
        "pattern_count": 1,
    }


def test_apply_guardrail_redactions_reports_unchanged_when_rules_do_not_match() -> None:
    body, trace = apply_guardrail_redactions(
        {
            "guardrails": {
                "redactions": {
                    "replacement": "[MASKED]",
                    "patterns": [{"name": "token", "pattern": r"token-[0-9]+"}],
                }
            }
        },
        {"messages": [{"role": "user", "content": "hello"}]},
    )

    assert body == {"messages": [{"role": "user", "content": "hello"}]}
    assert trace == {
        "enabled": True,
        "status": "unchanged",
        "replacement": "[MASKED]",
        "total_replacements": 0,
        "counts": [],
        "pattern_count": 1,
    }


def test_apply_guardrail_redactions_reports_missing_rules_with_default_replacement() -> None:
    body, trace = apply_guardrail_redactions(
        {"guardrails": {"redactions": {"enabled": True, "replacement": 123}}},
        {"messages": [{"role": "user", "content": "hello"}]},
    )

    assert body == {"messages": [{"role": "user", "content": "hello"}]}
    assert trace == {
        "enabled": True,
        "status": "skipped",
        "reason": "missing_rules",
        "replacement": "[REDACTED]",
    }


def test_apply_guardrail_redactions_respects_disabled_or_missing_redactions() -> None:
    request_body = {"messages": [{"role": "user", "content": "token-123"}]}

    for config in (
        {},
        {"guardrails": ["redactions"]},
        {"guardrails": {"redactions": ["enabled"]}},
        {
            "guardrails": {
                "redactions": {
                    "enabled": False,
                    "patterns": [{"name": "token", "pattern": r"token-[0-9]+"}],
                }
            }
        },
    ):
        body, trace = apply_guardrail_redactions(config, request_body)

        assert body == request_body
        assert body is not request_body
        assert trace is None

import re

from gateway.services.routing_guardrail_redactions import (
    _REDACTABLE_REQUEST_FIELDS,
    _redact_list,
    _redact_mapping,
    _redact_message,
    _redact_messages,
    _redact_request_fields,
    _redact_string,
    _redaction_rules,
    _redaction_trace,
    _RedactionContext,
    _typed_redaction_rules,
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


def test_redact_message_redacts_content_copy_and_passes_through_other_items() -> None:
    context = _redaction_context()
    message = {"role": "user", "content": "token-123", "metadata": {"keep": True}}

    redacted = _redact_message(message, context=context)

    assert redacted == {"role": "user", "content": "[MASKED]", "metadata": {"keep": True}}
    assert redacted is not message
    assert message["content"] == "token-123"
    passthrough = {"role": "assistant", "tool_calls": []}
    assert _redact_message(passthrough, context=context) is passthrough
    assert _redact_message("raw", context=context) == "raw"
    assert context.counts == {("pattern", "token"): 1}


def test_redact_messages_redacts_content_and_preserves_other_items() -> None:
    context = _redaction_context()
    messages = [
        {"role": "user", "content": "token-123"},
        {"role": "assistant", "tool_calls": []},
        "raw",
    ]

    redacted = _redact_messages(messages, context=context)

    assert redacted == [
        {"role": "user", "content": "[MASKED]"},
        {"role": "assistant", "tool_calls": []},
        "raw",
    ]
    assert messages[0]["content"] == "token-123"
    assert context.counts == {("pattern", "token"): 1}
    assert _redact_messages("not-a-list", context=context) is None


def test_redactable_request_fields_names_supported_provider_fields() -> None:
    assert _REDACTABLE_REQUEST_FIELDS == ("input", "instructions")


def test_typed_redaction_rules_preserve_kind_order_names_and_patterns() -> None:
    token = re.compile(r"token-[0-9]+", re.IGNORECASE)
    email = re.compile(r"@example\.com", re.IGNORECASE)

    assert _typed_redaction_rules("pattern", [("token", token), ("email", email)]) == [
        ("pattern", "token", token),
        ("pattern", "email", email),
    ]


def test_redact_request_fields_redacts_supported_fields_only() -> None:
    context = _redaction_context()
    body = {
        "input": {"note": "token-123"},
        "instructions": "token-456",
        "other": "token-789",
    }

    _redact_request_fields(body, context=context)

    assert body == {
        "input": {"note": "[MASKED]"},
        "instructions": "[MASKED]",
        "other": "token-789",
    }
    assert context.counts == {("pattern", "token"): 2}


def test_redaction_trace_sorts_counts_and_marks_status() -> None:
    context = _RedactionContext(
        rules=[],
        replacement="[MASKED]",
        counts={("pii", "email"): 2, ("pattern", "token"): 1},
    )

    trace = _redaction_trace(
        context=context,
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
    unchanged_context = _RedactionContext(rules=[], replacement="[MASKED]", counts={})
    assert _redaction_trace(context=unchanged_context, pattern_count=0) == {
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

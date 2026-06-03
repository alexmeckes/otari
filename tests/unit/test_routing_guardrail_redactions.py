from gateway.services.routing_guardrail_redactions import (
    _missing_redaction_rules_trace,
    _redact_messages,
    _redaction_rules,
    _redaction_trace,
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

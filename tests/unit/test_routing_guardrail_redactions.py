from gateway.services.routing_guardrail_redactions import (
    _redaction_rules,
    apply_guardrail_redactions,
)


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
        "input": {"items": ["token-456", {"nested": "token-654"}]},
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
    assert body["input"] == {"items": ["[MASKED]", {"nested": "[MASKED]"}]}
    assert body["instructions"] == "Contact [MASKED]"
    assert body["other"] == "token-789"
    assert request_body["messages"][0]["content"] == "Email ada@example.com about token-123."
    assert request_body["input"] == {"items": ["token-456", {"nested": "token-654"}]}
    assert trace == {
        "enabled": True,
        "status": "redacted",
        "replacement": "[MASKED]",
        "total_replacements": 5,
        "counts": [
            {"type": "pattern", "rule": "token", "count": 3},
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

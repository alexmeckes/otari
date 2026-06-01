from gateway.services.routing_guardrail_redactions import apply_guardrail_redactions


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

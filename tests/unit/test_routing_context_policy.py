from gateway.services.routing_context_policy import apply_context_policy


def test_context_policy_summarizes_with_normalized_string_settings() -> None:
    messages = [
        {"role": " USER ", "content": "alpha " * 80},
        {"role": "assistant", "content": "beta " * 80},
        {"role": "user", "content": "final"},
    ]

    body, trace = apply_context_policy(
        {
            "context": {
                "enabled": True,
                "strategy": " summarize_MESSAGES ",
                "max_prompt_tokens": 80,
                "preserve_system_messages": False,
                "preserve_last_messages": 1,
                "summary_max_tokens": 40,
                "summary_message_max_chars": 60,
                "summary_role": " Developer ",
            }
        },
        {"messages": messages},
    )

    assert trace is not None
    assert trace["status"] == "summarized"
    assert trace["strategy"] == "summarize_messages"
    assert trace["summary_message_role"] == "developer"
    assert body["messages"][0]["role"] == "developer"
    assert body["messages"][1] == messages[-1]
    assert "- user: alpha" in body["messages"][0]["content"]


def test_context_policy_blank_summary_prefix_uses_default() -> None:
    messages = [
        {"role": "user", "content": "alpha " * 80},
        {"role": "assistant", "content": "beta " * 80},
        {"role": "user", "content": "final"},
    ]

    body, trace = apply_context_policy(
        {
            "context": {
                "enabled": True,
                "strategy": "summarize_messages",
                "max_prompt_tokens": 80,
                "preserve_system_messages": False,
                "preserve_last_messages": 1,
                "summary_max_tokens": 40,
                "summary_prefix": "   ",
            }
        },
        {"messages": messages},
    )

    assert trace is not None
    assert trace["status"] == "summarized"
    assert body["messages"][0]["content"].startswith("Earlier conversation summary:")


def test_context_policy_unsupported_blank_strategy_reports_original_value() -> None:
    request_body = {"messages": [{"role": "user", "content": "hello"}]}

    body, trace = apply_context_policy(
        {
            "context": {
                "enabled": True,
                "strategy": "   ",
                "max_prompt_tokens": 1,
            }
        },
        request_body,
    )

    assert body == request_body
    assert trace == {
        "enabled": True,
        "status": "skipped",
        "reason": "unsupported_strategy",
        "strategy": "   ",
    }

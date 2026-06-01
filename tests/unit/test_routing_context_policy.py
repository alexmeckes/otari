from gateway.services.routing_context_policy import apply_context_policy


def test_context_policy_enabled_defaults_follow_config_presence() -> None:
    request_body = {"messages": [{"role": "user", "content": "hello"}]}

    assert apply_context_policy({}, request_body) == (request_body, None)
    assert apply_context_policy({"context": {"enabled": False, "max_prompt_tokens": 1}}, request_body) == (
        request_body,
        None,
    )
    _, trace = apply_context_policy({"context": {"max_prompt_tokens": 1}}, request_body)

    assert trace is not None
    assert trace["status"] == "unchanged"


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


def test_context_policy_uses_context_policy_fallback_when_context_is_not_dict() -> None:
    messages = [
        {"role": "user", "content": "alpha " * 80},
        {"role": "user", "content": "final"},
    ]

    body, trace = apply_context_policy(
        {
            "context": "summarize",
            "context_policy": {
                "enabled": True,
                "strategy": "trim_messages",
                "max_prompt_tokens": 30,
                "preserve_system_messages": False,
                "preserve_last_messages": 1,
            },
        },
        {"messages": messages},
    )

    assert trace is not None
    assert trace["status"] == "trimmed"
    assert body["messages"] == [messages[-1]]


def test_context_policy_summary_reads_all_settings_from_context_policy_fallback() -> None:
    messages = [
        {"role": "user", "content": "alpha " * 80},
        {"role": "assistant", "content": "beta " * 80},
        {"role": "user", "content": "final"},
    ]

    body, trace = apply_context_policy(
        {
            "context": "summarize",
            "context_policy": {
                "enabled": True,
                "strategy": "summarize_messages",
                "max_prompt_tokens": 80,
                "preserve_system_messages": False,
                "preserve_last_messages": 1,
                "summary_max_tokens": 40,
                "summary_message_max_chars": 20,
                "summary_prefix": "Recap:",
                "summary_role": "user",
            },
        },
        {"messages": messages},
    )

    assert trace is not None
    assert trace["status"] == "summarized"
    assert trace["summary_message_role"] == "user"
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"].startswith("Recap:")
    assert body["messages"][1] == messages[-1]


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

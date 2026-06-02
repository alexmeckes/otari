"""Unit tests for chat tool extraction helpers.

The helper has to recognise both the gateway-native short form and the
provider-native shapes (OpenAI `code_interpreter`, Anthropic versioned
`code_execution_*`) so that pointing an OpenAI/Anthropic SDK at the gateway's
`base_url` keeps working unchanged.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._chat_request_fields import chat_provider_call_kwargs, chat_provider_request_fields
from gateway.api.routes._chat_tools import resolve_chat_tool_selection
from gateway.models.mcp import McpServerConfig
from gateway.services.chat_tool_config import (
    build_web_search_backend,
    extract_code_execution_tool,
    extract_web_search_tool,
)
from gateway.services.web_search_backend import WebSearchBackend


def test_extracts_gateway_native_short_form() -> None:
    entry, remaining = extract_code_execution_tool([{"type": "code_execution"}])
    assert entry == {"type": "code_execution"}
    assert remaining is None


def test_extracts_openai_code_interpreter_alias() -> None:
    entry, remaining = extract_code_execution_tool([{"type": "code_interpreter"}])
    assert entry == {"type": "code_interpreter"}
    assert remaining is None


def test_extracts_anthropic_versioned_alias() -> None:
    entry, remaining = extract_code_execution_tool([{"type": "code_execution_20250825"}])
    assert entry == {"type": "code_execution_20250825"}
    assert remaining is None


def test_extracts_future_anthropic_version_by_prefix() -> None:
    entry, _ = extract_code_execution_tool([{"type": "code_execution_20991231"}])
    assert entry is not None


def test_passes_through_unrelated_tools() -> None:
    user_tool = {"type": "function", "function": {"name": "get_weather"}}
    entry, remaining = extract_code_execution_tool([user_tool, {"type": "code_execution"}])
    assert entry == {"type": "code_execution"}
    assert remaining == [user_tool]


def test_takes_only_the_first_code_execution_entry() -> None:
    entry, remaining = extract_code_execution_tool(
        [
            {"type": "code_execution", "purpose_hint": "first"},
            {"type": "code_interpreter"},
        ]
    )
    assert entry == {"type": "code_execution", "purpose_hint": "first"}
    assert remaining == [{"type": "code_interpreter"}]


def test_returns_no_entry_when_absent() -> None:
    entry, remaining = extract_code_execution_tool([{"type": "function", "function": {"name": "f"}}])
    assert entry is None
    assert remaining == [{"type": "function", "function": {"name": "f"}}]


def test_empty_tools_returns_no_entry() -> None:
    entry, remaining = extract_code_execution_tool(None)
    assert entry is None
    assert remaining is None


def test_does_not_match_unrelated_types_starting_with_code() -> None:
    entry, _ = extract_code_execution_tool([{"type": "code_review"}, {"type": "code_executioner"}])
    assert entry is None


def test_non_string_type_does_not_match() -> None:
    entry, _ = extract_code_execution_tool([{"type": None}, {"type": 42}])
    assert entry is None


# --- web_search extraction ---------------------------------------------------


def test_web_search_extracts_gateway_native_short_form() -> None:
    entry, remaining = extract_web_search_tool([{"type": "web_search"}])
    assert entry == {"type": "web_search"}
    assert remaining is None


def test_web_search_extracts_anthropic_versioned_alias() -> None:
    entry, remaining = extract_web_search_tool([{"type": "web_search_20250305"}])
    assert entry == {"type": "web_search_20250305"}
    assert remaining is None


def test_web_search_extracts_future_anthropic_version_by_prefix() -> None:
    entry, _ = extract_web_search_tool([{"type": "web_search_20991231"}])
    assert entry is not None


def test_web_search_passes_through_unrelated_tools() -> None:
    user_tool = {"type": "function", "function": {"name": "get_weather"}}
    entry, remaining = extract_web_search_tool([user_tool, {"type": "web_search"}])
    assert entry == {"type": "web_search"}
    assert remaining == [user_tool]


def test_web_search_does_not_match_code_execution() -> None:
    entry, _ = extract_web_search_tool([{"type": "code_execution"}])
    assert entry is None


def test_web_search_does_not_match_unrelated_prefix_boundary() -> None:
    entry, _ = extract_web_search_tool([{"type": "web_searcher"}])
    assert entry is None


def test_web_search_non_string_type_does_not_match() -> None:
    entry, _ = extract_web_search_tool([{"type": None}, {"type": 42}])
    assert entry is None


def test_web_search_carries_per_tool_config_through() -> None:
    entry, _ = extract_web_search_tool(
        [{"type": "web_search", "max_results": 3, "allowed_domains": ["docs.python.org"]}]
    )
    assert entry is not None
    assert entry["max_results"] == 3
    assert entry["allowed_domains"] == ["docs.python.org"]


def test_build_web_search_backend_uses_trimmed_env_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_ENGINES", " duckduckgo, , brave ")

    backend = build_web_search_backend(base_url="http://search.local", tool_entry={"type": "web_search"})

    assert backend._engines == ("duckduckgo", "brave")


def test_build_web_search_backend_ignores_blank_env_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GATEWAY_WEB_SEARCH_ENGINES", raising=False)
    default_backend = build_web_search_backend(base_url="http://search.local", tool_entry={"type": "web_search"})
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_ENGINES", " , ")

    backend = build_web_search_backend(base_url="http://search.local", tool_entry={"type": "web_search"})

    assert backend._engines == default_backend._engines


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("false", False),
        (" NO ", False),
        ("true", True),
        (" yes ", True),
        ("", True),
        ("maybe", True),
    ],
)
def test_build_web_search_backend_uses_shared_extract_env_bool_parsing(
    env_value: str,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_EXTRACT", env_value)

    backend = build_web_search_backend(base_url="http://search.local", tool_entry={"type": "web_search"})

    assert backend._extract_content is expected


def test_build_web_search_backend_normalizes_domain_lists() -> None:
    backend = build_web_search_backend(
        base_url="http://search.local",
        tool_entry={
            "type": "web_search",
            "allowed_domains": ["Docs.Python.org", 7],
            "blocked_domains": ["Example.com", False],
        },
    )

    assert backend._allowed_domains == ("docs.python.org", "7")
    assert backend._blocked_domains == ("example.com", "false")


def test_build_web_search_backend_uses_env_purpose_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_PURPOSE_HINT", "use search for current facts")

    backend = build_web_search_backend(base_url="http://search.local", tool_entry={"type": "web_search"})

    assert backend._purpose_hint == "use search for current facts"


def test_build_web_search_backend_tool_purpose_hint_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_PURPOSE_HINT", "env hint")

    backend = build_web_search_backend(
        base_url="http://search.local",
        tool_entry={"type": "web_search", "purpose_hint": "tool hint"},
    )

    assert backend._purpose_hint == "tool hint"


def test_build_web_search_backend_without_purpose_hint_uses_backend_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GATEWAY_WEB_SEARCH_PURPOSE_HINT", raising=False)

    backend = build_web_search_backend(base_url="http://search.local", tool_entry={"type": "web_search"})
    default_backend = WebSearchBackend(base_url="http://search.local")

    assert backend._purpose_hint == default_backend._purpose_hint


# --- route-level tool selection ----------------------------------------------


def test_resolve_chat_tool_selection_without_gateway_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GATEWAY_SANDBOX_URL", raising=False)
    monkeypatch.delenv("GATEWAY_WEB_SEARCH_URL", raising=False)

    selection = resolve_chat_tool_selection(
        tools=[{"type": "function", "function": {"name": "user_tool"}}],
        mcp_servers=None,
    )

    assert selection.use_sandbox is False
    assert selection.use_web_search is False
    assert selection.tools_extracted is False
    assert selection.remaining_user_tools == [{"type": "function", "function": {"name": "user_tool"}}]


def test_resolve_chat_tool_selection_requires_sandbox_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GATEWAY_SANDBOX_URL", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        resolve_chat_tool_selection(tools=[{"type": "code_execution"}], mcp_servers=None)

    assert exc_info.value.status_code == 400
    assert "no sandbox is configured" in str(exc_info.value.detail)


def test_resolve_chat_tool_selection_rejects_sandbox_with_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://sandbox.local")
    mcp_server = McpServerConfig.model_construct(name="tools", url="https://example.com/mcp")

    with pytest.raises(HTTPException) as exc_info:
        resolve_chat_tool_selection(tools=[{"type": "code_execution"}], mcp_servers=[mcp_server])

    assert exc_info.value.status_code == 400
    assert "code_execution and mcp_servers cannot be combined" in str(exc_info.value.detail)


def test_resolve_chat_tool_selection_allows_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://sandbox.local")

    selection = resolve_chat_tool_selection(tools=[{"type": "code_execution"}], mcp_servers=None)

    assert selection.use_sandbox is True
    assert selection.sandbox_url == "http://sandbox.local"
    assert selection.tools_extracted is True


def test_resolve_chat_tool_selection_requires_web_search_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GATEWAY_WEB_SEARCH_URL", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        resolve_chat_tool_selection(tools=[{"type": "web_search"}], mcp_servers=None)

    assert exc_info.value.status_code == 400
    assert "no search backend is configured" in str(exc_info.value.detail)


def test_resolve_chat_tool_selection_rejects_web_search_with_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://sandbox.local")
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://search.local")

    with pytest.raises(HTTPException) as exc_info:
        resolve_chat_tool_selection(
            tools=[{"type": "code_execution"}, {"type": "web_search"}],
            mcp_servers=None,
        )

    assert exc_info.value.status_code == 400
    assert "web_search cannot be combined with code_execution" in str(exc_info.value.detail)


def test_resolve_chat_tool_selection_allows_web_search_and_preserves_user_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://search.local")
    user_tool = {"type": "function", "function": {"name": "user_tool"}}

    selection = resolve_chat_tool_selection(tools=[user_tool, {"type": "web_search"}], mcp_servers=None)

    assert selection.use_web_search is True
    assert selection.web_search_url == "http://search.local"
    assert selection.remaining_user_tools == [user_tool]
    assert selection.tools_extracted is True


def test_chat_provider_request_fields_strip_gateway_metadata_and_preserve_user_tools() -> None:
    user_tool = {"type": "function", "function": {"name": "get_weather"}}
    request = ChatCompletionRequest(
        model="openai:gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        user="user-1",
        project_id="project-1",
        tags={"team": "platform"},
        tools=[{"type": "web_search"}, user_tool],
        tools_header="prefer external tools",
        max_tool_iterations=3,
    )

    fields = chat_provider_request_fields(
        request,
        tools_extracted=True,
        remaining_user_tools=[user_tool],
    )

    assert fields == {
        "model": "openai:gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [user_tool],
    }


def test_chat_provider_call_kwargs_preserves_request_and_attempt_model_precedence() -> None:
    call_kwargs = chat_provider_call_kwargs(
        {
            "api_key": "sk-test",
            "model": "provider-default",
            "temperature": 0.9,
            "timeout": 30,
        },
        {
            "model": "openai:gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
        },
        model="anthropic:claude-3-5-sonnet",
    )

    assert call_kwargs == {
        "api_key": "sk-test",
        "model": "anthropic:claude-3-5-sonnet",
        "temperature": 0.2,
        "timeout": 30,
        "messages": [{"role": "user", "content": "hi"}],
    }

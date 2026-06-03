"""Helpers for gateway-managed chat tools and request-field filtering."""

import os
from typing import Any

from gateway.log_config import logger
from gateway.services.routing_config_values import bool_config, comma_separated_string_list
from gateway.services.web_search_backend import WebSearchBackend

# Gateway-internal fields the provider SDKs (any-llm, anthropic, openai, ...)
# do not accept as ``acompletion`` kwargs.
_GATEWAY_INTERNAL_FIELDS = (
    "mcp_servers",
    "mcp_server_ids",
    "tools_header",
    "max_tool_iterations",
    "user",
    "project_id",
    "tags",
)


def strip_gateway_fields(
    fields: dict[str, Any],
    *,
    tools_extracted: bool = False,
    remaining_user_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Strip gateway-internal fields from a request payload before provider dispatch."""
    for key in _GATEWAY_INTERNAL_FIELDS:
        fields.pop(key, None)
    if not tools_extracted:
        return fields
    if remaining_user_tools:
        fields["tools"] = remaining_user_tools
        return fields
    fields.pop("tools", None)
    return fields


def resolve_sandbox_purpose_hint(sandbox_tool_entry: dict[str, Any] | None) -> str | None:
    """Resolve per-tool sandbox purpose hint, then environment fallback."""
    return (
        (sandbox_tool_entry.get("purpose_hint") if sandbox_tool_entry else None)
        or os.environ.get("GATEWAY_SANDBOX_PURPOSE_HINT")
        or None
    )


def _extract_first_matching_tool(
    tools: list[dict[str, Any]] | None,
    *,
    exact_types: tuple[str, ...],
    prefixes: tuple[str, ...] = (),
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first tool entry whose ``type`` matches the configured shapes."""
    if not tools:
        return None, tools
    entry: dict[str, Any] | None = None
    remaining: list[dict[str, Any]] = []
    for tool in tools:
        type_value = tool.get("type") if isinstance(tool, dict) else None
        if entry is None and isinstance(type_value, str) and (
            type_value in exact_types or any(type_value.startswith(prefix) for prefix in prefixes)
        ):
            entry = tool
        else:
            remaining.append(tool)
    return entry, (remaining or None)


def extract_code_execution_tool(
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first code-execution-style entry out of ``tools``."""
    return _extract_first_matching_tool(
        tools,
        exact_types=("code_execution", "code_interpreter"),
        prefixes=("code_execution_",),
    )


def extract_web_search_tool(
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first web-search-style entry out of ``tools``."""
    return _extract_first_matching_tool(
        tools,
        exact_types=("web_search",),
        prefixes=("web_search_",),
    )


def _web_search_env_max_results() -> int | None:
    max_env = os.environ.get("GATEWAY_WEB_SEARCH_MAX_RESULTS")
    if not max_env:
        return None
    try:
        parsed_max = int(max_env)
    except ValueError:
        logger.warning("GATEWAY_WEB_SEARCH_MAX_RESULTS=%r is not an int; ignoring", max_env)
        return None
    if parsed_max < 1:
        logger.warning("GATEWAY_WEB_SEARCH_MAX_RESULTS=%r is not >= 1; ignoring", max_env)
        return None
    return parsed_max


def build_web_search_backend(*, base_url: str, tool_entry: dict[str, Any]) -> WebSearchBackend:
    """Construct a WebSearchBackend honoring environment and per-tool config."""
    kwargs: dict[str, Any] = {"base_url": base_url}

    engines = tuple(comma_separated_string_list(os.environ.get("GATEWAY_WEB_SEARCH_ENGINES")))
    if engines:
        kwargs["engines"] = engines

    if (max_env := _web_search_env_max_results()) is not None:
        kwargs["max_results"] = max_env
    req_max = tool_entry.get("max_results")
    if isinstance(req_max, int) and req_max > 0:
        kwargs["max_results"] = req_max

    extract_env = os.environ.get("GATEWAY_WEB_SEARCH_EXTRACT")
    if extract_env is not None:
        kwargs["extract_content"] = bool_config(extract_env, True, coerce_strings=True)

    for key in ("allowed_domains", "blocked_domains"):
        domains = tool_entry.get(key)
        if isinstance(domains, list) and domains:
            kwargs[key] = tuple(str(domain) for domain in domains)

    purpose_hint = tool_entry.get("purpose_hint") or os.environ.get("GATEWAY_WEB_SEARCH_PURPOSE_HINT")
    if purpose_hint:
        kwargs["purpose_hint"] = purpose_hint

    return WebSearchBackend(**kwargs)

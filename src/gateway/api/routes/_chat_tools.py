import os
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, status

from gateway.models.mcp import McpServerConfig
from gateway.services.chat_tool_config import extract_code_execution_tool, extract_web_search_tool


@dataclass(frozen=True)
class ChatToolSelection:
    sandbox_tool_entry: dict[str, Any] | None
    sandbox_url: str | None
    use_sandbox: bool
    web_search_tool_entry: dict[str, Any] | None
    web_search_url: str | None
    use_web_search: bool
    remaining_user_tools: list[dict[str, Any]] | None

    @property
    def tools_extracted(self) -> bool:
        return self.sandbox_tool_entry is not None or self.web_search_tool_entry is not None


def resolve_chat_tool_selection(
    *,
    tools: list[dict[str, Any]] | None,
    mcp_servers: list[McpServerConfig] | None,
) -> ChatToolSelection:
    """Resolve gateway-managed chat tools and validate unsupported combinations."""
    sandbox_tool_entry, tools_after_sandbox = extract_code_execution_tool(tools)
    sandbox_url: str | None = os.environ.get("GATEWAY_SANDBOX_URL") or None
    use_sandbox = False
    if sandbox_tool_entry is not None:
        if sandbox_url is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "code_execution tool requested but no sandbox is configured on this gateway. "
                    "Set GATEWAY_SANDBOX_URL on the gateway, or remove code_execution from `tools`."
                ),
            )
        if mcp_servers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "code_execution and mcp_servers cannot be combined in the same request yet; "
                    "pick one. Multi-backend dispatch is a planned refinement."
                ),
            )
        use_sandbox = True

    web_search_tool_entry, remaining_user_tools = extract_web_search_tool(tools_after_sandbox)
    web_search_url: str | None = os.environ.get("GATEWAY_WEB_SEARCH_URL") or None
    use_web_search = False
    if web_search_tool_entry is not None:
        if web_search_url is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "web_search tool requested but no search backend is configured on this gateway. "
                    "Set GATEWAY_WEB_SEARCH_URL on the gateway, or remove web_search from `tools`."
                ),
            )
        if use_sandbox or mcp_servers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "web_search cannot be combined with code_execution or mcp_servers in the same request yet; "
                    "pick one."
                ),
            )
        use_web_search = True

    return ChatToolSelection(
        sandbox_tool_entry=sandbox_tool_entry,
        sandbox_url=sandbox_url,
        use_sandbox=use_sandbox,
        web_search_tool_entry=web_search_tool_entry,
        web_search_url=web_search_url,
        use_web_search=use_web_search,
        remaining_user_tools=remaining_user_tools,
    )

import uuid
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from gateway.models.mcp import McpServerConfig
from gateway.services.mcp_loop import MAX_TOOL_ITERATIONS_CAP
from gateway.services.routing_policy_service import (
    DEFAULT_ROUTE_TRACE_ENDPOINT,
    DEFAULT_ROUTING_MODEL,
    require_routing_model_selector,
)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    _route_trace_endpoint: str = PrivateAttr(default=DEFAULT_ROUTE_TRACE_ENDPOINT)

    model: str = DEFAULT_ROUTING_MODEL
    messages: list[dict[str, Any]] = Field(min_length=1)

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, v: Any) -> str:
        """Accept omitted/null/case-insensitive default_routing sentinels."""
        return require_routing_model_selector(v)

    @field_validator("messages")
    @classmethod
    def validate_message_structure(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, message in enumerate(v):
            if "role" not in message:
                msg = f"messages[{i}]: 'role' is required"
                raise ValueError(msg)
        return v

    user: str | None = None
    project_id: str | None = Field(default=None, description="Optional gateway project id for routing policy lookup")
    tags: dict[str, str] | None = Field(default=None, description="Optional trace tags for routing observability")
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    mcp_servers: list[McpServerConfig] | None = None
    mcp_server_ids: list[uuid.UUID] | None = None
    tools_header: str | None = Field(
        default=None,
        max_length=4000,
        description=(
            "Optional override for the lead-in that the gateway prepends before the "
            "per-tool hint block in the system message. Useful for expressing "
            "global tool-selection policy (e.g. 'prefer MCP tools over code_execution'). "
            "Falls back to GATEWAY_TOOLS_HEADER env, then to the built-in default."
        ),
    )
    max_tool_iterations: int | None = Field(default=None, ge=1, le=MAX_TOOL_ITERATIONS_CAP)

    @property
    def route_trace_endpoint(self) -> str:
        """Return the endpoint label used for standalone routing traces."""
        return self._route_trace_endpoint

    def set_route_trace_endpoint(self, endpoint: str) -> None:
        """Set the endpoint label used for standalone routing traces."""
        self._route_trace_endpoint = endpoint

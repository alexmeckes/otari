from typing import Any

from pydantic import BaseModel, Field


class MessagesRequest(BaseModel):
    """Anthropic Messages API-compatible request."""

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None

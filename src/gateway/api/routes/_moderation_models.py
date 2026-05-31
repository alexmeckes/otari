from typing import Any

from pydantic import BaseModel, Field


class ModerationRequest(BaseModel):
    """OpenAI-compatible moderation request."""

    model: str
    input: str | list[str] | list[dict[str, Any]] = Field(
        description="Text, list of texts, or list of content-part dicts to moderate",
    )
    user: str | None = None

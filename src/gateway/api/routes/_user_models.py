"""Request and response models for user routes."""

from typing import Any

from pydantic import BaseModel, Field

from gateway.api.routes._response_datetime import optional_datetime_isoformat
from gateway.models.entities import UsageLog, User


class CreateUserRequest(BaseModel):
    """Request model for creating a new user."""

    user_id: str = Field(description="Unique user identifier")
    alias: str | None = Field(default=None, description="Optional admin-facing alias")
    budget_id: str | None = Field(default=None, description="Optional budget ID")
    blocked: bool = Field(default=False, description="Whether user is blocked")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class UserResponse(BaseModel):
    """Response model for user information."""

    user_id: str
    alias: str | None
    spend: float
    budget_id: str | None
    budget_started_at: str | None
    next_budget_reset_at: str | None
    blocked: bool
    created_at: str
    updated_at: str
    metadata: dict[str, Any]

    @classmethod
    def from_model(cls, user: User) -> "UserResponse":
        return cls(
            user_id=user.user_id,
            alias=user.alias,
            spend=float(user.spend),
            budget_id=user.budget_id,
            budget_started_at=optional_datetime_isoformat(user.budget_started_at),
            next_budget_reset_at=optional_datetime_isoformat(user.next_budget_reset_at),
            blocked=bool(user.blocked),
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat(),
            metadata=user.metadata_dict(),
        )


class UpdateUserRequest(BaseModel):
    """Request model for updating a user."""

    alias: str | None = None
    budget_id: str | None = None
    blocked: bool | None = None
    metadata: dict[str, Any] | None = None


class UsageLogResponse(BaseModel):
    """Response model for usage log."""

    id: str
    user_id: str | None
    api_key_id: str | None
    project_id: str | None
    timestamp: str
    model: str
    provider: str | None
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost: float | None
    status: str
    error_message: str | None
    tags: dict[str, Any]

    @classmethod
    def from_model(cls, log: UsageLog) -> "UsageLogResponse":
        return cls(
            id=log.id,
            user_id=log.user_id,
            api_key_id=log.api_key_id,
            project_id=log.project_id,
            timestamp=log.timestamp.isoformat(),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cost=log.cost,
            status=log.status,
            error_message=log.error_message,
            tags=log.tag_dict(),
        )

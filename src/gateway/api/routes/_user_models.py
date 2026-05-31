"""Request and response models for user routes."""

from typing import Any

from pydantic import BaseModel, Field

from gateway.api.routes._response_datetime import datetime_isoformat, optional_datetime_isoformat
from gateway.api.routes._usage_models import UsageEntry
from gateway.models.entities import User


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
            created_at=datetime_isoformat(user.created_at),
            updated_at=datetime_isoformat(user.updated_at),
            metadata=user.metadata_dict(),
        )


class UpdateUserRequest(BaseModel):
    """Request model for updating a user."""

    alias: str | None = None
    budget_id: str | None = None
    blocked: bool | None = None
    metadata: dict[str, Any] | None = None


class UsageLogResponse(UsageEntry):
    """Response model for usage log."""

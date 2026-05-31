"""Request and response models for project routes."""

from typing import Any

from pydantic import BaseModel, Field

from gateway.api.routes._response_datetime import datetime_isoformat, optional_datetime_isoformat
from gateway.models.entities import Project


class CreateProjectRequest(BaseModel):
    """Request model for creating a project."""

    project_id: str | None = Field(default=None, min_length=1)
    name: str | None = None
    routing_policy_id: str | None = None
    budget_id: str | None = None
    blocked: bool = False
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateProjectRequest(BaseModel):
    """Request model for updating a project."""

    name: str | None = None
    routing_policy_id: str | None = None
    budget_id: str | None = None
    blocked: bool | None = None
    is_active: bool | None = None
    metadata: dict[str, Any] | None = None


class ProjectResponse(BaseModel):
    """Response model for project information."""

    project_id: str
    name: str | None
    routing_policy_id: str | None
    spend: float
    budget_id: str | None
    budget_started_at: str | None
    next_budget_reset_at: str | None
    blocked: bool
    is_active: bool
    metadata: dict[str, Any]
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, project: Project) -> "ProjectResponse":
        """Create a response from an ORM model."""
        return cls(
            project_id=project.project_id,
            name=project.name,
            routing_policy_id=project.routing_policy_id,
            spend=float(project.spend),
            budget_id=project.budget_id,
            budget_started_at=optional_datetime_isoformat(project.budget_started_at),
            next_budget_reset_at=optional_datetime_isoformat(project.next_budget_reset_at),
            blocked=bool(project.blocked),
            is_active=bool(project.is_active),
            metadata=project.metadata_dict(),
            created_at=datetime_isoformat(project.created_at),
            updated_at=datetime_isoformat(project.updated_at),
        )

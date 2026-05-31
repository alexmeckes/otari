from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.api.routes._database import commit_or_database_error, get_budget_or_404
from gateway.api.routes._project_models import CreateProjectRequest, ProjectResponse, UpdateProjectRequest
from gateway.models.entities import Budget, Project, RoutingPolicy
from gateway.services.budget_service import start_budget_period
from gateway.services.routing_policy_service import ACTIVE_ROUTING_POLICY_STATUS

router = APIRouter(prefix="/v1/projects", tags=["projects"])


async def _ensure_policy_exists(db: AsyncSession, policy_id: str) -> None:
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )
    if policy.status != ACTIVE_ROUTING_POLICY_STATUS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Routing policy '{policy_id}' is not active",
        )


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_project(
    request: CreateProjectRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectResponse:
    """Create a project."""
    if request.project_id:
        existing = await db.get(Project, request.project_id)
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Project '{request.project_id}' already exists",
            )
    if request.routing_policy_id:
        await _ensure_policy_exists(db, request.routing_policy_id)
    budget: Budget | None = None
    if request.budget_id:
        budget = await get_budget_or_404(db, request.budget_id)

    project_kwargs: dict[str, Any] = {
        "name": request.name,
        "routing_policy_id": request.routing_policy_id,
        "budget_id": request.budget_id,
        "blocked": request.blocked,
        "is_active": request.is_active,
        "metadata_": request.metadata,
    }
    if request.project_id is not None:
        project_kwargs["project_id"] = request.project_id
    project = Project(**project_kwargs)
    if budget is not None:
        start_budget_period(project, budget, datetime.now(UTC))
    db.add(project)
    await commit_or_database_error(db)
    await db.refresh(project)
    return ProjectResponse.from_model(project)


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_projects(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[ProjectResponse]:
    """List projects."""
    result = await db.execute(select(Project).order_by(Project.created_at.desc()).offset(skip).limit(limit))
    projects = result.scalars().all()
    return [ProjectResponse.from_model(project) for project in projects]


@router.get("/{project_id}", dependencies=[Depends(verify_master_key)])
async def get_project(
    project_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectResponse:
    """Get a project."""
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found",
        )
    return ProjectResponse.from_model(project)


@router.patch("/{project_id}", dependencies=[Depends(verify_master_key)])
async def update_project(
    project_id: str,
    request: UpdateProjectRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectResponse:
    """Update a project."""
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found",
        )

    payload = request.model_dump(exclude_unset=True)
    if "routing_policy_id" in payload and payload["routing_policy_id"] is not None:
        await _ensure_policy_exists(db, str(payload["routing_policy_id"]))
        project.routing_policy_id = str(payload["routing_policy_id"])
    elif "routing_policy_id" in payload:
        project.routing_policy_id = None
    if "budget_id" in payload and payload["budget_id"] is not None:
        budget = await get_budget_or_404(db, str(payload["budget_id"]))
        project.budget_id = str(payload["budget_id"])
        start_budget_period(project, budget, datetime.now(UTC))
    elif "budget_id" in payload:
        project.budget_id = None
        project.budget_started_at = None
        project.next_budget_reset_at = None
    if "name" in payload:
        project.name = payload["name"]
    if "blocked" in payload and payload["blocked"] is not None:
        project.blocked = bool(payload["blocked"])
    if "is_active" in payload and payload["is_active"] is not None:
        project.is_active = bool(payload["is_active"])
    if "metadata" in payload:
        project.metadata_ = dict(payload["metadata"] or {})

    await commit_or_database_error(db)
    await db.refresh(project)
    return ProjectResponse.from_model(project)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_master_key)],
)
async def delete_project(
    project_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a project."""
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found",
        )

    await db.delete(project)
    await commit_or_database_error(db)

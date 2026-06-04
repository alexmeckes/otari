from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.api.routes._database import commit_or_database_error, get_budget_or_404
from gateway.api.routes._user_models import CreateUserRequest, UpdateUserRequest, UsageLogResponse, UserResponse
from gateway.models.entities import APIKey, UsageLog, User
from gateway.repositories.users_repository import get_active_user, get_user_by_id
from gateway.services.budget_service import start_budget_period

router = APIRouter(prefix="/v1/users", tags=["users"])


async def _get_active_user_or_404(db: AsyncSession, user_id: str) -> User:
    user = await get_active_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )
    return user


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_user(
    request: CreateUserRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    """Create a new user."""
    existing_user = await get_user_by_id(db, request.user_id)
    if existing_user and existing_user.deleted_at is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with id '{request.user_id}' already exists",
        )

    budget = await get_budget_or_404(db, request.budget_id) if request.budget_id else None

    if existing_user and existing_user.deleted_at is not None:
        user = existing_user
        user.deleted_at = None
        user.spend = 0.0
        user.alias = request.alias
        user.budget_id = request.budget_id
        user.blocked = request.blocked
        user.metadata_ = request.metadata
        user.budget_started_at = None
        user.next_budget_reset_at = None
    else:
        user = User(
            user_id=request.user_id,
            alias=request.alias,
            budget_id=request.budget_id,
            blocked=request.blocked,
            metadata_=request.metadata,
        )
        db.add(user)

    if budget is not None:
        start_budget_period(user, budget)

    await commit_or_database_error(db)
    await db.refresh(user)

    return UserResponse.from_model(user)


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_users(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UserResponse]:
    """List all users with pagination."""
    result = await db.execute(select(User).where(User.deleted_at.is_(None)).offset(skip).limit(limit))
    users = result.scalars().all()

    return [UserResponse.from_model(user) for user in users]


@router.get("/{user_id}", dependencies=[Depends(verify_master_key)])
async def get_user(
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    """Get details of a specific user."""
    user = await _get_active_user_or_404(db, user_id)
    return UserResponse.from_model(user)


@router.patch("/{user_id}", dependencies=[Depends(verify_master_key)])
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    """Update a user."""
    user = await _get_active_user_or_404(db, user_id)
    if request.alias is not None:
        user.alias = request.alias
    if request.budget_id is not None:
        budget = await get_budget_or_404(db, request.budget_id)
        user.budget_id = request.budget_id
        start_budget_period(user, budget)
    if request.blocked is not None:
        user.blocked = request.blocked
    if request.metadata is not None:
        user.metadata_ = request.metadata

    await commit_or_database_error(db)
    await db.refresh(user)

    return UserResponse.from_model(user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_user(
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a user."""
    user = await _get_active_user_or_404(db, user_id)
    await db.execute(
        update(APIKey)
        .where(APIKey.user_id == user_id)
        .values(is_active=False)
        .execution_options(synchronize_session=False)
    )
    user.deleted_at = datetime.now(UTC)

    await commit_or_database_error(db)


@router.get("/{user_id}/usage", dependencies=[Depends(verify_master_key)])
async def get_user_usage(
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageLogResponse]:
    """Get usage history for a specific user."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )

    usage_result = await db.execute(
        select(UsageLog)
        .where(UsageLog.user_id == user_id)
        .order_by(UsageLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
    )
    usage_logs = usage_result.scalars().all()

    return [UsageLogResponse.from_model(log) for log in usage_logs]

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.api.routes._budget_models import (
    BudgetAlertResponse,
    BudgetResponse,
    CreateBudgetRequest,
    UpdateBudgetRequest,
    validate_tag_budget_shape,
)
from gateway.api.routes._database import commit_or_database_error, get_budget_or_404
from gateway.models.entities import Budget, BudgetAlert
from gateway.services.budget_alert_webhook_service import dispatch_budget_alert_webhook
from gateway.services.budget_service import (
    TAG_BUDGET_SCOPE,
    calculate_next_reset,
    normalize_alert_thresholds,
    normalize_alert_webhook_url,
)

router = APIRouter(prefix="/v1/budgets", tags=["budgets"])


def _budget_window_start(duration_sec: int | None) -> tuple[datetime, datetime | None]:
    now = datetime.now(UTC)
    next_reset_at = calculate_next_reset(now, duration_sec) if duration_sec else None
    return now, next_reset_at


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_budget(
    request: CreateBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Create a new budget."""
    validate_tag_budget_shape(request.scope_type, request.match_tags)
    budget_started_at = None
    next_budget_reset_at = None
    if request.scope_type == TAG_BUDGET_SCOPE:
        budget_started_at, next_budget_reset_at = _budget_window_start(request.budget_duration_sec)

    budget = Budget(
        max_budget=request.max_budget,
        budget_duration_sec=request.budget_duration_sec,
        scope_type=request.scope_type,
        match_tags=dict(request.match_tags or {}),
        alert_thresholds=normalize_alert_thresholds(request.alert_thresholds),
        alert_webhook_url=normalize_alert_webhook_url(request.alert_webhook_url),
        spend=0.0,
        budget_started_at=budget_started_at,
        next_budget_reset_at=next_budget_reset_at,
        blocked=request.blocked,
        is_active=request.is_active,
    )

    db.add(budget)
    await commit_or_database_error(db)
    await db.refresh(budget)

    return BudgetResponse.from_model(budget)


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_budgets(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[BudgetResponse]:
    """List all budgets with pagination."""
    result = await db.execute(select(Budget).offset(skip).limit(limit))
    budgets = result.scalars().all()

    return [BudgetResponse.from_model(budget) for budget in budgets]


@router.get("/alerts", dependencies=[Depends(verify_master_key)])
async def list_budget_alerts(
    db: Annotated[AsyncSession, Depends(get_db)],
    budget_id: Annotated[str | None, Query()] = None,
    scope_type: Annotated[str | None, Query()] = None,
    scope_id: Annotated[str | None, Query()] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[BudgetAlertResponse]:
    """List budget alert events."""
    stmt = select(BudgetAlert)
    if budget_id is not None:
        stmt = stmt.where(BudgetAlert.budget_id == budget_id)
    if scope_type is not None:
        stmt = stmt.where(BudgetAlert.scope_type == scope_type)
    if scope_id is not None:
        stmt = stmt.where(BudgetAlert.scope_id == scope_id)
    stmt = stmt.order_by(BudgetAlert.created_at.desc(), BudgetAlert.id.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return [BudgetAlertResponse.from_model(alert) for alert in result.scalars().all()]


@router.post("/alerts/{alert_id}/deliver", dependencies=[Depends(verify_master_key)])
async def deliver_budget_alert(
    alert_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetAlertResponse:
    """Retry webhook delivery for a budget alert event."""
    result = await db.execute(select(BudgetAlert.id).where(BudgetAlert.id == alert_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget alert with id '{alert_id}' not found",
        )
    await dispatch_budget_alert_webhook(alert_id)
    refreshed = await db.get(BudgetAlert, alert_id)
    if refreshed is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget alert with id '{alert_id}' not found",
        )
    return BudgetAlertResponse.from_model(refreshed)


@router.get("/{budget_id}/alerts", dependencies=[Depends(verify_master_key)])
async def list_alerts_for_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[BudgetAlertResponse]:
    """List alert events for a specific budget."""
    await get_budget_or_404(db, budget_id)
    stmt = (
        select(BudgetAlert)
        .where(BudgetAlert.budget_id == budget_id)
        .order_by(BudgetAlert.created_at.desc(), BudgetAlert.id.desc())
        .offset(skip)
        .limit(limit)
    )
    alerts = await db.execute(stmt)
    return [BudgetAlertResponse.from_model(alert) for alert in alerts.scalars().all()]


@router.get("/{budget_id}", dependencies=[Depends(verify_master_key)])
async def get_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Get details of a specific budget."""
    budget = await get_budget_or_404(db, budget_id)
    return BudgetResponse.from_model(budget)


@router.patch("/{budget_id}", dependencies=[Depends(verify_master_key)])
async def update_budget(
    budget_id: str,
    request: UpdateBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Update a budget."""
    budget = await get_budget_or_404(db, budget_id)

    if request.max_budget is not None:
        budget.max_budget = request.max_budget
    if request.budget_duration_sec is not None:
        budget.budget_duration_sec = request.budget_duration_sec
    if request.scope_type is not None:
        budget.scope_type = request.scope_type
    if request.match_tags is not None:
        budget.match_tags = dict(request.match_tags)
    if "alert_thresholds" in request.model_fields_set:
        budget.alert_thresholds = normalize_alert_thresholds(request.alert_thresholds)
    if "alert_webhook_url" in request.model_fields_set:
        budget.alert_webhook_url = normalize_alert_webhook_url(request.alert_webhook_url)
    validate_tag_budget_shape(budget.scope_type, budget.match_tags)
    if request.spend is not None:
        budget.spend = request.spend
    if request.blocked is not None:
        budget.blocked = request.blocked
    if request.is_active is not None:
        budget.is_active = request.is_active
    if budget.scope_type == TAG_BUDGET_SCOPE and budget.budget_started_at is None:
        budget.budget_started_at, budget.next_budget_reset_at = _budget_window_start(budget.budget_duration_sec)

    await commit_or_database_error(db)
    await db.refresh(budget)

    return BudgetResponse.from_model(budget)


@router.delete("/{budget_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a budget."""
    budget = await get_budget_or_404(db, budget_id)

    await db.delete(budget)
    await commit_or_database_error(db)

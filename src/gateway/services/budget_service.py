from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from any_llm.exceptions import AnyLLMError
from fastapi import HTTPException, status
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.metrics import record_budget_exceeded
from gateway.models.entities import Budget, BudgetAlert, Project, User
from gateway.repositories.budgets_repository import get_budget_by_id
from gateway.repositories.projects_repository import get_project_by_id
from gateway.repositories.users_repository import get_active_user
from gateway.services import budget_alerts as _budget_alerts
from gateway.services import budget_periods as _budget_periods
from gateway.services import budget_tags as _budget_tags
from gateway.services.budget_reset_logs import new_budget_reset_log
from gateway.services.pricing_service import find_model_pricing, split_pricing_model_ref

TAG_BUDGET_SCOPE = _budget_tags.TAG_BUDGET_SCOPE
BUDGET_ALERT_SCOPE_PROJECT = _budget_alerts.BUDGET_ALERT_SCOPE_PROJECT
BUDGET_ALERT_SCOPE_USER = _budget_alerts.BUDGET_ALERT_SCOPE_USER
normalize_alert_thresholds = _budget_alerts.normalize_alert_thresholds
normalize_alert_webhook_url = _budget_alerts.normalize_alert_webhook_url
record_budget_alerts = _budget_alerts.record_budget_alerts
record_project_budget_alerts_after_spend = _budget_alerts.record_project_budget_alerts_after_spend
record_user_budget_alerts_after_spend = _budget_alerts.record_user_budget_alerts_after_spend
budget_matches_tags = _budget_tags.budget_matches_tags
reset_tag_budget = _budget_tags.reset_tag_budget
_tag_scope_id = _budget_tags.tag_scope_id
_matching_tag_budgets = _budget_tags.matching_tag_budgets
_cas_reset_tag_budget = _budget_tags.cas_reset_tag_budget
_normalize_budget_strategy = _budget_tags.normalize_budget_strategy
calculate_next_reset = _budget_periods.calculate_next_reset
_as_utc = _budget_periods.as_utc


def start_budget_period(subject: User | Project, budget: Budget, start: datetime | None = None) -> None:
    subject.budget_started_at, subject.next_budget_reset_at = _budget_periods.budget_period_window(
        budget.budget_duration_sec,
        start,
    )


async def reset_user_budget(db: AsyncSession, user: User, budget: Budget, now: datetime) -> None:
    """Reset user's budget spend and schedule next reset."""

    previous_spend = float(user.spend)
    user_id_str = user.user_id

    user.spend = 0.0
    start_budget_period(user, budget, now)

    reset_log = new_budget_reset_log(
        user,
        budget_id=budget.budget_id,
        previous_spend=previous_spend,
        reset_at=now,
        next_reset_at=user.next_budget_reset_at,
    )
    db.add(reset_log)

    try:
        await db.commit()
        await db.refresh(user)
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error("Failed to commit budget reset for user '%s': %s", user_id_str, e)
        raise


async def _cas_reset_user_budget(db: AsyncSession, user: User, budget: Budget, now: datetime) -> User:
    _, next_reset_at = _budget_periods.budget_period_window(budget.budget_duration_sec, now)

    result = await db.execute(
        update(User)
        .where(
            User.user_id == user.user_id,
            User.deleted_at.is_(None),
            User.next_budget_reset_at.is_not(None),
            User.next_budget_reset_at <= now,
        )
        .values(
            spend=0.0,
            budget_started_at=now,
            next_budget_reset_at=next_reset_at,
        )
        .execution_options(synchronize_session=False)
    )

    rowcount = getattr(result, "rowcount", 0)
    if rowcount and rowcount > 0:
        reset_log = new_budget_reset_log(
            user,
            budget_id=budget.budget_id,
            previous_spend=float(user.spend),
            reset_at=now,
            next_reset_at=next_reset_at,
        )
        db.add(reset_log)
        try:
            await db.commit()
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error("Failed to commit CAS budget reset for user '%s': %s", user.user_id, e)
            raise
        refreshed = await get_active_user(db, user.user_id)
        return refreshed or user

    await db.rollback()
    return user


async def reset_project_budget(db: AsyncSession, project: Project, budget: Budget, now: datetime) -> None:
    """Reset project's budget spend and schedule next reset."""

    previous_spend = float(project.spend)
    project_id_str = project.project_id

    project.spend = 0.0
    start_budget_period(project, budget, now)

    reset_log = new_budget_reset_log(
        project,
        budget_id=budget.budget_id,
        previous_spend=previous_spend,
        reset_at=now,
        next_reset_at=project.next_budget_reset_at,
    )
    db.add(reset_log)

    try:
        await db.commit()
        await db.refresh(project)
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error("Failed to commit budget reset for project '%s': %s", project_id_str, e)
        raise


async def _cas_reset_project_budget(db: AsyncSession, project: Project, budget: Budget, now: datetime) -> Project:
    _, next_reset_at = _budget_periods.budget_period_window(budget.budget_duration_sec, now)

    result = await db.execute(
        update(Project)
        .where(
            Project.project_id == project.project_id,
            Project.next_budget_reset_at.is_not(None),
            Project.next_budget_reset_at <= now,
        )
        .values(
            spend=0.0,
            budget_started_at=now,
            next_budget_reset_at=next_reset_at,
        )
        .execution_options(synchronize_session=False)
    )

    rowcount = getattr(result, "rowcount", 0)
    if rowcount and rowcount > 0:
        reset_log = new_budget_reset_log(
            project,
            budget_id=budget.budget_id,
            previous_spend=float(project.spend),
            reset_at=now,
            next_reset_at=next_reset_at,
        )
        db.add(reset_log)
        try:
            await db.commit()
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error("Failed to commit CAS budget reset for project '%s': %s", project.project_id, e)
            raise
        refreshed = await get_project_by_id(db, project.project_id)
        return refreshed or project

    await db.rollback()
    return project


async def validate_tag_budgets(
    db: AsyncSession,
    tags: dict[str, Any] | None,
    model: str | None = None,
    *,
    strategy: str = "for_update",
) -> list[Budget]:
    return await _budget_tags.validate_tag_budgets(
        db,
        tags,
        model,
        strategy=strategy,
        is_model_free=_is_model_free,
    )


async def increment_matching_tag_budget_spend(
    db: AsyncSession,
    *,
    tags: dict[str, Any] | None,
    cost: float | None,
    metadata: dict[str, Any] | None = None,
) -> list[BudgetAlert]:
    return await _budget_tags.increment_matching_tag_budget_spend(
        db,
        tags=tags,
        cost=cost,
        metadata=metadata,
    )


async def validate_user_budget(
    db: AsyncSession,
    user_id: str,
    model: str | None = None,
    *,
    strategy: str = "for_update",
) -> User:
    """Validate user exists, is not blocked, and has available budget.

    Args:
        db: Database session
        user_id: User identifier
        model: Optional model identifier (e.g., "provider:model" or "provider/model") to check if it's a free model

    Returns:
        User object if validation passes

    Raises:
        HTTPException: If user is blocked, doesn't exist, or exceeded budget

    """
    normalized_strategy = _normalize_budget_strategy(strategy)

    lock_for_update = normalized_strategy == "for_update"
    user = await get_active_user(db, user_id, for_update=lock_for_update)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found",
        )

    if user.blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' is blocked",
        )

    if normalized_strategy == "disabled" or not user.budget_id:
        return user

    budget = await get_budget_by_id(db, user.budget_id)
    if not budget:
        return user

    now = datetime.now(UTC)
    if _budget_periods.budget_reset_due(user.next_budget_reset_at, now):
        if normalized_strategy == "cas":
            user = await _cas_reset_user_budget(db, user, budget, now)
        else:
            await reset_user_budget(db, user, budget, now)

    if budget.max_budget is not None and user.spend >= budget.max_budget:
        if model and await _is_model_free(db, model):
            return user
        record_budget_exceeded()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' has exceeded budget limit",
        )

    return user


async def validate_project_budget(
    db: AsyncSession,
    project_id: str,
    model: str | None = None,
    *,
    strategy: str = "for_update",
) -> Project:
    """Validate project exists, is active, is not blocked, and has available budget."""

    normalized_strategy = _normalize_budget_strategy(strategy)

    project = await get_project_by_id(db, project_id, for_update=normalized_strategy == "for_update")

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found",
        )

    if not project.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Project '{project_id}' is inactive",
        )

    if project.blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Project '{project_id}' is blocked",
        )

    if normalized_strategy == "disabled" or not project.budget_id:
        return project

    budget = await get_budget_by_id(db, project.budget_id)
    if not budget:
        return project

    now = datetime.now(UTC)
    if _budget_periods.budget_reset_due(project.next_budget_reset_at, now):
        if normalized_strategy == "cas":
            project = await _cas_reset_project_budget(db, project, budget, now)
        else:
            await reset_project_budget(db, project, budget, now)

    if budget.max_budget is not None and project.spend >= budget.max_budget:
        if model and await _is_model_free(db, model):
            return project
        record_budget_exceeded()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Project '{project_id}' has exceeded budget limit",
        )

    return project


async def _is_model_free(db: AsyncSession, model: str) -> bool:
    """Check if a model is free (both input and output prices are 0).

    Args:
        db: Database session
        model: Model identifier (e.g., "provider:model" or "provider/model")

    Returns:
        True if the model is free, False otherwise or if pricing not found

    """
    try:
        provider, model_name = split_pricing_model_ref(model)
        pricing = await find_model_pricing(db, provider, model_name)
        if pricing:
            return pricing.input_price_per_million == 0 and pricing.output_price_per_million == 0
    except (AnyLLMError, ValueError, SQLAlchemyError) as e:
        logger.warning("Failed to determine provider pricing: %s", e)

    return False

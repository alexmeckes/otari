from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.metrics import record_budget_exceeded
from gateway.models.entities import Budget, BudgetAlert
from gateway.repositories.budgets_repository import get_budget_by_id
from gateway.services.budget_alerts import record_budget_alerts
from gateway.services.budget_periods import as_utc, budget_period_window
from gateway.services.budget_reset_logs import new_budget_reset_log

TAG_BUDGET_SCOPE = "tag"

IsModelFree = Callable[[AsyncSession, str], Awaitable[bool]]


def normalize_budget_strategy(strategy: str) -> str:
    normalized_strategy = strategy or "for_update"
    normalized_strategy = normalized_strategy.strip().lower()
    if normalized_strategy not in {"for_update", "cas", "disabled"}:
        return "for_update"
    return normalized_strategy


def tag_scope_id(budget: Budget) -> str | None:
    match_tags = budget.match_tag_dict()
    if not match_tags:
        return None
    return ",".join(f"{key}={match_tags[key]}" for key in sorted(match_tags))


def _request_tag_dict(tags: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(tags, dict):
        return tags
    return {}


def budget_matches_tags(budget: Budget, tags: dict[str, Any] | None) -> bool:
    """Return whether a tag-scoped budget applies to a request's tags."""
    if budget.scope_type != TAG_BUDGET_SCOPE or not budget.is_active:
        return False
    match_tags = budget.match_tag_dict()
    if not match_tags:
        return False
    request_tags = _request_tag_dict(tags)
    return all(str(request_tags.get(key)) == str(value) for key, value in match_tags.items())


async def matching_tag_budgets(
    db: AsyncSession,
    tags: dict[str, Any] | None,
    *,
    for_update: bool = False,
) -> list[Budget]:
    request_tags = _request_tag_dict(tags)
    if not request_tags:
        return []
    stmt = select(Budget).where(Budget.scope_type == TAG_BUDGET_SCOPE, Budget.is_active.is_(True))
    if for_update:
        stmt = stmt.with_for_update()
    result = await db.execute(stmt)
    return [budget for budget in result.scalars().all() if budget_matches_tags(budget, request_tags)]


async def reset_tag_budget(db: AsyncSession, budget: Budget, now: datetime) -> None:
    """Reset a tag-scoped budget group's spend and schedule next reset."""
    previous_spend = float(budget.spend)
    budget.spend = 0.0
    budget.budget_started_at, budget.next_budget_reset_at = budget_period_window(budget.budget_duration_sec, now)
    db.add(
        new_budget_reset_log(
            budget,
            previous_spend=previous_spend,
            reset_at=now,
            next_reset_at=budget.next_budget_reset_at,
        )
    )
    try:
        await db.commit()
        await db.refresh(budget)
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error("Failed to commit budget reset for tag budget '%s': %s", budget.budget_id, e)
        raise


async def cas_reset_tag_budget(db: AsyncSession, budget: Budget, now: datetime) -> Budget:
    _, next_reset_at = budget_period_window(budget.budget_duration_sec, now)
    result = await db.execute(
        update(Budget)
        .where(
            Budget.budget_id == budget.budget_id,
            Budget.scope_type == TAG_BUDGET_SCOPE,
            Budget.next_budget_reset_at.is_not(None),
            Budget.next_budget_reset_at <= now,
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
        db.add(
            new_budget_reset_log(
                budget,
                previous_spend=float(budget.spend),
                reset_at=now,
                next_reset_at=next_reset_at,
            )
        )
        try:
            await db.commit()
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error("Failed to commit CAS budget reset for tag budget '%s': %s", budget.budget_id, e)
            raise
        refreshed = await get_budget_by_id(db, budget.budget_id)
        return refreshed or budget

    await db.rollback()
    return budget


async def validate_tag_budgets(
    db: AsyncSession,
    tags: dict[str, Any] | None,
    model: str | None = None,
    *,
    strategy: str = "for_update",
    is_model_free: IsModelFree,
) -> list[Budget]:
    """Validate matching tag-scoped budgets have available spend."""
    normalized_strategy = normalize_budget_strategy(strategy)
    if normalized_strategy == "disabled":
        return []

    matching_budgets = await matching_tag_budgets(
        db,
        tags,
        for_update=normalized_strategy == "for_update",
    )
    if not matching_budgets:
        return []

    now = datetime.now(UTC)
    validated: list[Budget] = []
    for budget in matching_budgets:
        if budget.blocked:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Budget group '{budget.budget_id}' is blocked",
            )

        next_budget_reset_at = as_utc(budget.next_budget_reset_at)
        if next_budget_reset_at and now >= next_budget_reset_at:
            if normalized_strategy == "cas":
                budget = await cas_reset_tag_budget(db, budget, now)
            else:
                await reset_tag_budget(db, budget, now)

        if budget.max_budget is not None and budget.spend >= budget.max_budget:
            if model and await is_model_free(db, model):
                validated.append(budget)
                continue
            record_budget_exceeded()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Budget group '{budget.budget_id}' has exceeded budget limit",
            )
        validated.append(budget)
    return validated


async def increment_matching_tag_budget_spend(
    db: AsyncSession,
    *,
    tags: dict[str, Any] | None,
    cost: float | None,
    metadata: dict[str, Any] | None = None,
) -> list[BudgetAlert]:
    """Increment spend for active tag-scoped budgets matching usage tags."""
    if not cost or cost <= 0:
        return []
    matching_budgets = await matching_tag_budgets(db, tags)
    if not matching_budgets:
        return []
    budget_ids = [budget.budget_id for budget in matching_budgets]
    await db.execute(
        update(Budget)
        .where(Budget.budget_id.in_(budget_ids))
        .values(spend=Budget.spend + cost)
        .execution_options(synchronize_session=False)
    )
    updated_result = await db.execute(
        select(Budget)
        .where(Budget.budget_id.in_(budget_ids))
        .execution_options(populate_existing=True)
    )
    created_alerts: list[BudgetAlert] = []
    for budget in updated_result.scalars().all():
        alerts = await record_budget_alerts(
            db,
            budget=budget,
            scope_type=TAG_BUDGET_SCOPE,
            scope_id=tag_scope_id(budget),
            spend=float(budget.spend),
            period_start=budget.budget_started_at,
            metadata=metadata,
        )
        created_alerts.extend(alerts)
    return created_alerts

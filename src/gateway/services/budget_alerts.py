from datetime import datetime
from typing import Any

from pydantic import AnyHttpUrl, TypeAdapter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.models.entities import Budget, BudgetAlert, Project
from gateway.repositories.budgets_repository import get_budget_by_id
from gateway.repositories.users_repository import get_active_user

BUDGET_ALERT_SCOPE_PROJECT = "project"
BUDGET_ALERT_SCOPE_USER = "user"

_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)


def normalize_alert_thresholds(value: Any) -> list[float]:
    """Normalize budget alert thresholds as sorted spend ratios."""
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise ValueError("alert_thresholds must be a list of ratios between 0 and 1")

    thresholds: set[float] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError("alert_thresholds must contain only numeric ratios")
        threshold = float(item)
        if threshold <= 0 or threshold > 1:
            raise ValueError("alert_thresholds entries must be greater than 0 and less than or equal to 1")
        thresholds.add(threshold)
    return sorted(thresholds)


def normalize_alert_webhook_url(value: str | None) -> str | None:
    """Normalize and validate an optional budget alert webhook URL."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return str(_HTTP_URL_ADAPTER.validate_python(stripped))
    except ValueError as exc:
        raise ValueError("alert_webhook_url must be a valid http or https URL") from exc


async def record_budget_alerts(
    db: AsyncSession,
    *,
    budget: Budget,
    scope_type: str,
    scope_id: str | None,
    spend: float,
    period_start: datetime | None,
    metadata: dict[str, Any] | None = None,
) -> list[BudgetAlert]:
    """Record newly crossed alert thresholds for a budget scope."""
    max_budget = budget.max_budget
    if max_budget is None or max_budget <= 0 or spend <= 0:
        return []

    thresholds = normalize_alert_thresholds(budget.alert_thresholds)
    if not thresholds:
        return []

    created: list[BudgetAlert] = []
    for threshold in thresholds:
        if spend < max_budget * threshold:
            continue

        stmt = select(BudgetAlert).where(
            BudgetAlert.budget_id == budget.budget_id,
            BudgetAlert.scope_type == scope_type,
            BudgetAlert.scope_id == scope_id,
            BudgetAlert.threshold == threshold,
        )
        if period_start is None:
            stmt = stmt.where(BudgetAlert.budget_period_start.is_(None))
        else:
            stmt = stmt.where(BudgetAlert.budget_period_start == period_start)

        existing = await db.execute(stmt.limit(1))
        if existing.scalar_one_or_none() is not None:
            continue

        alert = BudgetAlert(
            budget_id=budget.budget_id,
            scope_type=scope_type,
            scope_id=scope_id,
            threshold=threshold,
            spend=spend,
            max_budget=max_budget,
            budget_period_start=period_start,
            webhook_url=budget.alert_webhook_url,
            delivery_status="pending" if budget.alert_webhook_url else "not_configured",
            metadata_=metadata or {},
        )
        db.add(alert)
        created.append(alert)
        logger.warning(
            "Budget alert %.0f%% crossed for %s '%s' on budget '%s' (spend %.6f / %.6f)",
            threshold * 100,
            scope_type,
            scope_id,
            budget.budget_id,
            spend,
            max_budget,
        )

    return created


async def record_user_budget_alerts_after_spend(
    db: AsyncSession,
    *,
    user_id: str,
    metadata: dict[str, Any] | None = None,
) -> list[BudgetAlert]:
    """Record alert events for a user's budget after usage spend increments."""
    user = await get_active_user(db, user_id)
    if user is None or not user.budget_id:
        return []
    budget = await get_budget_by_id(db, user.budget_id)
    if budget is None:
        return []
    return await record_budget_alerts(
        db,
        budget=budget,
        scope_type=BUDGET_ALERT_SCOPE_USER,
        scope_id=user.user_id,
        spend=float(user.spend),
        period_start=user.budget_started_at,
        metadata=metadata,
    )


async def record_project_budget_alerts_after_spend(
    db: AsyncSession,
    *,
    project_id: str,
    metadata: dict[str, Any] | None = None,
) -> list[BudgetAlert]:
    """Record alert events for a project's budget after usage spend increments."""
    result = await db.execute(select(Project).where(Project.project_id == project_id))
    project = result.scalar_one_or_none()
    if project is None or not project.budget_id:
        return []
    budget = await get_budget_by_id(db, project.budget_id)
    if budget is None:
        return []
    return await record_budget_alerts(
        db,
        budget=budget,
        scope_type=BUDGET_ALERT_SCOPE_PROJECT,
        scope_id=project.project_id,
        spend=float(project.spend),
        period_start=project.budget_started_at,
        metadata=metadata,
    )

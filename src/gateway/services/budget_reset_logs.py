from datetime import datetime

from gateway.models.entities import Budget, BudgetResetLog, Project, User


def new_budget_reset_log(
    subject: Budget | Project | User,
    *,
    budget_id: str | None = None,
    previous_spend: float,
    reset_at: datetime,
    next_reset_at: datetime | None,
) -> BudgetResetLog:
    """Create a budget reset log for an entity or tag-scoped budget."""

    if isinstance(subject, Budget):
        return BudgetResetLog(
            budget_id=subject.budget_id,
            previous_spend=previous_spend,
            reset_at=reset_at,
            next_reset_at=next_reset_at,
        )
    if budget_id is None:
        msg = "budget_id is required for user and project reset logs"
        raise ValueError(msg)
    if isinstance(subject, User):
        return BudgetResetLog(
            user_id=subject.user_id,
            budget_id=budget_id,
            previous_spend=previous_spend,
            reset_at=reset_at,
            next_reset_at=next_reset_at,
        )
    return BudgetResetLog(
        project_id=subject.project_id,
        budget_id=budget_id,
        previous_spend=previous_spend,
        reset_at=reset_at,
        next_reset_at=next_reset_at,
    )

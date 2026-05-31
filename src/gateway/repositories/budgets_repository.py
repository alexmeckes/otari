from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import Budget


async def get_budget_by_id(db: AsyncSession, budget_id: str) -> Budget | None:
    """Query a budget by budget_id."""

    result = await db.execute(select(Budget).where(Budget.budget_id == budget_id))
    return result.scalar_one_or_none()

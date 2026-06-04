from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.services.budget_service import validate_project_budget, validate_tag_budgets, validate_user_budget


async def release_budget_locks(db: AsyncSession, strategy: str) -> None:
    if strategy == "for_update":
        await db.rollback()


async def validate_user_request_budget(
    db: AsyncSession,
    user_id: str,
    model: str,
    *,
    strategy: str,
) -> None:
    _ = await validate_user_budget(db, user_id, model, strategy=strategy)
    await release_budget_locks(db, strategy)


async def validate_scoped_request_budgets(
    db: AsyncSession,
    *,
    user_id: str,
    model: str,
    project_id: str | None = None,
    tags: dict[str, Any] | None = None,
    strategy: str,
) -> None:
    _ = await validate_user_budget(db, user_id, model, strategy=strategy)
    if project_id is not None:
        _ = await validate_project_budget(db, project_id, model, strategy=strategy)
    _ = await validate_tag_budgets(db, tags, model, strategy=strategy)
    await release_budget_locks(db, strategy)

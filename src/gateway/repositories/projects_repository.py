from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import Project


async def get_project_by_id(db: AsyncSession, project_id: str, *, for_update: bool = False) -> Project | None:
    """Query a project by project_id."""

    stmt = select(Project).where(Project.project_id == project_id)
    if for_update:
        stmt = stmt.with_for_update()
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

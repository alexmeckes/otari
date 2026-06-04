from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import APIKey


async def get_api_key_by_id(db: AsyncSession, key_id: str) -> APIKey | None:
    """Query an API key by id."""

    result = await db.execute(select(APIKey).where(APIKey.id == key_id))
    return result.scalar_one_or_none()


async def get_api_key_by_hash(db: AsyncSession, key_hash: str) -> APIKey | None:
    """Query an API key by its hashed token value."""

    result = await db.execute(select(APIKey).where(APIKey.key_hash == key_hash))
    return result.scalar_one_or_none()

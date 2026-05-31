from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.repositories.users_repository import get_active_user, get_user_by_id


@pytest.mark.asyncio
async def test_get_user_by_id_queries_including_deleted_users() -> None:
    db = AsyncMock()
    expected = object()
    result = MagicMock()
    result.scalar_one_or_none.return_value = expected
    db.execute = AsyncMock(return_value=result)

    user = await get_user_by_id(db, "user-0")

    assert user is expected
    executed_stmt = db.execute.call_args.args[0]
    where_criteria = [str(criterion) for criterion in executed_stmt._where_criteria]
    assert where_criteria == ["users.user_id = :user_id_1"]


@pytest.mark.asyncio
async def test_get_active_user_applies_for_update_when_requested() -> None:
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = object()
    db.execute = AsyncMock(return_value=result)

    await get_active_user(db, "user-1", for_update=True)

    executed_stmt = db.execute.call_args.args[0]
    assert executed_stmt._for_update_arg is not None


@pytest.mark.asyncio
async def test_get_active_user_skips_for_update_when_not_requested() -> None:
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = object()
    db.execute = AsyncMock(return_value=result)

    await get_active_user(db, "user-2", for_update=False)

    executed_stmt = db.execute.call_args.args[0]
    assert executed_stmt._for_update_arg is None

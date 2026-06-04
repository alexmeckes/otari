from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.repositories.projects_repository import get_project_by_id


@pytest.mark.asyncio
async def test_get_project_by_id_applies_for_update_when_requested() -> None:
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = object()
    db.execute = AsyncMock(return_value=result)

    await get_project_by_id(db, "project-1", for_update=True)

    executed_stmt = db.execute.call_args.args[0]
    assert executed_stmt._for_update_arg is not None


@pytest.mark.asyncio
async def test_get_project_by_id_skips_for_update_when_not_requested() -> None:
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = object()
    db.execute = AsyncMock(return_value=result)

    await get_project_by_id(db, "project-2", for_update=False)

    executed_stmt = db.execute.call_args.args[0]
    assert executed_stmt._for_update_arg is None

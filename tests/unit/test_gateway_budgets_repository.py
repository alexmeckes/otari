from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.repositories.budgets_repository import get_budget_by_id


@pytest.mark.asyncio
async def test_get_budget_by_id_queries_budget_id() -> None:
    db = AsyncMock()
    expected = object()
    result = MagicMock()
    result.scalar_one_or_none.return_value = expected
    db.execute = AsyncMock(return_value=result)

    budget = await get_budget_by_id(db, "budget-1")

    assert budget is expected
    executed_stmt = db.execute.call_args.args[0]
    where_criteria = [str(criterion) for criterion in executed_stmt._where_criteria]
    assert where_criteria == ["budgets.budget_id = :budget_id_1"]

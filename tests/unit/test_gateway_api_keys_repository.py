from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.repositories.api_keys_repository import get_api_key_by_hash, get_api_key_by_id


@pytest.mark.asyncio
async def test_get_api_key_by_id_queries_id() -> None:
    db = AsyncMock()
    expected = object()
    result = MagicMock()
    result.scalar_one_or_none.return_value = expected
    db.execute = AsyncMock(return_value=result)

    key = await get_api_key_by_id(db, "key-1")

    assert key is expected
    executed_stmt = db.execute.call_args.args[0]
    where_criteria = [str(criterion) for criterion in executed_stmt._where_criteria]
    assert where_criteria == ["api_keys.id = :id_1"]


@pytest.mark.asyncio
async def test_get_api_key_by_hash_queries_key_hash() -> None:
    db = AsyncMock()
    expected = object()
    result = MagicMock()
    result.scalar_one_or_none.return_value = expected
    db.execute = AsyncMock(return_value=result)

    key = await get_api_key_by_hash(db, "hash-1")

    assert key is expected
    executed_stmt = db.execute.call_args.args[0]
    where_criteria = [str(criterion) for criterion in executed_stmt._where_criteria]
    assert where_criteria == ["api_keys.key_hash = :key_hash_1"]

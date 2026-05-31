from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.services import budget_service


@pytest.mark.asyncio
async def test_is_model_free_uses_pricing_model_ref_splitter(monkeypatch: pytest.MonkeyPatch) -> None:
    db = object()
    pricing = SimpleNamespace(input_price_per_million=0.0, output_price_per_million=0.0)
    split_model_ref = Mock(return_value=("openai", "gpt-4o-mini"))
    find_model_pricing = AsyncMock(return_value=pricing)

    monkeypatch.setattr(budget_service, "split_pricing_model_ref", split_model_ref)
    monkeypatch.setattr(budget_service, "find_model_pricing", find_model_pricing)

    assert await budget_service._is_model_free(db, "openai/gpt-4o-mini") is True

    split_model_ref.assert_called_once_with("openai/gpt-4o-mini")
    find_model_pricing.assert_awaited_once_with(db, "openai", "gpt-4o-mini")

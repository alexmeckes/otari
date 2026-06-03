from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from gateway.models.entities import BudgetAlert, UsageLog
from gateway.services.log_writer import SingleLogWriter, _record_and_dispatch_new_budget_alerts


@pytest.mark.asyncio
async def test_single_log_writer_rolls_back_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    session = AsyncMock()
    session.add = MagicMock()
    session.commit.side_effect = SQLAlchemyError("boom")

    @asynccontextmanager
    async def _session_cm() -> AsyncGenerator[AsyncMock, None]:
        yield session

    monkeypatch.setattr("gateway.services.log_writer.create_session", lambda: _session_cm())
    writer = SingleLogWriter()

    log = UsageLog(id="log", model="test-model", endpoint="/v1/test", status="success")
    await writer.put(log)

    session.rollback.assert_awaited()


@pytest.mark.asyncio
async def test_record_and_dispatch_new_budget_alerts_records_metrics_and_webhook_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[tuple[str, str]] = []
    dispatched: list[list[int]] = []

    def record_alert(scope_type: str, delivery_status: str) -> None:
        recorded.append((scope_type, delivery_status))

    async def dispatch_alerts(alert_ids: list[int]) -> None:
        dispatched.append(alert_ids)

    monkeypatch.setattr("gateway.services.log_writer.record_budget_alert_created", record_alert)
    monkeypatch.setattr("gateway.services.log_writer.dispatch_budget_alert_webhooks", dispatch_alerts)

    await _record_and_dispatch_new_budget_alerts(
        [
            BudgetAlert(
                id=1,
                budget_id="budget-1",
                scope_type="project",
                scope_id="project-1",
                threshold=0.5,
                spend=5.0,
                max_budget=10.0,
                webhook_url="https://alerts.example.test/hook",
                delivery_status="pending",
            ),
            BudgetAlert(
                id=2,
                budget_id="budget-2",
                scope_type="user",
                scope_id="user-1",
                threshold=0.8,
                spend=8.0,
                max_budget=10.0,
                webhook_url=None,
                delivery_status="not_configured",
            ),
            BudgetAlert(
                budget_id="budget-3",
                scope_type="tag",
                scope_id=None,
                threshold=0.9,
                spend=9.0,
                max_budget=10.0,
                webhook_url="https://alerts.example.test/hook",
                delivery_status="pending",
            ),
        ]
    )

    assert recorded == [
        ("project", "pending"),
        ("user", "not_configured"),
        ("tag", "pending"),
    ]
    assert dispatched == [[1]]

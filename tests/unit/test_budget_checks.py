from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, status

from gateway.api.routes._budget_checks import validate_scoped_request_budgets, validate_user_request_budget


@pytest.mark.asyncio
async def test_validate_user_request_budget_releases_for_update_locks() -> None:
    db = AsyncMock()

    with patch("gateway.api.routes._budget_checks.validate_user_budget", new_callable=AsyncMock) as validate_user:
        await validate_user_request_budget(db, "user-1", "openai:gpt-4o", strategy="for_update")

    validate_user.assert_awaited_once_with(db, "user-1", "openai:gpt-4o", strategy="for_update")
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_validate_user_request_budget_skips_rollback_for_cas() -> None:
    db = AsyncMock()

    with patch("gateway.api.routes._budget_checks.validate_user_budget", new_callable=AsyncMock):
        await validate_user_request_budget(db, "user-1", "openai:gpt-4o", strategy="cas")

    db.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_user_request_budget_keeps_exception_path_unchanged() -> None:
    db = AsyncMock()
    error = HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="blocked")

    with patch(
        "gateway.api.routes._budget_checks.validate_user_budget",
        new_callable=AsyncMock,
        side_effect=error,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await validate_user_request_budget(db, "user-1", "openai:gpt-4o", strategy="for_update")

    assert exc_info.value is error
    db.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_scoped_request_budgets_checks_all_scopes_then_releases() -> None:
    db = AsyncMock()
    tags = {"team": "platform"}

    with (
        patch("gateway.api.routes._budget_checks.validate_user_budget", new_callable=AsyncMock) as validate_user,
        patch("gateway.api.routes._budget_checks.validate_project_budget", new_callable=AsyncMock) as validate_project,
        patch("gateway.api.routes._budget_checks.validate_tag_budgets", new_callable=AsyncMock) as validate_tags,
    ):
        await validate_scoped_request_budgets(
            db,
            user_id="user-1",
            model="openai:gpt-4o",
            project_id="project-1",
            tags=tags,
            strategy="for_update",
        )

    validate_user.assert_awaited_once_with(db, "user-1", "openai:gpt-4o", strategy="for_update")
    validate_project.assert_awaited_once_with(db, "project-1", "openai:gpt-4o", strategy="for_update")
    validate_tags.assert_awaited_once_with(db, tags, "openai:gpt-4o", strategy="for_update")
    db.rollback.assert_awaited_once()

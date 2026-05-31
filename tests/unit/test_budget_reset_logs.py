from datetime import UTC, datetime, timedelta

import pytest

from gateway.models.entities import Budget, Project, User
from gateway.services.budget_reset_logs import new_budget_reset_log


def test_new_budget_reset_log_for_user() -> None:
    reset_at = datetime(2026, 1, 1, tzinfo=UTC)
    next_reset_at = reset_at + timedelta(hours=1)
    user = User(user_id="user-1")

    reset_log = new_budget_reset_log(
        user,
        budget_id="budget-1",
        previous_spend=12.5,
        reset_at=reset_at,
        next_reset_at=next_reset_at,
    )

    assert reset_log.user_id == "user-1"
    assert reset_log.project_id is None
    assert reset_log.budget_id == "budget-1"
    assert reset_log.previous_spend == 12.5
    assert reset_log.reset_at == reset_at
    assert reset_log.next_reset_at == next_reset_at


def test_new_budget_reset_log_for_project() -> None:
    reset_at = datetime(2026, 1, 1, tzinfo=UTC)
    project = Project(project_id="project-1")

    reset_log = new_budget_reset_log(
        project,
        budget_id="budget-1",
        previous_spend=7.5,
        reset_at=reset_at,
        next_reset_at=None,
    )

    assert reset_log.user_id is None
    assert reset_log.project_id == "project-1"
    assert reset_log.budget_id == "budget-1"
    assert reset_log.previous_spend == 7.5
    assert reset_log.reset_at == reset_at
    assert reset_log.next_reset_at is None


def test_new_budget_reset_log_for_tag_budget() -> None:
    reset_at = datetime(2026, 1, 1, tzinfo=UTC)
    budget = Budget(budget_id="tag-budget-1")

    reset_log = new_budget_reset_log(
        budget,
        previous_spend=3.25,
        reset_at=reset_at,
        next_reset_at=None,
    )

    assert reset_log.user_id is None
    assert reset_log.project_id is None
    assert reset_log.budget_id == "tag-budget-1"
    assert reset_log.previous_spend == 3.25
    assert reset_log.reset_at == reset_at
    assert reset_log.next_reset_at is None


def test_new_budget_reset_log_requires_budget_id_for_user_or_project() -> None:
    with pytest.raises(ValueError, match="budget_id is required"):
        new_budget_reset_log(
            User(user_id="user-1"),
            previous_spend=1.0,
            reset_at=datetime(2026, 1, 1, tzinfo=UTC),
            next_reset_at=None,
        )

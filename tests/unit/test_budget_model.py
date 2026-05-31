from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from gateway.api.routes._budget_models import (
    BudgetAlertResponse,
    BudgetResponse,
    CreateBudgetRequest,
    UpdateBudgetRequest,
)
from gateway.models.entities import Budget, BudgetAlert


def test_create_and_update_budget_requests_share_scope_normalization() -> None:
    assert CreateBudgetRequest(scope_type=" TAG ", match_tags={"team": "platform"}).scope_type == "tag"
    assert UpdateBudgetRequest(scope_type=" ENTITY ").scope_type == "entity"
    assert UpdateBudgetRequest(scope_type=None).scope_type is None


def test_budget_request_scope_normalization_rejects_unsupported_values() -> None:
    with pytest.raises(ValidationError, match="scope_type must be 'entity' or 'tag'"):
        CreateBudgetRequest(scope_type="workspace")
    with pytest.raises(ValidationError, match="scope_type must be 'entity' or 'tag'"):
        UpdateBudgetRequest(scope_type="workspace")


def test_budget_requests_normalize_alert_fields() -> None:
    create_request = CreateBudgetRequest(
        alert_thresholds=[0.8, 0.5, 0.5],
        alert_webhook_url=" https://alerts.example.test/hook ",
    )
    update_request = UpdateBudgetRequest(
        alert_thresholds=[0.8, 0.5, 0.5],
        alert_webhook_url=" https://alerts.example.test/hook ",
    )

    assert create_request.alert_thresholds == [0.5, 0.8]
    assert create_request.alert_webhook_url == "https://alerts.example.test/hook"
    assert update_request.alert_thresholds == [0.5, 0.8]
    assert update_request.alert_webhook_url == "https://alerts.example.test/hook"


def test_budget_match_tag_dict_returns_match_tags_when_dict() -> None:
    budget = Budget(match_tags={"team": "platform"})

    assert budget.match_tag_dict() == {"team": "platform"}
    assert budget.to_dict()["match_tags"] == {"team": "platform"}


def test_budget_match_tag_dict_returns_empty_dict_for_invalid_match_tags() -> None:
    budget = Budget(match_tags=None)

    assert budget.match_tag_dict() == {}
    assert budget.to_dict()["match_tags"] == {}


def test_budget_alert_threshold_list_returns_thresholds_when_list() -> None:
    budget = Budget(alert_thresholds=[0.5, 0.8])

    assert budget.alert_threshold_list() == [0.5, 0.8]
    assert budget.to_dict()["alert_thresholds"] == [0.5, 0.8]


def test_budget_alert_threshold_list_returns_empty_list_for_invalid_thresholds() -> None:
    budget = Budget(alert_thresholds=None)

    assert budget.alert_threshold_list() == []
    assert budget.to_dict()["alert_thresholds"] == []


def test_budget_response_formats_optional_datetimes() -> None:
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    budget = Budget(
        budget_id="budget_123",
        max_budget=10.0,
        budget_duration_sec=None,
        scope_type="entity",
        match_tags={},
        alert_thresholds=[],
        alert_webhook_url=None,
        spend=1.25,
        budget_started_at=timestamp,
        next_budget_reset_at=None,
        blocked=False,
        is_active=True,
        created_at=timestamp,
        updated_at=timestamp,
    )

    response = BudgetResponse.from_model(budget)

    assert response.budget_started_at == timestamp.isoformat()
    assert response.next_budget_reset_at is None
    assert response.created_at == timestamp.isoformat()
    assert response.updated_at == timestamp.isoformat()


def test_budget_alert_response_formats_optional_datetimes() -> None:
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    alert = BudgetAlert(
        id=1,
        budget_id="budget_123",
        scope_type="project",
        scope_id="project_123",
        threshold=0.5,
        spend=5.0,
        max_budget=10.0,
        budget_period_start=timestamp,
        webhook_url=None,
        delivery_status="delivered",
        delivery_attempts=1,
        last_delivery_status_code=202,
        last_delivery_error=None,
        last_delivery_attempt_at=timestamp,
        next_delivery_attempt_at=None,
        delivered_at=timestamp,
        dead_lettered_at=None,
        created_at=timestamp,
        metadata_={"project_id": "project_123"},
    )

    response = BudgetAlertResponse.from_model(alert)

    assert response.budget_period_start == timestamp.isoformat()
    assert response.last_delivery_attempt_at == timestamp.isoformat()
    assert response.next_delivery_attempt_at is None
    assert response.delivered_at == timestamp.isoformat()
    assert response.dead_lettered_at is None
    assert response.created_at == timestamp.isoformat()

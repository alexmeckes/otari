from datetime import UTC, datetime

from gateway.api.routes._key_models import KeyInfo
from gateway.api.routes._project_models import ProjectResponse
from gateway.api.routes._response_datetime import optional_datetime_isoformat
from gateway.api.routes._user_models import UserResponse
from gateway.models.entities import APIKey, Project, User


def test_optional_datetime_isoformat_formats_datetimes_only() -> None:
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    assert optional_datetime_isoformat(timestamp) == timestamp.isoformat()
    assert optional_datetime_isoformat(None) is None
    assert optional_datetime_isoformat("2026-01-02T03:04:05+00:00") is None


def test_key_response_formats_optional_datetimes() -> None:
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    key = APIKey(
        id="key_123",
        key_hash="hash",
        key_name="test key",
        user_id="user_123",
        created_at=timestamp,
        last_used_at=timestamp,
        expires_at=None,
        is_active=True,
        metadata_={"env": "test"},
    )

    response = KeyInfo.from_model(key)

    assert response.created_at == timestamp.isoformat()
    assert response.last_used_at == timestamp.isoformat()
    assert response.expires_at is None


def test_project_response_formats_optional_datetimes() -> None:
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    project = Project(
        project_id="project_123",
        name="Test Project",
        routing_policy_id=None,
        spend=1.25,
        budget_id="budget_123",
        budget_started_at=timestamp,
        next_budget_reset_at=None,
        blocked=False,
        is_active=True,
        metadata_={"team": "platform"},
        created_at=timestamp,
        updated_at=timestamp,
    )

    response = ProjectResponse.from_model(project)

    assert response.budget_started_at == timestamp.isoformat()
    assert response.next_budget_reset_at is None
    assert response.created_at == timestamp.isoformat()
    assert response.updated_at == timestamp.isoformat()


def test_user_response_formats_optional_datetimes() -> None:
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    user = User(
        user_id="user_123",
        alias="Test User",
        spend=1.25,
        budget_id="budget_123",
        budget_started_at=None,
        next_budget_reset_at=timestamp,
        blocked=False,
        metadata_={"plan": "team"},
        created_at=timestamp,
        updated_at=timestamp,
    )

    response = UserResponse.from_model(user)

    assert response.budget_started_at is None
    assert response.next_budget_reset_at == timestamp.isoformat()
    assert response.created_at == timestamp.isoformat()
    assert response.updated_at == timestamp.isoformat()

import uuid
from typing import Any

import pytest

from gateway.models.entities import (
    Budget,
    Project,
    RouteTrace,
    RoutingPolicy,
    RoutingPolicyRevision,
    UsageLog,
)

UUID_DEFAULT_COLUMNS = [
    (Budget, "budget_id"),
    (RoutingPolicy, "policy_id"),
    (RoutingPolicyRevision, "revision_id"),
    (Project, "project_id"),
    (UsageLog, "id"),
    (RouteTrace, "trace_id"),
]


@pytest.mark.parametrize(("model", "column_name"), UUID_DEFAULT_COLUMNS)
def test_uuid_defaults_generate_uuid_strings(model: type[Any], column_name: str) -> None:
    default = model.__table__.columns[column_name].default
    assert default is not None

    value = default.arg(None)

    assert isinstance(value, str)
    assert str(uuid.UUID(value)) == value

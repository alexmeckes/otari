from datetime import UTC, datetime
from typing import Any

import pytest

from gateway.models.entities import (
    APIKey,
    Budget,
    BudgetAlert,
    BudgetResetLog,
    ModelPricing,
    Project,
    RouteTrace,
    RoutingPolicy,
    RoutingPolicyRevision,
    UsageLog,
    User,
)

TIMESTAMP_DEFAULT_COLUMNS = [
    (APIKey, "created_at"),
    (Budget, "created_at"),
    (Budget, "updated_at"),
    (User, "created_at"),
    (User, "updated_at"),
    (ModelPricing, "effective_at"),
    (ModelPricing, "created_at"),
    (ModelPricing, "updated_at"),
    (RoutingPolicy, "created_at"),
    (RoutingPolicy, "updated_at"),
    (RoutingPolicyRevision, "created_at"),
    (Project, "created_at"),
    (Project, "updated_at"),
    (UsageLog, "timestamp"),
    (RouteTrace, "timestamp"),
    (BudgetResetLog, "reset_at"),
    (BudgetAlert, "created_at"),
]

TIMESTAMP_ONUPDATE_COLUMNS = [
    (Budget, "updated_at"),
    (User, "updated_at"),
    (ModelPricing, "updated_at"),
    (RoutingPolicy, "updated_at"),
    (Project, "updated_at"),
]


def _call_column_default(model: type[Any], column_name: str) -> datetime:
    default = model.__table__.columns[column_name].default
    assert default is not None
    value = default.arg(None)
    assert isinstance(value, datetime)
    return value


@pytest.mark.parametrize(("model", "column_name"), TIMESTAMP_DEFAULT_COLUMNS)
def test_timestamp_defaults_generate_utc_datetimes(model: type[Any], column_name: str) -> None:
    value = _call_column_default(model, column_name)

    assert value.tzinfo is UTC


@pytest.mark.parametrize(("model", "column_name"), TIMESTAMP_ONUPDATE_COLUMNS)
def test_timestamp_onupdate_generates_utc_datetimes(model: type[Any], column_name: str) -> None:
    onupdate = model.__table__.columns[column_name].onupdate
    assert onupdate is not None
    value = onupdate.arg(None)

    assert isinstance(value, datetime)
    assert value.tzinfo is UTC

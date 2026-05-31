"""Request and response models for budget routes."""

from datetime import datetime
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel, Field, field_validator

from gateway.models.entities import Budget, BudgetAlert
from gateway.services.budget_service import (
    TAG_BUDGET_SCOPE,
    normalize_alert_thresholds,
    normalize_alert_webhook_url,
)


def _budget_match_tags(budget: Budget) -> dict[str, Any]:
    match_tag_dict = getattr(budget, "match_tag_dict", None)
    if callable(match_tag_dict):
        match_tags = match_tag_dict()
        if isinstance(match_tags, dict):
            return match_tags
    match_tags = getattr(budget, "match_tags", {})
    return match_tags if isinstance(match_tags, dict) else {}


def _budget_alert_thresholds(budget: Budget) -> list[float]:
    alert_threshold_list = getattr(budget, "alert_threshold_list", None)
    if callable(alert_threshold_list):
        alert_thresholds = alert_threshold_list()
        if isinstance(alert_thresholds, list):
            return alert_thresholds
    alert_thresholds = getattr(budget, "alert_thresholds", [])
    return alert_thresholds if isinstance(alert_thresholds, list) else []


class CreateBudgetRequest(BaseModel):
    """Request model for creating a new budget."""

    max_budget: float | None = Field(default=None, ge=0, description="Maximum spending limit")
    budget_duration_sec: int | None = Field(
        default=None,
        gt=0,
        description="Budget duration in seconds (e.g., 86400 for daily, 604800 for weekly)",
    )
    scope_type: str = Field(default="entity", description="Budget scope: entity or tag")
    match_tags: dict[str, str] | None = Field(
        default=None,
        description="Exact request tags matched by tag-scoped budget groups",
    )
    alert_thresholds: list[float] | None = Field(
        default=None,
        description="Optional spend ratios that emit budget alert events when crossed, e.g. [0.5, 0.8, 0.95]",
    )
    alert_webhook_url: str | None = Field(
        default=None,
        description="Optional HTTP(S) endpoint that receives budget threshold alert webhooks",
    )
    blocked: bool = False
    is_active: bool = True

    @field_validator("scope_type")
    @classmethod
    def validate_scope_type(cls, value: str) -> str:
        """Validate supported budget scopes."""
        normalized = value.strip().lower()
        if normalized not in {"entity", TAG_BUDGET_SCOPE}:
            raise ValueError("scope_type must be 'entity' or 'tag'")
        return normalized

    @field_validator("alert_thresholds")
    @classmethod
    def validate_alert_thresholds(cls, value: list[float] | None) -> list[float]:
        """Validate optional budget alert thresholds."""
        return normalize_alert_thresholds(value)

    @field_validator("alert_webhook_url")
    @classmethod
    def validate_alert_webhook_url(cls, value: str | None) -> str | None:
        """Validate optional budget alert webhook URL."""
        return normalize_alert_webhook_url(value)


class BudgetResponse(BaseModel):
    """Response model for budget information."""

    budget_id: str
    max_budget: float | None
    budget_duration_sec: int | None
    scope_type: str
    match_tags: dict[str, Any]
    alert_thresholds: list[float]
    alert_webhook_url: str | None
    spend: float
    budget_started_at: str | None
    next_budget_reset_at: str | None
    blocked: bool
    is_active: bool
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, budget: Budget) -> "BudgetResponse":
        """Create a BudgetResponse from a Budget ORM model."""
        scope_type = getattr(budget, "scope_type", "entity")
        alert_webhook_url = getattr(budget, "alert_webhook_url", None)
        spend = getattr(budget, "spend", 0.0)
        budget_started_at = getattr(budget, "budget_started_at", None)
        next_budget_reset_at = getattr(budget, "next_budget_reset_at", None)
        blocked = getattr(budget, "blocked", False)
        is_active = getattr(budget, "is_active", True)
        return cls(
            budget_id=budget.budget_id,
            max_budget=budget.max_budget,
            budget_duration_sec=budget.budget_duration_sec,
            scope_type=scope_type if isinstance(scope_type, str) else "entity",
            match_tags=_budget_match_tags(budget),
            alert_thresholds=_budget_alert_thresholds(budget),
            alert_webhook_url=alert_webhook_url if isinstance(alert_webhook_url, str) else None,
            spend=float(spend) if isinstance(spend, int | float) else 0.0,
            budget_started_at=budget_started_at.isoformat() if isinstance(budget_started_at, datetime) else None,
            next_budget_reset_at=(
                next_budget_reset_at.isoformat()
                if isinstance(next_budget_reset_at, datetime)
                else None
            ),
            blocked=blocked if isinstance(blocked, bool) else False,
            is_active=is_active if isinstance(is_active, bool) else True,
            created_at=budget.created_at.isoformat(),
            updated_at=budget.updated_at.isoformat(),
        )


class UpdateBudgetRequest(BaseModel):
    """Request model for updating a budget."""

    max_budget: float | None = Field(default=None, ge=0)
    budget_duration_sec: int | None = Field(default=None, gt=0)
    scope_type: str | None = None
    match_tags: dict[str, str] | None = None
    alert_thresholds: list[float] | None = None
    alert_webhook_url: str | None = None
    spend: float | None = Field(default=None, ge=0)
    blocked: bool | None = None
    is_active: bool | None = None

    @field_validator("scope_type")
    @classmethod
    def validate_scope_type(cls, value: str | None) -> str | None:
        """Validate supported budget scopes."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in {"entity", TAG_BUDGET_SCOPE}:
            raise ValueError("scope_type must be 'entity' or 'tag'")
        return normalized

    @field_validator("alert_thresholds")
    @classmethod
    def validate_alert_thresholds(cls, value: list[float] | None) -> list[float]:
        """Validate optional budget alert thresholds."""
        return normalize_alert_thresholds(value)

    @field_validator("alert_webhook_url")
    @classmethod
    def validate_alert_webhook_url(cls, value: str | None) -> str | None:
        """Validate optional budget alert webhook URL."""
        return normalize_alert_webhook_url(value)


class BudgetAlertResponse(BaseModel):
    """Response model for budget threshold alert events."""

    id: int
    budget_id: str
    scope_type: str
    scope_id: str | None
    threshold: float
    spend: float
    max_budget: float
    budget_period_start: str | None
    webhook_url: str | None
    delivery_status: str
    delivery_attempts: int
    last_delivery_status_code: int | None
    last_delivery_error: str | None
    last_delivery_attempt_at: str | None
    next_delivery_attempt_at: str | None
    delivered_at: str | None
    dead_lettered_at: str | None
    created_at: str
    metadata: dict[str, Any]

    @classmethod
    def from_model(cls, alert: BudgetAlert) -> "BudgetAlertResponse":
        """Create a BudgetAlertResponse from a BudgetAlert ORM model."""
        return cls(
            id=alert.id,
            budget_id=alert.budget_id,
            scope_type=alert.scope_type,
            scope_id=alert.scope_id,
            threshold=alert.threshold,
            spend=alert.spend,
            max_budget=alert.max_budget,
            budget_period_start=(
                alert.budget_period_start.isoformat()
                if isinstance(alert.budget_period_start, datetime)
                else None
            ),
            webhook_url=alert.webhook_url,
            delivery_status=alert.delivery_status,
            delivery_attempts=alert.delivery_attempts,
            last_delivery_status_code=alert.last_delivery_status_code,
            last_delivery_error=alert.last_delivery_error,
            last_delivery_attempt_at=(
                alert.last_delivery_attempt_at.isoformat()
                if isinstance(alert.last_delivery_attempt_at, datetime)
                else None
            ),
            next_delivery_attempt_at=(
                alert.next_delivery_attempt_at.isoformat()
                if isinstance(alert.next_delivery_attempt_at, datetime)
                else None
            ),
            delivered_at=alert.delivered_at.isoformat() if isinstance(alert.delivered_at, datetime) else None,
            dead_lettered_at=(
                alert.dead_lettered_at.isoformat()
                if isinstance(alert.dead_lettered_at, datetime)
                else None
            ),
            created_at=alert.created_at.isoformat(),
            metadata=alert.metadata_dict(),
        )


def validate_tag_budget_shape(scope_type: str, match_tags: dict[str, str] | None) -> None:
    if scope_type == TAG_BUDGET_SCOPE and not match_tags:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Tag-scoped budgets require match_tags",
        )

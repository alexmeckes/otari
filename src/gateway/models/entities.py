import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _uuid_str() -> str:
    return str(uuid.uuid4())


def _dict_or_empty(value: Any, *, copy_value: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if copy_value:
        return dict(value)
    return value


def _isoformat_or_none(value: datetime | None) -> str | None:
    if value:
        return value.isoformat()
    return None


class APIKey(Base):
    """API Key model for authentication and authorization."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(primary_key=True)
    key_hash: Mapped[str] = mapped_column(unique=True, index=True)
    key_name: Mapped[str | None] = mapped_column()
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(default=True)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    user = relationship("User", back_populates="api_keys")
    usage_logs = relationship("UsageLog", back_populates="api_key", passive_deletes=True)
    route_traces = relationship("RouteTrace", back_populates="api_key", passive_deletes=True)

    def metadata_dict(self) -> dict[str, Any]:
        return _dict_or_empty(self.metadata_)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "key_name": self.key_name,
            "user_id": self.user_id,
            "created_at": _isoformat_or_none(self.created_at),
            "last_used_at": _isoformat_or_none(self.last_used_at),
            "expires_at": _isoformat_or_none(self.expires_at),
            "is_active": self.is_active,
            "metadata": self.metadata_dict(),
        }


class Budget(Base):
    """Budget model for spending limits."""

    __tablename__ = "budgets"

    budget_id: Mapped[str] = mapped_column(primary_key=True, default=_uuid_str)
    max_budget: Mapped[float | None] = mapped_column()
    budget_duration_sec: Mapped[int | None] = mapped_column()
    scope_type: Mapped[str] = mapped_column(default="entity", index=True)
    match_tags: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    alert_thresholds: Mapped[list[float] | None] = mapped_column(JSON, default=list)
    alert_webhook_url: Mapped[str | None] = mapped_column()
    spend: Mapped[float] = mapped_column(default=0.0)
    budget_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_budget_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    blocked: Mapped[bool] = mapped_column(default=False)
    is_active: Mapped[bool] = mapped_column(default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        onupdate=_utc_now,
    )

    users = relationship("User", back_populates="budget")
    projects = relationship("Project", back_populates="budget")
    reset_logs = relationship("BudgetResetLog", back_populates="budget")
    alerts = relationship("BudgetAlert", back_populates="budget", cascade="all, delete-orphan")

    def match_tag_dict(self) -> dict[str, Any]:
        return _dict_or_empty(self.match_tags)

    def alert_threshold_list(self) -> list[float]:
        if isinstance(self.alert_thresholds, list):
            return self.alert_thresholds
        return []

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "budget_id": self.budget_id,
            "max_budget": self.max_budget,
            "budget_duration_sec": self.budget_duration_sec,
            "scope_type": self.scope_type,
            "match_tags": self.match_tag_dict(),
            "alert_thresholds": self.alert_threshold_list(),
            "alert_webhook_url": self.alert_webhook_url,
            "spend": self.spend,
            "budget_started_at": _isoformat_or_none(self.budget_started_at),
            "next_budget_reset_at": _isoformat_or_none(self.next_budget_reset_at),
            "blocked": self.blocked,
            "is_active": self.is_active,
            "created_at": _isoformat_or_none(self.created_at),
            "updated_at": _isoformat_or_none(self.updated_at),
        }


class User(Base):
    """User/Customer model for end-user tracking."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(primary_key=True)
    alias: Mapped[str | None] = mapped_column()
    spend: Mapped[float] = mapped_column(default=0.0)
    budget_id: Mapped[str | None] = mapped_column(ForeignKey("budgets.budget_id"))
    budget_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_budget_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    blocked: Mapped[bool] = mapped_column(default=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        onupdate=_utc_now,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    budget = relationship("Budget", back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", passive_deletes=True)
    usage_logs = relationship("UsageLog", back_populates="user", passive_deletes=True)
    route_traces = relationship("RouteTrace", back_populates="user", passive_deletes=True)
    reset_logs = relationship("BudgetResetLog", back_populates="user", passive_deletes=True)

    def metadata_dict(self) -> dict[str, Any]:
        return _dict_or_empty(self.metadata_)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "user_id": self.user_id,
            "alias": self.alias,
            "spend": self.spend,
            "budget_id": self.budget_id,
            "budget_started_at": _isoformat_or_none(self.budget_started_at),
            "next_budget_reset_at": _isoformat_or_none(self.next_budget_reset_at),
            "blocked": self.blocked,
            "created_at": _isoformat_or_none(self.created_at),
            "updated_at": _isoformat_or_none(self.updated_at),
            "metadata": self.metadata_dict(),
        }


class ModelPricing(Base):
    """Model pricing configuration."""

    __tablename__ = "model_pricing"

    model_key: Mapped[str] = mapped_column(primary_key=True)
    effective_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        default=_utc_now,
    )
    input_price_per_million: Mapped[float] = mapped_column()
    output_price_per_million: Mapped[float] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        onupdate=_utc_now,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "model_key": self.model_key,
            "effective_at": _isoformat_or_none(self.effective_at),
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "created_at": _isoformat_or_none(self.created_at),
            "updated_at": _isoformat_or_none(self.updated_at),
        }


class RoutingPolicy(Base):
    """Routing policy for selecting upstream LLM models."""

    __tablename__ = "routing_policies"

    policy_id: Mapped[str] = mapped_column(primary_key=True, default=_uuid_str)
    name: Mapped[str] = mapped_column()
    strategy: Mapped[str] = mapped_column()
    config_: Mapped[dict[str, Any]] = mapped_column("config", JSON, default=dict)
    is_default: Mapped[bool] = mapped_column(default=False, index=True)
    revision: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        onupdate=_utc_now,
    )

    projects = relationship("Project", back_populates="routing_policy")
    route_traces = relationship("RouteTrace", back_populates="routing_policy", passive_deletes=True)

    def config_dict(self) -> dict[str, Any]:
        return _dict_or_empty(self.config_, copy_value=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "strategy": self.strategy,
            "config": self.config_dict(),
            "is_default": self.is_default,
            "revision": self.revision,
            "status": self.status,
            "created_at": _isoformat_or_none(self.created_at),
            "updated_at": _isoformat_or_none(self.updated_at),
        }


class RoutingPolicyRevision(Base):
    """Immutable snapshot of a routing policy change."""

    __tablename__ = "routing_policy_revisions"
    __table_args__ = (
        Index("ix_routing_policy_revisions_policy_id_created_at", "policy_id", "created_at"),
        Index("ix_routing_policy_revisions_policy_id_revision", "policy_id", "revision", unique=True),
    )

    revision_id: Mapped[str] = mapped_column(primary_key=True, default=_uuid_str)
    policy_id: Mapped[str] = mapped_column(index=True)
    revision: Mapped[int] = mapped_column()
    action: Mapped[str] = mapped_column()
    name: Mapped[str] = mapped_column()
    strategy: Mapped[str] = mapped_column()
    config_: Mapped[dict[str, Any]] = mapped_column("config", JSON, default=dict)
    is_default: Mapped[bool] = mapped_column(default=False)
    status: Mapped[str] = mapped_column(default="active")
    change_note: Mapped[str | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    def config_dict(self) -> dict[str, Any]:
        return _dict_or_empty(self.config_, copy_value=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "revision_id": self.revision_id,
            "policy_id": self.policy_id,
            "revision": self.revision,
            "action": self.action,
            "name": self.name,
            "strategy": self.strategy,
            "config": self.config_dict(),
            "is_default": self.is_default,
            "status": self.status,
            "change_note": self.change_note,
            "created_at": _isoformat_or_none(self.created_at),
        }


class Project(Base):
    """Gateway project for policy scoping and observability."""

    __tablename__ = "projects"

    project_id: Mapped[str] = mapped_column(primary_key=True, default=_uuid_str)
    name: Mapped[str | None] = mapped_column()
    routing_policy_id: Mapped[str | None] = mapped_column(
        ForeignKey("routing_policies.policy_id", ondelete="SET NULL"),
        index=True,
    )
    spend: Mapped[float] = mapped_column(default=0.0)
    budget_id: Mapped[str | None] = mapped_column(ForeignKey("budgets.budget_id"), index=True)
    budget_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_budget_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    blocked: Mapped[bool] = mapped_column(default=False)
    is_active: Mapped[bool] = mapped_column(default=True, index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        onupdate=_utc_now,
    )

    budget = relationship("Budget", back_populates="projects")
    routing_policy = relationship("RoutingPolicy", back_populates="projects")
    usage_logs = relationship("UsageLog", back_populates="project", passive_deletes=True)
    route_traces = relationship("RouteTrace", back_populates="project", passive_deletes=True)
    reset_logs = relationship("BudgetResetLog", back_populates="project", passive_deletes=True)

    def metadata_dict(self) -> dict[str, Any]:
        return _dict_or_empty(self.metadata_)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "routing_policy_id": self.routing_policy_id,
            "spend": self.spend,
            "budget_id": self.budget_id,
            "budget_started_at": _isoformat_or_none(self.budget_started_at),
            "next_budget_reset_at": _isoformat_or_none(self.next_budget_reset_at),
            "blocked": self.blocked,
            "is_active": self.is_active,
            "metadata": self.metadata_dict(),
            "created_at": _isoformat_or_none(self.created_at),
            "updated_at": _isoformat_or_none(self.updated_at),
        }


class UsageLog(Base):
    """Usage log model for tracking API requests."""

    __tablename__ = "usage_logs"
    __table_args__ = (
        Index("ix_usage_logs_user_id_timestamp", "user_id", "timestamp"),
        Index("ix_usage_logs_project_id_timestamp", "project_id", "timestamp"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=_uuid_str)
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="SET NULL"), index=True)
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.project_id", ondelete="SET NULL"),
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, index=True)

    model: Mapped[str] = mapped_column()
    provider: Mapped[str | None] = mapped_column()
    endpoint: Mapped[str] = mapped_column()

    prompt_tokens: Mapped[int | None] = mapped_column()
    completion_tokens: Mapped[int | None] = mapped_column()
    total_tokens: Mapped[int | None] = mapped_column()
    cost: Mapped[float | None] = mapped_column()

    status: Mapped[str] = mapped_column()
    error_message: Mapped[str | None] = mapped_column()
    tags: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)

    api_key = relationship("APIKey", back_populates="usage_logs")
    user = relationship("User", back_populates="usage_logs")
    project = relationship("Project", back_populates="usage_logs")

    def tag_dict(self) -> dict[str, Any]:
        return _dict_or_empty(self.tags)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "api_key_id": self.api_key_id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "timestamp": _isoformat_or_none(self.timestamp),
            "model": self.model,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "status": self.status,
            "error_message": self.error_message,
            "tags": self.tag_dict(),
        }


class RouteTrace(Base):
    """Trace of a gateway routing decision and provider attempts."""

    __tablename__ = "route_traces"
    __table_args__ = (
        Index("ix_route_traces_project_id_timestamp", "project_id", "timestamp"),
        Index("ix_route_traces_user_id_timestamp", "user_id", "timestamp"),
    )

    trace_id: Mapped[str] = mapped_column(primary_key=True, default=_uuid_str)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, index=True)
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="SET NULL"), index=True)
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.project_id", ondelete="SET NULL"),
        index=True,
    )
    policy_id: Mapped[str | None] = mapped_column(
        ForeignKey("routing_policies.policy_id", ondelete="SET NULL"),
        index=True,
    )

    requested_model: Mapped[str] = mapped_column()
    endpoint: Mapped[str] = mapped_column(default="/v1/chat/completions", index=True)
    selected_model: Mapped[str | None] = mapped_column()
    selected_provider: Mapped[str | None] = mapped_column()
    strategy: Mapped[str | None] = mapped_column()
    status: Mapped[str] = mapped_column()
    error_message: Mapped[str | None] = mapped_column()
    selected_reason: Mapped[str | None] = mapped_column()

    estimated_prompt_tokens: Mapped[int | None] = mapped_column()
    estimated_output_tokens: Mapped[int | None] = mapped_column()
    estimated_cost: Mapped[float | None] = mapped_column()
    fallback_enabled: Mapped[bool] = mapped_column(default=True)
    policy_source: Mapped[str | None] = mapped_column()
    tags: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    guardrails: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    context: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    candidates: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    attempts: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    api_key = relationship("APIKey", back_populates="route_traces")
    user = relationship("User", back_populates="route_traces")
    project = relationship("Project", back_populates="route_traces")
    routing_policy = relationship("RoutingPolicy", back_populates="route_traces")

    def tag_dict(self) -> dict[str, Any]:
        return _dict_or_empty(self.tags)

    def guardrail_dict(self) -> dict[str, Any]:
        return _dict_or_empty(self.guardrails)

    def context_dict(self) -> dict[str, Any]:
        return _dict_or_empty(self.context)

    def candidate_list(self) -> list[dict[str, Any]]:
        if isinstance(self.candidates, list):
            return self.candidates
        return []

    def attempt_list(self) -> list[dict[str, Any]]:
        if isinstance(self.attempts, list):
            return self.attempts
        return []

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "trace_id": self.trace_id,
            "timestamp": _isoformat_or_none(self.timestamp),
            "api_key_id": self.api_key_id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "policy_id": self.policy_id,
            "requested_model": self.requested_model,
            "endpoint": self.endpoint,
            "selected_model": self.selected_model,
            "selected_provider": self.selected_provider,
            "strategy": self.strategy,
            "status": self.status,
            "error_message": self.error_message,
            "selected_reason": self.selected_reason,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_cost": self.estimated_cost,
            "fallback_enabled": self.fallback_enabled,
            "policy_source": self.policy_source,
            "tags": self.tag_dict(),
            "guardrails": self.guardrail_dict(),
            "context": self.context_dict(),
            "candidates": self.candidate_list(),
            "attempts": self.attempt_list(),
        }


class BudgetResetLog(Base):
    """Budget reset log model for tracking budget resets."""

    __tablename__ = "budget_reset_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="SET NULL"), index=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.project_id", ondelete="SET NULL"), index=True)
    budget_id: Mapped[str] = mapped_column(ForeignKey("budgets.budget_id"))
    previous_spend: Mapped[float] = mapped_column()
    reset_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    next_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user = relationship("User", back_populates="reset_logs")
    project = relationship("Project", back_populates="reset_logs")
    budget = relationship("Budget", back_populates="reset_logs")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "budget_id": self.budget_id,
            "previous_spend": self.previous_spend,
            "reset_at": _isoformat_or_none(self.reset_at),
            "next_reset_at": _isoformat_or_none(self.next_reset_at),
        }


class BudgetAlert(Base):
    """Budget threshold alert emitted when tracked spend crosses a configured ratio."""

    __tablename__ = "budget_alerts"
    __table_args__ = (
        Index("ix_budget_alerts_budget_id_created_at", "budget_id", "created_at"),
        Index("ix_budget_alerts_scope_created_at", "scope_type", "scope_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    budget_id: Mapped[str] = mapped_column(ForeignKey("budgets.budget_id", ondelete="CASCADE"), index=True)
    scope_type: Mapped[str] = mapped_column(index=True)
    scope_id: Mapped[str | None] = mapped_column(index=True)
    threshold: Mapped[float] = mapped_column()
    spend: Mapped[float] = mapped_column()
    max_budget: Mapped[float] = mapped_column()
    budget_period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    webhook_url: Mapped[str | None] = mapped_column()
    delivery_status: Mapped[str] = mapped_column(default="not_configured", index=True)
    delivery_attempts: Mapped[int] = mapped_column(default=0)
    last_delivery_status_code: Mapped[int | None] = mapped_column()
    last_delivery_error: Mapped[str | None] = mapped_column()
    last_delivery_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_delivery_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dead_lettered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    budget = relationship("Budget", back_populates="alerts")

    def metadata_dict(self) -> dict[str, Any]:
        return _dict_or_empty(self.metadata_)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "budget_id": self.budget_id,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "threshold": self.threshold,
            "spend": self.spend,
            "max_budget": self.max_budget,
            "budget_period_start": _isoformat_or_none(self.budget_period_start),
            "webhook_url": self.webhook_url,
            "delivery_status": self.delivery_status,
            "delivery_attempts": self.delivery_attempts,
            "last_delivery_status_code": self.last_delivery_status_code,
            "last_delivery_error": self.last_delivery_error,
            "last_delivery_attempt_at": _isoformat_or_none(self.last_delivery_attempt_at),
            "next_delivery_attempt_at": _isoformat_or_none(self.next_delivery_attempt_at),
            "delivered_at": _isoformat_or_none(self.delivered_at),
            "dead_lettered_at": _isoformat_or_none(self.dead_lettered_at),
            "created_at": _isoformat_or_none(self.created_at),
            "metadata": self.metadata_dict(),
        }

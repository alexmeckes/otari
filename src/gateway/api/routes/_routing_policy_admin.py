"""Admin helpers for routing policy route mutations."""

from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._routing_policy_models import CreateRoutingPolicyRequest
from gateway.models.entities import RoutingPolicy, RoutingPolicyRevision
from gateway.services import routing_policy_shape
from gateway.services.routing_policy_service import ACTIVE_ROUTING_POLICY_STATUS, ROUTING_STRATEGIES

ROUTING_POLICY_STATUSES = {"draft", ACTIVE_ROUTING_POLICY_STATUS, "archived"}


def validate_strategy(strategy: str) -> None:
    if strategy not in ROUTING_STRATEGIES:
        supported = ", ".join(sorted(ROUTING_STRATEGIES))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported routing strategy '{strategy}'. Supported strategies: {supported}",
        )


def unprocessable(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


def create_policy_shape(request: CreateRoutingPolicyRequest) -> tuple[str, dict[str, Any]]:
    try:
        return routing_policy_shape.create_policy_shape(
            strategy=request.strategy,
            config=request.config,
            default_strategy=request.default_strategy,
        )
    except routing_policy_shape.RoutingPolicyShapeError as exc:
        raise unprocessable(str(exc)) from exc


def update_policy_shape(
    policy: RoutingPolicy,
    payload: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    try:
        return routing_policy_shape.update_policy_shape(
            current_config=policy.config_ or {},
            payload=payload,
        )
    except routing_policy_shape.RoutingPolicyShapeError as exc:
        raise unprocessable(str(exc)) from exc


def validate_status(policy_status: str) -> None:
    if policy_status not in ROUTING_POLICY_STATUSES:
        supported = ", ".join(sorted(ROUTING_POLICY_STATUSES))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unsupported routing policy status '{policy_status}'. Supported statuses: {supported}",
        )


def ensure_default_policy_is_active(*, policy_status: str, is_default: bool) -> None:
    if is_default and policy_status != ACTIVE_ROUTING_POLICY_STATUS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Only active routing policies can be default",
        )


def record_policy_revision(
    db: AsyncSession,
    policy: RoutingPolicy,
    *,
    action: str,
    change_note: str | None,
) -> None:
    db.add(
        RoutingPolicyRevision(
            policy_id=policy.policy_id,
            revision=policy.revision,
            action=action,
            name=policy.name,
            strategy=policy.strategy,
            config_=dict(policy.config_) if policy.config_ else {},
            is_default=bool(policy.is_default),
            status=policy.status,
            change_note=change_note,
        )
    )


def bump_policy_revision(policy: RoutingPolicy) -> None:
    policy.revision = int(policy.revision or 0) + 1


async def get_policy_or_404(db: AsyncSession, policy_id: str) -> RoutingPolicy:
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )
    return policy


async def get_active_policy_or_error(db: AsyncSession, policy_id: str) -> RoutingPolicy:
    policy = await get_policy_or_404(db, policy_id)
    if policy.status != ACTIVE_ROUTING_POLICY_STATUS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Routing policy '{policy_id}' is not active",
        )
    return policy


async def get_policy_revision(
    db: AsyncSession,
    *,
    policy_id: str,
    revision: int,
) -> RoutingPolicyRevision:
    result = await db.execute(
        select(RoutingPolicyRevision).where(
            RoutingPolicyRevision.policy_id == policy_id,
            RoutingPolicyRevision.revision == revision,
        )
    )
    policy_revision = result.scalar_one_or_none()
    if policy_revision is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy revision '{policy_id}@{revision}' not found",
        )
    return policy_revision


async def unset_default_policies(
    db: AsyncSession,
    *,
    except_policy_id: str | None = None,
    change_note: str | None = None,
) -> None:
    stmt = select(RoutingPolicy).where(RoutingPolicy.is_default.is_(True))
    if except_policy_id is not None:
        stmt = stmt.where(RoutingPolicy.policy_id != except_policy_id)
    result = await db.execute(stmt)
    for policy in result.scalars().all():
        policy.is_default = False
        bump_policy_revision(policy)
        record_policy_revision(
            db,
            policy,
            action="unset_default",
            change_note=change_note,
        )

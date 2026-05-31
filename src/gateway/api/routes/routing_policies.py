from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.api.routes._routing_policy_models import (
    AppliedRoutingPolicyEvalScoreResponse,
    ApplyRoutingPolicyEvalScoresRequest,
    ApplyRoutingPolicyEvalScoresResponse,
    ApplyRoutingPolicyRevisionRequest,
    CloneRoutingPolicyRequest,
    CreateRoutingPolicyRequest,
    RoutingPolicyResponse,
    RoutingPolicyRevisionResponse,
    UpdateRoutingPolicyRequest,
    eval_score_input,
)
from gateway.models.entities import RoutingPolicy, RoutingPolicyRevision
from gateway.services import routing_policy_eval_scores, routing_policy_shape
from gateway.services.routing_policy_service import ACTIVE_ROUTING_POLICY_STATUS, ROUTING_STRATEGIES

router = APIRouter(prefix="/v1/routing-policies", tags=["routing-policies"])

ROUTING_POLICY_STATUSES = {"draft", ACTIVE_ROUTING_POLICY_STATUS, "archived"}


def _validate_strategy(strategy: str) -> None:
    if strategy not in ROUTING_STRATEGIES:
        supported = ", ".join(sorted(ROUTING_STRATEGIES))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported routing strategy '{strategy}'. Supported strategies: {supported}",
        )


def _unprocessable(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


def _create_policy_shape(request: CreateRoutingPolicyRequest) -> tuple[str, dict[str, Any]]:
    try:
        return routing_policy_shape.create_policy_shape(
            strategy=request.strategy,
            config=request.config,
            default_strategy=request.default_strategy,
        )
    except routing_policy_shape.RoutingPolicyShapeError as exc:
        raise _unprocessable(str(exc)) from exc


def _update_policy_shape(
    policy: RoutingPolicy,
    payload: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    try:
        return routing_policy_shape.update_policy_shape(
            current_config=policy.config_ or {},
            payload=payload,
        )
    except routing_policy_shape.RoutingPolicyShapeError as exc:
        raise _unprocessable(str(exc)) from exc


def _validate_status(policy_status: str) -> None:
    if policy_status not in ROUTING_POLICY_STATUSES:
        supported = ", ".join(sorted(ROUTING_POLICY_STATUSES))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unsupported routing policy status '{policy_status}'. Supported statuses: {supported}",
        )


def _ensure_default_policy_is_active(*, policy_status: str, is_default: bool) -> None:
    if is_default and policy_status != ACTIVE_ROUTING_POLICY_STATUS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Only active routing policies can be default",
        )


def _record_policy_revision(
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


def _bump_policy_revision(policy: RoutingPolicy) -> None:
    policy.revision = int(policy.revision or 0) + 1


async def _get_policy_revision(
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


async def _unset_default_policies(
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
        _bump_policy_revision(policy)
        _record_policy_revision(
            db,
            policy,
            action="unset_default",
            change_note=change_note,
        )


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_routing_policy(
    request: CreateRoutingPolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Create a routing policy."""
    policy_strategy, policy_config = _create_policy_shape(request)
    _validate_strategy(policy_strategy)
    _validate_status(request.status)
    _ensure_default_policy_is_active(policy_status=request.status, is_default=request.is_default)
    if request.is_default:
        await _unset_default_policies(
            db,
            change_note=request.change_note or f"Unset default before creating routing policy '{request.name}'",
        )

    policy = RoutingPolicy(
        name=request.name,
        strategy=policy_strategy,
        config_=policy_config,
        is_default=request.is_default,
        revision=1,
        status=request.status,
    )
    db.add(policy)
    await db.flush()
    _record_policy_revision(
        db,
        policy,
        action="create",
        change_note=request.change_note,
    )
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(policy)
    return RoutingPolicyResponse.from_model(policy)


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_routing_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[RoutingPolicyResponse]:
    """List routing policies."""
    if status_filter is not None:
        _validate_status(status_filter)
    stmt = select(RoutingPolicy)
    if status_filter is not None:
        stmt = stmt.where(RoutingPolicy.status == status_filter)
    result = await db.execute(stmt.order_by(RoutingPolicy.created_at.desc()).offset(skip).limit(limit))
    policies = result.scalars().all()
    return [RoutingPolicyResponse.from_model(policy) for policy in policies]


@router.get("/{policy_id}/revisions", dependencies=[Depends(verify_master_key)])
async def list_routing_policy_revisions(
    policy_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[RoutingPolicyRevisionResponse]:
    """List immutable revisions for a routing policy."""
    result = await db.execute(
        select(RoutingPolicyRevision)
        .where(RoutingPolicyRevision.policy_id == policy_id)
        .order_by(RoutingPolicyRevision.revision.desc())
        .offset(skip)
        .limit(limit)
    )
    revisions = result.scalars().all()
    return [RoutingPolicyRevisionResponse.from_model(revision) for revision in revisions]


@router.get("/{policy_id}/revisions/{revision}", dependencies=[Depends(verify_master_key)])
async def get_routing_policy_revision(
    policy_id: str,
    revision: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyRevisionResponse:
    """Get one immutable routing policy revision."""
    policy_revision = await _get_policy_revision(db, policy_id=policy_id, revision=revision)
    return RoutingPolicyRevisionResponse.from_model(policy_revision)


@router.post("/{policy_id}/revisions/{revision}/apply", dependencies=[Depends(verify_master_key)])
async def apply_routing_policy_revision(
    policy_id: str,
    revision: int,
    request: ApplyRoutingPolicyRevisionRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Apply a previous policy revision as a new audited revision."""
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )
    policy_revision = await _get_policy_revision(db, policy_id=policy_id, revision=revision)
    _validate_strategy(policy_revision.strategy)
    _validate_status(policy_revision.status)
    _ensure_default_policy_is_active(
        policy_status=policy_revision.status,
        is_default=bool(policy_revision.is_default),
    )

    change_note = request.change_note or f"Applied routing policy revision {revision}"
    if policy_revision.is_default:
        await _unset_default_policies(
            db,
            except_policy_id=policy.policy_id,
            change_note=f"Unset default before applying routing policy revision '{policy_id}@{revision}'",
        )

    policy.name = policy_revision.name
    policy.strategy = policy_revision.strategy
    policy.config_ = dict(policy_revision.config_) if policy_revision.config_ else {}
    policy.is_default = bool(policy_revision.is_default)
    policy.status = policy_revision.status
    _bump_policy_revision(policy)
    _record_policy_revision(
        db,
        policy,
        action="apply_revision",
        change_note=change_note,
    )
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(policy)
    return RoutingPolicyResponse.from_model(policy)


@router.post("/{policy_id}/eval-scores", dependencies=[Depends(verify_master_key)])
async def apply_routing_policy_eval_scores(
    policy_id: str,
    request: ApplyRoutingPolicyEvalScoresRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApplyRoutingPolicyEvalScoresResponse:
    """Apply uploaded eval or benchmark scores to weighted routing candidates."""
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )
    if policy.strategy != "weighted_score":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Eval score ingestion requires a weighted_score routing policy",
        )

    try:
        score_application = routing_policy_eval_scores.apply_eval_scores_to_policy_config(
            policy.config_ or {},
            [eval_score_input(item) for item in request.scores],
        )
    except routing_policy_eval_scores.RoutingPolicyEvalScoreError as exc:
        raise _unprocessable(str(exc)) from exc
    if not score_application.applied_scores:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No eval scores matched routing policy candidates",
        )

    policy.config_ = score_application.config
    _bump_policy_revision(policy)
    _record_policy_revision(
        db,
        policy,
        action="apply_eval_scores",
        change_note=request.change_note or "Applied routing policy eval scores",
    )
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(policy)
    return ApplyRoutingPolicyEvalScoresResponse(
        policy=RoutingPolicyResponse.from_model(policy),
        applied_count=len(score_application.applied_scores),
        unmatched_models=score_application.unmatched_models,
        applied_scores=[
            AppliedRoutingPolicyEvalScoreResponse(
                model=score.model,
                previous_quality_score=score.previous_quality_score,
                quality_score=score.quality_score,
                sample_count=score.sample_count,
                metrics=score.metrics,
            )
            for score in score_application.applied_scores
        ],
    )


@router.get("/{policy_id}", dependencies=[Depends(verify_master_key)])
async def get_routing_policy(
    policy_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Get a routing policy."""
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )
    return RoutingPolicyResponse.from_model(policy)


@router.post("/{policy_id}/clone", dependencies=[Depends(verify_master_key)])
async def clone_routing_policy(
    policy_id: str,
    request: CloneRoutingPolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Clone a routing policy into an inactive draft for safe editing."""
    source = await db.get(RoutingPolicy, policy_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )

    clone = RoutingPolicy(
        name=request.name or f"{source.name} draft",
        strategy=source.strategy,
        config_=dict(source.config_) if source.config_ else {},
        is_default=False,
        revision=1,
        status="draft",
    )
    db.add(clone)
    await db.flush()
    _record_policy_revision(
        db,
        clone,
        action="clone",
        change_note=request.change_note or f"Cloned from routing policy '{source.policy_id}'",
    )
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(clone)
    return RoutingPolicyResponse.from_model(clone)


@router.patch("/{policy_id}", dependencies=[Depends(verify_master_key)])
async def update_routing_policy(
    policy_id: str,
    request: UpdateRoutingPolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Update a routing policy."""
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )

    payload = request.model_dump(exclude_unset=True)
    change_note = request.change_note
    next_status = str(payload["status"]) if "status" in payload and payload["status"] is not None else policy.status
    next_is_default = (
        bool(payload["is_default"])
        if "is_default" in payload and payload["is_default"] is not None
        else bool(policy.is_default)
    )
    _validate_status(next_status)
    _ensure_default_policy_is_active(policy_status=next_status, is_default=next_is_default)
    next_strategy, next_config = _update_policy_shape(policy, payload)
    if next_strategy is not None:
        _validate_strategy(next_strategy)
        policy.strategy = next_strategy
    if "name" in payload and payload["name"] is not None:
        policy.name = str(payload["name"])
    if next_config is not None:
        policy.config_ = next_config
    if "status" in payload and payload["status"] is not None:
        policy.status = str(payload["status"])
    if "is_default" in payload and payload["is_default"] is not None:
        policy.is_default = bool(payload["is_default"])
        if policy.is_default:
            await _unset_default_policies(
                db,
                except_policy_id=policy.policy_id,
                change_note=change_note or f"Unset default before promoting routing policy '{policy.policy_id}'",
            )
    _bump_policy_revision(policy)
    _record_policy_revision(
        db,
        policy,
        action="update",
        change_note=change_note,
    )

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(policy)
    return RoutingPolicyResponse.from_model(policy)


@router.delete(
    "/{policy_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_master_key)],
)
async def delete_routing_policy(
    policy_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    change_note: str | None = None,
) -> None:
    """Delete a routing policy."""
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )

    _bump_policy_revision(policy)
    _record_policy_revision(
        db,
        policy,
        action="delete",
        change_note=change_note,
    )
    await db.flush()
    await db.delete(policy)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None

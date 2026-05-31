from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.api.routes._database import commit_or_database_error
from gateway.api.routes._routing_policy_admin import (
    bump_policy_revision,
    create_policy_shape,
    ensure_default_policy_is_active,
    get_policy_or_404,
    get_policy_revision,
    record_policy_revision,
    unprocessable,
    unset_default_policies,
    update_policy_shape,
    validate_status,
    validate_strategy,
)
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
from gateway.services import routing_policy_eval_scores

router = APIRouter(prefix="/v1/routing-policies", tags=["routing-policies"])


async def _committed_policy_response(db: AsyncSession, policy: RoutingPolicy) -> RoutingPolicyResponse:
    await commit_or_database_error(db)
    await db.refresh(policy)
    return RoutingPolicyResponse.from_model(policy)


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_routing_policy(
    request: CreateRoutingPolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Create a routing policy."""
    policy_strategy, policy_config = create_policy_shape(request)
    validate_strategy(policy_strategy)
    validate_status(request.status)
    ensure_default_policy_is_active(policy_status=request.status, is_default=request.is_default)
    if request.is_default:
        await unset_default_policies(
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
    record_policy_revision(
        db,
        policy,
        action="create",
        change_note=request.change_note,
    )
    return await _committed_policy_response(db, policy)


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_routing_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[RoutingPolicyResponse]:
    """List routing policies."""
    if status_filter is not None:
        validate_status(status_filter)
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
    policy_revision = await get_policy_revision(db, policy_id=policy_id, revision=revision)
    return RoutingPolicyRevisionResponse.from_model(policy_revision)


@router.post("/{policy_id}/revisions/{revision}/apply", dependencies=[Depends(verify_master_key)])
async def apply_routing_policy_revision(
    policy_id: str,
    revision: int,
    request: ApplyRoutingPolicyRevisionRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Apply a previous policy revision as a new audited revision."""
    policy = await get_policy_or_404(db, policy_id)
    policy_revision = await get_policy_revision(db, policy_id=policy_id, revision=revision)
    validate_strategy(policy_revision.strategy)
    validate_status(policy_revision.status)
    ensure_default_policy_is_active(
        policy_status=policy_revision.status,
        is_default=bool(policy_revision.is_default),
    )

    change_note = request.change_note or f"Applied routing policy revision {revision}"
    if policy_revision.is_default:
        await unset_default_policies(
            db,
            except_policy_id=policy.policy_id,
            change_note=f"Unset default before applying routing policy revision '{policy_id}@{revision}'",
        )

    policy.name = policy_revision.name
    policy.strategy = policy_revision.strategy
    policy.config_ = dict(policy_revision.config_) if policy_revision.config_ else {}
    policy.is_default = bool(policy_revision.is_default)
    policy.status = policy_revision.status
    bump_policy_revision(policy)
    record_policy_revision(
        db,
        policy,
        action="apply_revision",
        change_note=change_note,
    )
    return await _committed_policy_response(db, policy)


@router.post("/{policy_id}/eval-scores", dependencies=[Depends(verify_master_key)])
async def apply_routing_policy_eval_scores(
    policy_id: str,
    request: ApplyRoutingPolicyEvalScoresRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApplyRoutingPolicyEvalScoresResponse:
    """Apply uploaded eval or benchmark scores to weighted routing candidates."""
    policy = await get_policy_or_404(db, policy_id)
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
        raise unprocessable(str(exc)) from exc
    if not score_application.applied_scores:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No eval scores matched routing policy candidates",
        )

    policy.config_ = score_application.config
    bump_policy_revision(policy)
    record_policy_revision(
        db,
        policy,
        action="apply_eval_scores",
        change_note=request.change_note or "Applied routing policy eval scores",
    )
    policy_response = await _committed_policy_response(db, policy)
    return ApplyRoutingPolicyEvalScoresResponse(
        policy=policy_response,
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
    policy = await get_policy_or_404(db, policy_id)
    return RoutingPolicyResponse.from_model(policy)


@router.post("/{policy_id}/clone", dependencies=[Depends(verify_master_key)])
async def clone_routing_policy(
    policy_id: str,
    request: CloneRoutingPolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Clone a routing policy into an inactive draft for safe editing."""
    source = await get_policy_or_404(db, policy_id)

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
    record_policy_revision(
        db,
        clone,
        action="clone",
        change_note=request.change_note or f"Cloned from routing policy '{source.policy_id}'",
    )
    return await _committed_policy_response(db, clone)


@router.patch("/{policy_id}", dependencies=[Depends(verify_master_key)])
async def update_routing_policy(
    policy_id: str,
    request: UpdateRoutingPolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Update a routing policy."""
    policy = await get_policy_or_404(db, policy_id)

    payload = request.model_dump(exclude_unset=True)
    change_note = request.change_note
    next_status = str(payload["status"]) if "status" in payload and payload["status"] is not None else policy.status
    next_is_default = (
        bool(payload["is_default"])
        if "is_default" in payload and payload["is_default"] is not None
        else bool(policy.is_default)
    )
    validate_status(next_status)
    ensure_default_policy_is_active(policy_status=next_status, is_default=next_is_default)
    next_strategy, next_config = update_policy_shape(policy, payload)
    if next_strategy is not None:
        validate_strategy(next_strategy)
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
            await unset_default_policies(
                db,
                except_policy_id=policy.policy_id,
                change_note=change_note or f"Unset default before promoting routing policy '{policy.policy_id}'",
            )
    bump_policy_revision(policy)
    record_policy_revision(
        db,
        policy,
        action="update",
        change_note=change_note,
    )

    return await _committed_policy_response(db, policy)


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
    policy = await get_policy_or_404(db, policy_id)

    bump_policy_revision(policy)
    record_policy_revision(
        db,
        policy,
        action="delete",
        change_note=change_note,
    )
    await db.flush()
    await db.delete(policy)
    await commit_or_database_error(db)

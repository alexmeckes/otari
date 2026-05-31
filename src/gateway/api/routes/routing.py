from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.api.routes._routing_models import ResolveRoutingRequest, ResolveRoutingResponse
from gateway.services.routing_policy_service import (
    DEFAULT_ROUTING_MODEL,
    RoutingPolicyError,
    resolve_routing_plan,
)

router = APIRouter(prefix="/v1/routing", tags=["routing"])


@router.post("/resolve", dependencies=[Depends(verify_master_key)])
async def resolve_routing(
    request: ResolveRoutingRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ResolveRoutingResponse:
    """Dry-run a routing policy decision without calling a provider."""
    if request.model != DEFAULT_ROUTING_MODEL:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Routing resolution only supports model '{DEFAULT_ROUTING_MODEL}'",
        )

    request_body = request.model_dump(exclude_none=True)
    try:
        plan = await resolve_routing_plan(
            db,
            request_body=request_body,
            project_id=request.project_id,
            tags=request.tags,
            policy_id=request.policy_id,
            allow_inactive_policy=request.policy_id is not None,
        )
    except RoutingPolicyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return ResolveRoutingResponse.from_plan(plan)

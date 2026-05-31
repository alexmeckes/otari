from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.api.routes._route_trace_models import (
    RouteTraceResponse,
    RouteTraceSummaryResponse,
    summarize_route_trace_logs,
)
from gateway.models.entities import RouteTrace

router = APIRouter(prefix="/v1/route-traces", tags=["route-traces"])


def _filtered_trace_stmt(
    project_id: str | None = None,
    user_id: str | None = None,
    policy_id: str | None = None,
    endpoint: str | None = None,
    status: str | None = None,
) -> Select[tuple[RouteTrace]]:
    stmt = select(RouteTrace)
    if project_id is not None:
        stmt = stmt.where(RouteTrace.project_id == project_id)
    if user_id is not None:
        stmt = stmt.where(RouteTrace.user_id == user_id)
    if policy_id is not None:
        stmt = stmt.where(RouteTrace.policy_id == policy_id)
    if endpoint is not None:
        stmt = stmt.where(RouteTrace.endpoint == endpoint)
    if status is not None:
        stmt = stmt.where(RouteTrace.status == status)
    return stmt


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_route_traces(
    db: Annotated[AsyncSession, Depends(get_db)],
    project_id: str | None = None,
    user_id: str | None = None,
    policy_id: str | None = None,
    endpoint: str | None = None,
    status: str | None = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[RouteTraceResponse]:
    """List route traces with optional filters."""
    stmt = _filtered_trace_stmt(
        project_id=project_id,
        user_id=user_id,
        policy_id=policy_id,
        endpoint=endpoint,
        status=status,
    )
    result = await db.execute(stmt.order_by(RouteTrace.timestamp.desc()).offset(skip).limit(limit))
    traces = result.scalars().all()
    return [RouteTraceResponse.from_model(trace) for trace in traces]


@router.get("/summary", dependencies=[Depends(verify_master_key)])
async def summarize_route_traces(
    db: Annotated[AsyncSession, Depends(get_db)],
    project_id: str | None = None,
    user_id: str | None = None,
    policy_id: str | None = None,
    endpoint: str | None = None,
    status: str | None = None,
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
) -> RouteTraceSummaryResponse:
    """Summarize route traces with optional filters."""
    stmt = _filtered_trace_stmt(
        project_id=project_id,
        user_id=user_id,
        policy_id=policy_id,
        endpoint=endpoint,
        status=status,
    )
    result = await db.execute(stmt.order_by(RouteTrace.timestamp.desc()).limit(limit))
    traces = list(result.scalars().all())
    return summarize_route_trace_logs(traces)


@router.get("/{trace_id}", dependencies=[Depends(verify_master_key)])
async def get_route_trace(
    trace_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RouteTraceResponse:
    """Get a route trace by id."""
    trace = await db.get(RouteTrace, trace_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Route trace '{trace_id}' not found",
        )
    return RouteTraceResponse.from_model(trace)

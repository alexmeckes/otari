"""Bulk usage log endpoint.

Provides a single query interface over all usage logs with optional
time range and user filters, ordered newest-first. Intended for
external systems that need to sync usage data (billing, analytics).
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.api.routes._usage_models import (
    UsageEntry,
    UsageSummaryResponse,
    matches_tag_filter,
    summarize_usage_logs,
)
from gateway.models.entities import UsageLog

router = APIRouter(prefix="/v1/usage", tags=["usage"])


def _filtered_usage_stmt(
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    user_id: str | None = None,
    project_id: str | None = None,
) -> Select[tuple[UsageLog]]:
    stmt = select(UsageLog)
    if start_date is not None:
        stmt = stmt.where(UsageLog.timestamp >= start_date)
    if end_date is not None:
        stmt = stmt.where(UsageLog.timestamp < end_date)
    if user_id is not None:
        stmt = stmt.where(UsageLog.user_id == user_id)
    if project_id is not None:
        stmt = stmt.where(UsageLog.project_id == project_id)
    return stmt


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(
        default=None,
        description="Return logs with timestamp >= start_date (ISO 8601 or Unix epoch seconds)",
    ),
    end_date: datetime | None = Query(
        default=None,
        description="Return logs with timestamp < end_date (ISO 8601 or Unix epoch seconds)",
    ),
    user_id: str | None = Query(default=None, description="Filter to a single user"),
    project_id: str | None = Query(default=None, description="Filter to a gateway project"),
    tag_key: str | None = Query(default=None, description="Filter to logs containing this usage tag key"),
    tag_value: str | None = Query(
        default=None,
        description="When tag_key is set, filter to logs whose tag value string matches this value",
    ),
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageEntry]:
    """List usage logs ordered by timestamp (most recent first).

    Supports optional filters for time range, user, project, and usage tags.
    Paginated via skip/limit. Timestamps accept either ISO 8601 strings or
    Unix epoch seconds (numeric).
    """

    stmt = _filtered_usage_stmt(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        project_id=project_id,
    ).order_by(UsageLog.timestamp.desc())
    if tag_key is None:
        stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    logs = result.scalars().all()
    if tag_key is not None:
        matching_logs = [log for log in logs if matches_tag_filter(log, tag_key, tag_value)]
        logs = matching_logs[skip : skip + limit]
    return [UsageEntry.from_model(log) for log in logs]


@router.get("/summary", dependencies=[Depends(verify_master_key)])
async def summarize_usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(
        default=None,
        description="Return logs with timestamp >= start_date (ISO 8601 or Unix epoch seconds)",
    ),
    end_date: datetime | None = Query(
        default=None,
        description="Return logs with timestamp < end_date (ISO 8601 or Unix epoch seconds)",
    ),
    user_id: str | None = Query(default=None, description="Filter to a single user"),
    project_id: str | None = Query(default=None, description="Filter to a gateway project"),
    tag_key: str | None = Query(default=None, description="Filter to logs containing this usage tag key"),
    tag_value: str | None = Query(
        default=None,
        description="When tag_key is set, filter to logs whose tag value string matches this value",
    ),
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
) -> UsageSummaryResponse:
    """Summarize usage logs with optional project and tag filters."""

    stmt = _filtered_usage_stmt(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        project_id=project_id,
    ).order_by(UsageLog.timestamp.desc())
    result = await db.execute(stmt.limit(limit))
    logs = [log for log in result.scalars().all() if matches_tag_filter(log, tag_key, tag_value)]
    return summarize_usage_logs(logs)

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_api_key_or_master_key, verify_master_key
from gateway.api.routes._database import commit_or_database_error
from gateway.api.routes._pricing_models import PricingResponse, SetPricingRequest
from gateway.models.entities import ModelPricing
from gateway.services.pricing_service import (
    candidate_pricing_model_refs,
    normalize_effective_at,
    normalized_pricing_model_ref,
)

router = APIRouter(prefix="/v1/pricing", tags=["pricing"])


async def _pricing_rows(
    db: AsyncSession,
    model_keys: list[str],
    *criteria: Any,
    limit: int | None = None,
) -> list[ModelPricing]:
    for key in model_keys:
        stmt = (
            select(ModelPricing)
            .where(ModelPricing.model_key == key, *criteria)
            .order_by(ModelPricing.effective_at.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        pricings = list((await db.execute(stmt)).scalars().all())
        if pricings:
            return pricings
    return []


def _pricing_not_found(model_key: str, effective_at: datetime | None = None) -> HTTPException:
    detail = f"Pricing for model '{model_key}' not found"
    if effective_at is not None:
        detail = f"Pricing for model '{model_key}' with effective_at {effective_at.isoformat()} not found"
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


async def _pricing_rows_or_404(
    db: AsyncSession,
    model_key: str,
    *criteria: Any,
    effective_at: datetime | None = None,
    limit: int | None = None,
) -> list[ModelPricing]:
    rows = await _pricing_rows(db, candidate_pricing_model_refs(model_key), *criteria, limit=limit)
    if not rows:
        raise _pricing_not_found(model_key, effective_at)
    return rows


@router.post("", dependencies=[Depends(verify_master_key)])
async def set_pricing(
    request: SetPricingRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PricingResponse:
    """Set or update pricing for a model."""
    normalized_key = normalized_pricing_model_ref(request.model_key)
    effective_at = normalize_effective_at(request.effective_at)
    result = await db.execute(
        select(ModelPricing).where(
            ModelPricing.model_key == normalized_key,
            ModelPricing.effective_at == effective_at,
        )
    )
    pricing = result.scalar_one_or_none()

    if pricing:
        pricing.input_price_per_million = request.input_price_per_million
        pricing.output_price_per_million = request.output_price_per_million
    else:
        pricing = ModelPricing(
            model_key=normalized_key,
            effective_at=effective_at,
            input_price_per_million=request.input_price_per_million,
            output_price_per_million=request.output_price_per_million,
        )
        db.add(pricing)

    await commit_or_database_error(db)
    await db.refresh(pricing)

    return PricingResponse.from_model(pricing)


@router.get("", dependencies=[Depends(verify_api_key_or_master_key)])
async def list_pricing(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[PricingResponse]:
    """List all model pricing."""
    stmt = (
        select(ModelPricing)
        .order_by(ModelPricing.model_key, ModelPricing.effective_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    pricings = result.scalars().all()

    return [PricingResponse.from_model(pricing) for pricing in pricings]


@router.get("/{model_key:path}/history", dependencies=[Depends(verify_api_key_or_master_key)])
async def get_pricing_history(
    model_key: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[PricingResponse]:
    """Return the full pricing history for a model."""

    pricings = await _pricing_rows_or_404(db, model_key)
    return [PricingResponse.from_model(pricing) for pricing in pricings]


@router.get("/{model_key:path}", dependencies=[Depends(verify_api_key_or_master_key)])
async def get_pricing(
    model_key: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    as_of: Annotated[datetime | None, Query(description="ISO datetime for effective lookup", title="as_of")] = None,
) -> PricingResponse:
    """Get pricing for a specific model as of a timestamp."""

    pricing = (
        await _pricing_rows_or_404(
            db,
            model_key,
            ModelPricing.effective_at <= normalize_effective_at(as_of),
            limit=1,
        )
    )[0]
    return PricingResponse.from_model(pricing)


@router.delete(
    "/{model_key:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_master_key)],
)
async def delete_pricing(
    model_key: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    effective_at: Annotated[
        datetime | None,
        Query(
            description="ISO datetime identifying a specific pricing row to delete",
        ),
    ] = None,
) -> None:
    """Delete pricing entries for a model."""

    if effective_at is not None:
        normalized_effective_at = normalize_effective_at(effective_at)
        targets = await _pricing_rows_or_404(
            db,
            model_key,
            ModelPricing.effective_at == normalized_effective_at,
            effective_at=normalized_effective_at,
            limit=1,
        )
    else:
        targets = await _pricing_rows_or_404(db, model_key)

    for pricing in targets:
        await db.delete(pricing)

    await commit_or_database_error(db)

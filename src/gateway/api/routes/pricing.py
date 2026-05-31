from datetime import datetime
from typing import Annotated, Any

from any_llm import AnyLLM
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_api_key_or_master_key, verify_master_key
from gateway.api.routes._database import commit_or_database_error
from gateway.api.routes._pricing_models import PricingResponse, SetPricingRequest
from gateway.models.entities import ModelPricing
from gateway.services.pricing_service import normalize_effective_at

router = APIRouter(prefix="/v1/pricing", tags=["pricing"])


def _candidate_model_keys(raw_key: str) -> list[str]:
    """Return possible stored keys for a provided selector."""

    candidates = [raw_key]
    try:
        provider, model_name = AnyLLM.split_model_provider(raw_key)
    except ValueError:
        return candidates

    provider_value = provider.value if provider else None
    if not provider_value:
        return candidates

    for key in (f"{provider_value}:{model_name}", f"{provider_value}/{model_name}"):
        if key not in candidates:
            candidates.append(key)
    return candidates


async def _first_pricing(
    db: AsyncSession,
    model_keys: list[str],
    *criteria: Any,
) -> ModelPricing | None:
    rows = await _pricing_rows(db, model_keys, *criteria, limit=1)
    return rows[0] if rows else None


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


@router.post("", dependencies=[Depends(verify_master_key)])
async def set_pricing(
    request: SetPricingRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PricingResponse:
    """Set or update pricing for a model."""
    provider, model_name = AnyLLM.split_model_provider(request.model_key)
    normalized_key = f"{provider.value}:{model_name}"
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

    candidates = _candidate_model_keys(model_key)
    pricings = await _pricing_rows(db, candidates)
    if not pricings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pricing for model '{model_key}' not found",
        )

    return [PricingResponse.from_model(pricing) for pricing in pricings]


@router.get("/{model_key:path}", dependencies=[Depends(verify_api_key_or_master_key)])
async def get_pricing(
    model_key: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    as_of: Annotated[datetime | None, Query(description="ISO datetime for effective lookup", title="as_of")] = None,
) -> PricingResponse:
    """Get pricing for a specific model as of a timestamp."""

    candidates = _candidate_model_keys(model_key)
    pricing = await _first_pricing(db, candidates, ModelPricing.effective_at <= normalize_effective_at(as_of))

    if not pricing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pricing for model '{model_key}' not found",
        )

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

    candidates = _candidate_model_keys(model_key)

    if effective_at is not None:
        normalized_effective_at = normalize_effective_at(effective_at)
        pricing = await _first_pricing(db, candidates, ModelPricing.effective_at == normalized_effective_at)
        if not pricing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Pricing for model '{model_key}' with effective_at {normalized_effective_at.isoformat()} not found"
                ),
            )
        targets = [pricing]
    else:
        targets = await _pricing_rows(db, candidates)
        if not targets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pricing for model '{model_key}' not found",
            )

    for pricing in targets:
        await db.delete(pricing)

    await commit_or_database_error(db)

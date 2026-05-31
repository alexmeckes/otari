"""Model listing endpoints with OpenAI and gateway-catalog shapes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_api_key_or_master_key
from gateway.api.routes._model_catalog import (
    GatewayCatalogListResponse,
    GatewayCatalogModel,
    GatewayVendorListResponse,
    GatewayVendorResponse,
    ModelListResponse,
    ModelObject,
    filter_catalog_records,
    gateway_model_from_catalog_record,
    load_catalog_records,
    load_model_object,
    model_from_catalog_record,
    paginate,
    vendor_catalog_from_records,
)
from gateway.core.config import GatewayConfig

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", dependencies=[Depends(verify_api_key_or_master_key)])
async def list_models(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    provider: Annotated[str | None, Query(description="Filter models by provider name")] = None,
    vendor: Annotated[str | None, Query(description="Filter gateway catalog models by execution vendor")] = None,
    model: Annotated[str | None, Query(description="Fetch one gateway catalog model by provider/model id")] = None,
    catalog_format: Annotated[
        str,
        Query(alias="format", description="Response format: 'openai' (default) or 'gateway' catalog shape"),
    ] = "openai",
    cursor: Annotated[str | None, Query(description="Gateway catalog pagination cursor")] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Gateway catalog page size")] = 50,
) -> ModelListResponse | GatewayCatalogListResponse | GatewayCatalogModel:
    """List all available models.

    Returns models auto-discovered from configured providers, enriched with
    pricing data from the model_pricing table when available. Models that only
    exist in the pricing table are also included for backward compatibility.
    """
    normalized_format = catalog_format.lower()
    if normalized_format not in {"openai", "gateway"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="format must be 'openai' or 'gateway'",
        )

    records = await load_catalog_records(db, config, provider_filter=provider)

    use_gateway_catalog = normalized_format == "gateway" or vendor is not None or model is not None
    if use_gateway_catalog:
        filtered = filter_catalog_records(records, provider=provider, vendor=vendor, model=model)
        if model is not None:
            if not filtered:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model}' not found",
                )
            return gateway_model_from_catalog_record(filtered[0])
        page, has_more, next_cursor = paginate(filtered, cursor=cursor, limit=limit)
        return GatewayCatalogListResponse(
            data=[gateway_model_from_catalog_record(record) for record in page],
            has_more=has_more,
            next_cursor=next_cursor,
        )

    return ModelListResponse(data=[model_from_catalog_record(record) for record in records])


@router.get("/models/{model_id:path}", dependencies=[Depends(verify_api_key_or_master_key)])
async def get_model(
    model_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ModelObject:
    """Get details for a specific model."""
    return await load_model_object(db, config, model_id)


@router.get("/vendors", dependencies=[Depends(verify_api_key_or_master_key)], tags=["vendors"])
async def list_vendors(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    cursor: Annotated[str | None, Query(description="Pagination cursor")] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Page size")] = 50,
) -> GatewayVendorListResponse:
    """List execution vendors and the models they can serve."""
    records = await load_catalog_records(db, config)
    vendors = vendor_catalog_from_records(records, config)
    page, has_more, next_cursor = paginate(vendors, cursor=cursor, limit=limit)
    return GatewayVendorListResponse(data=page, has_more=has_more, next_cursor=next_cursor)


@router.get("/vendors/{vendor_id}", dependencies=[Depends(verify_api_key_or_master_key)], tags=["vendors"])
async def get_vendor(
    vendor_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> GatewayVendorResponse:
    """Fetch one execution vendor by id."""
    records = await load_catalog_records(db, config)
    vendors = vendor_catalog_from_records(records, config)
    for vendor in vendors:
        if vendor.vendor == vendor_id:
            return vendor
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Vendor '{vendor_id}' not found",
    )
